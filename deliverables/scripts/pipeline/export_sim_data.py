#!/usr/bin/env python
"""Export the robot model and every simulation state the clips show, as JSON for the renderer.

PyBullet and the project's trained networks compute all states here (FK poses, IK solutions,
collision contacts, classifier probabilities, dataset samples, network activations, agent episodes);
the browser renderer (deliverables/scripts/render/) only draws them. The renderer's own forward
kinematics is the same chain as refine/kinematics.py and is checked against PyBullet with the
"check" poses written into robot.json.

    python deliverables/scripts/pipeline/export_sim_data.py            # every scene
    python deliverables/scripts/pipeline/export_sim_data.py robot ik   # some
Output: $NIK_WORK/sim/*.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BEST_RUN, BEST_RUN_DIR, DATASET_200K, REPO, RESULTS, WORK, read_json  # noqa: E402

import numpy as np  # noqa: E402
import pybullet as p  # noqa: E402
import torch  # noqa: E402

import ik_metrics as M  # noqa: E402
import model_registry as MR  # noqa: E402
import train_ik  # noqa: E402
from refine import lm_solver  # noqa: E402
from refine.kinematics import Chain, robot_chain  # noqa: E402
from robot_config import connect, load_robot  # noqa: E402

OUT = WORK / "sim"
ROBOT = "xarm6"
R = lambda a, n=5: [round(float(v), n) for v in np.asarray(a).ravel()]


class Sim:
    def __init__(self):
        connect(gui=False)
        self.body, self.cfg, lim = load_robot(ROBOT)
        self.lim = np.asarray(lim)
        self.ji, self.ee = self.cfg["joint_indices"], self.cfg["ee_link_index"]
        self.dof = self.cfg["dof"]

    def set(self, q):
        for j, a in zip(self.ji, q):
            p.resetJointState(self.body, j, float(a))

    def links(self, q):
        self.set(q)
        out = {}
        for j in range(p.getNumJoints(self.body)):
            st = p.getLinkState(self.body, j, computeForwardKinematics=True)
            out[str(j)] = [R(st[4], 6), R(st[5], 6)]
        return out

    def ee_pose(self, q):
        self.set(q)
        st = p.getLinkState(self.body, self.ee, computeForwardKinematics=True)
        return np.array(st[4]), np.array(st[5])

    def contacts(self, q):
        """Self-contacts as PyBullet reports them: (linkA, linkB, point on A, point on B, distance)."""
        self.set(q)
        p.performCollisionDetection()
        cps = p.getContactPoints(bodyA=self.body, bodyB=self.body)
        return [{"a": c[3], "b": c[4], "pa": R(c[5]), "pb": R(c[6]), "d": round(float(c[8]), 5)} for c in cps]

    def check(self, q, target_pos=None, target_orn=None):
        pos, orn = self.ee_pose(q)
        cps = self.contacts(q)
        out = {"q": R(q), "ee_pos": R(pos, 6), "ee_orn": R(orn, 6), "self_collision": len(cps) > 0,
               "collision_links": sorted({c["a"] for c in cps} | {c["b"] for c in cps}),
               "limit_violation": bool(M.limit_violations(M.unwrap_to_limits(np.asarray(q)[None], self.lim), self.lim)[0])}
        if target_pos is not None:
            out["pos_err_mm"] = round(float(np.linalg.norm(pos - target_pos) * 1000), 5)
            out["orn_err_deg"] = round(float(M.quat_angle_deg(orn, target_orn)[0]), 5)
        return out


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.json", "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    print(f"  {name}.json  {(OUT / f'{name}.json').stat().st_size / 1024:.0f} KB")


# ------------------------------------------------------------------------------------------------ robot
def export_robot(sim):
    root, rel, axes, limits, rel_ee = robot_chain(ROBOT)
    visuals = []
    for v in p.getVisualShapeData(sim.body):
        fname = v[4].decode(errors="ignore")
        visuals.append({"link": v[1], "mesh": "/" + str(Path(fname).resolve().relative_to(REPO)).replace("\\", "/"),
                        "scale": R(v[3]), "pos": R(v[5]), "orn": R(v[6]), "rgba": R(v[7], 3)})
    rng = np.random.default_rng(3)
    checks = [{"q": R(q), "links": sim.links(q)} for q in
              [np.zeros(6)] + [rng.uniform(sim.lim[:, 0], sim.lim[:, 1]) for _ in range(5)]]
    names = [p.getJointInfo(sim.body, j)[1].decode() for j in sim.ji]
    save("robot", {
        "robot": ROBOT, "dof": sim.dof, "ee_link": sim.ee, "joint_names": names,
        "limits": [R(l) for l in sim.lim], "chain": {"root": R(root, 8), "rel": [R(m, 8) for m in rel],
                                                       "axes": [R(a, 8) for a in axes]},
        "visuals": visuals, "check": checks,
    })


# ------------------------------------------------------------------------------------------------ helpers
def load_eval():
    d = np.load(WORK / "eval_per_target.npz")
    data = np.load(DATASET_200K / "data" / "xarm6_ik_dataset.npz")
    idx = d["dataset_index"]
    return d, data["positions"][idx].astype(float), data["orientations"][idx].astype(float)


def best_models():
    meta = read_json(BEST_RUN_DIR / "meta.json")
    return MR.load_run_models(str(BEST_RUN_DIR), meta, 6, "cpu")


def baseline_models():
    ik = train_ik.IKNet(6)
    ik.load_state_dict(torch.load(RESULTS / "baseline_repro" / "xarm6_ik_net.pt", map_location="cpu"))
    col = train_ik.CollisionNet(6)
    col.load_state_dict(torch.load(RESULTS / "baseline_repro" / "xarm6_collision_net.pt", map_location="cpu"))
    return ik.eval(), col.eval()


def col_prob(cnet, q):
    with torch.no_grad():
        return float(torch.sigmoid(cnet(torch.as_tensor(np.asarray(q, dtype=np.float32)[None]))).item())


def wrap_near(q, ref, lim):
    """Shift each joint by k*360 deg to the copy inside the limits closest to ref (same pose)."""
    out = np.array(q, float)
    for j in range(len(out)):
        c = [out[j] + 2 * np.pi * k for k in range(-2, 3) if lim[j, 0] - 1e-9 <= out[j] + 2 * np.pi * k <= lim[j, 1] + 1e-9]
        if c:
            out[j] = min(c, key=lambda v: abs(v - ref[j]))
    return out


# ------------------------------------------------------------------------------------------------ FK
def export_fk(sim):
    q = np.radians([35.0, -25.0, -70.0, 20.0, 60.0, -15.0])
    pos, orn = sim.ee_pose(q)
    rpy = np.degrees(p.getEulerFromQuaternion(orn))
    joints = []
    for k in range(6):                         # pose after each joint has moved (others still at 0)
        qk = np.zeros(6)
        qk[:k + 1] = q[:k + 1]
        pk, ok = sim.ee_pose(qk)
        joints.append({"q": R(qk), "ee_pos": R(pk), "ee_orn": R(ok)})
    save("fk", {"q": R(q), "q_deg": R(np.degrees(q), 2), "ee_pos": R(pos, 6), "ee_orn": R(orn, 6),
                "ee_rpy_deg": R(rpy, 2), "steps": joints, "home_ee": R(sim.ee_pose(np.zeros(6))[0], 6),
                "check": sim.check(q)})


# ------------------------------------------------------------------------------------------------ IK branches
def _branches(sim, chain, pos, orn):
    cand = lm_solver.random_candidates(chain, pos[None], orn[None], n_random=96, iters=60, rng_seed=5)
    sols = []
    for r in np.argsort(cand["pos_err_mm"][:, 0]):
        if cand["pos_err_mm"][r, 0] > 0.01 or cand["orn_err_deg"][r, 0] > 0.01:
            continue
        q = cand["q"][r, 0]
        if all(np.max(np.abs(M.wrap(q - s))) > 0.35 for s in sols):
            sols.append(q)
    return sols


def _elbow_wrist(sim, q):
    sim.set(q)
    return np.concatenate([np.array(p.getLinkState(sim.body, j, computeForwardKinematics=True)[4]) for j in (3, 5)])


def _diverse(sim, sols, k=4):
    """greedy: k branches whose elbow / wrist positions are furthest apart"""
    if len(sols) < 2:
        return list(range(len(sols))), 0.0
    P = np.array([_elbow_wrist(sim, q) for q in sols])
    pick = [0]
    while len(pick) < min(k, len(sols)):
        d = [min(np.linalg.norm(P[i] - P[j]) for j in pick) if i not in pick else -1 for i in range(len(sols))]
        pick.append(int(np.argmax(d)))
    md = min(np.linalg.norm(P[a] - P[b]) for a in pick for b in pick if a < b)
    return pick, float(md)


def export_ik(sim, n_scan=60):
    """One reachable target, several different joint solutions (IK branches) found by LM from random
    starts; each re-checked with PyBullet. The network + LM, given a seed, returns the one near the seed."""
    d, P, O = load_eval()
    chain = Chain(ROBOT, "cpu")
    best, _ = best_models()
    scores = []
    for k in range(n_scan):
        if not (P[k][0] > 0.05 and 0.3 < P[k][2] < 0.75):
            continue
        sols = _branches(sim, chain, P[k], O[k])
        if len(sols) < 4:
            continue
        pick, md = _diverse(sim, sols)
        scores.append((md, k))
    md, k = max(scores)
    pos, orn, q_true = P[k], O[k], d["q_true"][k]
    sols = _branches(sim, chain, pos, orn)
    pick, _ = _diverse(sim, sols)
    seed = d["seed_warm"][k]
    x = M.make_inputs(pos[None], orn[None], seed[None])
    q_net = M.predict(best, x, 6)[0]
    res = lm_solver.solve(chain, pos[None], orn[None], [q_net[None], seed[None]], ref=seed[None], iters=30)
    q_fin = wrap_near(res["q"][0], seed, sim.lim)
    shown = [sols[i] for i in pick]
    match = [float(np.max(np.abs(M.wrap(q_fin - s)))) for s in shown]
    if min(match) > 0.05:                        # make sure the chosen answer is one of the shown branches
        shown[-1] = q_fin
        match[-1] = 0.0
    sel = int(np.argmin(match))
    shown = [wrap_near(q, seed, sim.lim) for q in shown]
    out = {"k": int(k), "target_pos": R(pos), "target_orn": R(orn),
           "target_rpy_deg": R(np.degrees(p.getEulerFromQuaternion(orn)), 2),
           "seed": R(wrap_near(seed, seed, sim.lim)), "q_true": R(q_true),
           "network": sim.check(wrap_near(q_net, seed, sim.lim), pos, orn), "final": sim.check(q_fin, pos, orn),
           "branches": [sim.check(q, pos, orn) for q in shown], "selected": sel, "n_branches_found": len(sols),
           "min_elbow_wrist_separation_m": md}
    print(f"    IK target {k}: {len(sols)} distinct solutions, showing {len(shown)} (min separation {md:.2f} m), selected {sel}")
    save("ik", out)


def scan_ik_targets(sim, n=60):
    d, P, O = load_eval()
    chain = Chain(ROBOT, "cpu")
    for k in range(n):
        cand = lm_solver.random_candidates(chain, P[k:k + 1], O[k:k + 1], n_random=96, iters=60, rng_seed=5)
        sols = []
        for r in np.argsort(cand["pos_err_mm"][:, 0]):
            if cand["pos_err_mm"][r, 0] > 0.01 or cand["orn_err_deg"][r, 0] > 0.01:
                continue
            q = cand["q"][r, 0]
            if all(np.max(np.abs(M.wrap(q - s))) > 0.35 for s in sols):
                sols.append(q)
        cols = [sim.check(q)["self_collision"] for q in sols]
        print(k, len(sols), sum(cols), np.round(P[k], 2))


# ------------------------------------------------------------------------------------------------ dataset
def export_dataset(sim):
    """Rows exactly as generate_dataset.py produced them (the 200 000-row Studio dataset, seed 0):
    the first uniform samples, then near-limit samples, plus a large point cloud for the workspace."""
    data = np.load(DATASET_200K / "data" / "xarm6_ik_dataset.npz")
    J, Pp, O, L = data["joints"], data["positions"], data["orientations"], data["collision_labels"]
    n_uniform = int(200000 * 0.7)
    rows = list(range(0, 90)) + list(range(n_uniform, n_uniform + 30))
    samples = []
    for i in rows:
        cps = sim.contacts(J[i])
        samples.append({"i": i, "q": R(J[i]), "pos": R(Pp[i]), "orn": R(O[i]), "label": int(L[i]),
                        "phase": "uniform" if i < n_uniform else "near-limit",
                        "contacts": cps[:4], "pb_label": int(len(cps) > 0)})
    rng = np.random.default_rng(11)
    cloud = np.sort(rng.choice(len(J), 7000, replace=False))
    save("dataset", {"n_total": int(len(J)), "n_colliding": int(L.sum()), "n_uniform": n_uniform,
                     "samples": samples, "cloud_pos": R(Pp[cloud], 4), "cloud_label": [int(v) for v in L[cloud]],
                     "agree": float(np.mean([s["label"] == s["pb_label"] for s in samples]))})


# ------------------------------------------------------------------------------------------------ network
def export_network(sim):
    """Real activations of the baseline network (train_ik.IKNet 13-256-256-128-12) for one test target."""
    d, P, O = load_eval()
    base, _ = baseline_models()
    best, _ = best_models()
    k = 3
    x = M.make_inputs(P[k:k + 1], O[k:k + 1], d["seed_warm"][k:k + 1])
    acts, h = [], torch.as_tensor(x)
    with torch.no_grad():
        for layer in base.net:
            h = layer(h)
            if isinstance(layer, torch.nn.ReLU):
                acts.append(R(h[0], 4))
        raw = h[0].numpy()
    q_base = np.arctan2(raw[:6], raw[6:])
    q_best = M.predict(best, x, 6)[0]
    save("network", {"x": R(x[0], 5), "acts": acts, "raw": R(raw, 5), "q_pred": R(q_base),
                     "q_pred_deg": R(np.degrees(q_base), 2), "q_true": R(d["q_true"][k]),
                     "q_true_deg": R(np.degrees(M.wrap(d["q_true"][k])), 2),
                     "seed_deg": R(np.degrees(d["seed_warm"][k]), 2),
                     "baseline": sim.check(q_base, P[k], O[k]),
                     "best": sim.check(wrap_near(q_best, d["seed_warm"][k], sim.lim), P[k], O[k]),
                     "params": {"baseline_mlp": 103820, "residual_mlp": 2050316}})


# ------------------------------------------------------------------------------------------------ collision
def export_collision(sim):
    """A motion from a free pose into a self-colliding pose: PyBullet contact vs classifier probability.
    The colliding end pose is a real dataset row (wrist folding into the base / upper arm); the start
    pose moves one joint back until PyBullet reports no contact."""
    _, cnet = best_models()
    data = np.load(DATASET_200K / "data" / "xarm6_ik_dataset.npz")
    J, L = data["joints"], data["collision_labels"]
    best = None
    for i in np.nonzero(L == 1)[0][:3000]:
        cps = sim.contacts(J[i])
        pairs = {(c["a"], c["b"]) for c in cps}
        if not any(abs(a - b) >= 3 for a, b in pairs):          # links far apart in the chain: a clear fold
            continue
        q_col = J[i].astype(float)
        for j in (2, 4, 1):                                      # elbow, wrist pitch, shoulder
            for sgn in (1, -1):
                for d in np.radians(np.arange(20, 75, 5)):
                    q = q_col.copy()
                    q[j] += sgn * d
                    if not (sim.lim[j, 0] <= q[j] <= sim.lim[j, 1]) or sim.contacts(q):
                        continue
                    if col_prob(cnet, q) > 0.08:                 # a start the classifier also calls safe
                        continue
                    pe = sim.ee_pose(q)[0]
                    if pe[2] < 0.12:                             # keep the wrist above the floor for the camera
                        continue
                    # the whole way back to q_col must be free until the first contact, then stay in contact
                    flags = [bool(sim.contacts(q + u * (q_col - q))) for u in np.linspace(0, 1, 25)]
                    k = flags.index(True) if True in flags else 99
                    if k > 6 and all(flags[k:]):
                        score = abs(d - np.radians(40))
                        if best is None or score < best[0]:
                            best = (score, q.copy(), q_col.copy(), int(i), j)
                    break
        if best is not None and best[0] <= np.radians(5):
            break
    _, q_safe, q_col, row, joint = best
    path = []
    for t in np.linspace(0, 1, 61):
        q = q_safe + t * (q_col - q_safe)
        cps = sim.contacts(q)
        path.append({"t": round(float(t), 4), "q": R(q), "contact": len(cps) > 0, "p": round(col_prob(cnet, q), 4),
                     "links": sorted({c["a"] for c in cps} | {c["b"] for c in cps}),
                     "points": [c["pa"] for c in cps[:6]]})
    first = next(i for i, s_ in enumerate(path) if s_["contact"])
    free_rows = np.nonzero(L == 0)[0][:500]
    col_rows = np.nonzero(L == 1)[0][:500]
    save("collision", {"q_safe": R(q_safe), "q_col": R(q_col), "dataset_row": row, "moved_joint": joint + 1,
                       "path": path, "first_contact": first,
                       "p_safe": col_prob(cnet, q_safe), "p_col": col_prob(cnet, q_col),
                       "classifier": "best run CollisionNet 6-64-64-1",
                       "mean_p_free": float(np.mean([col_prob(cnet, J[i]) for i in free_rows])),
                       "mean_p_colliding": float(np.mean([col_prob(cnet, J[i]) for i in col_rows])),
                       "weight": 0.05})


# ------------------------------------------------------------------------------------------------ training
def export_training(sim):
    import csv
    hist = list(csv.DictReader(open(RESULTS / "training_history.csv")))
    cases = read_json(RESULTS / "baseline_repro" / "case_predictions.json")["cases"]
    meta = read_json(BEST_RUN_DIR / "meta.json")
    best_ep = [e for e in meta["epochs"] if e["phase"] == "ik"]
    best_col = [e for e in meta["epochs"] if e["phase"] == "collision"]
    c = cases[2]                                # largest real improvement over the 60 epochs
    lim = sim.lim
    save("training", {
        "baseline": [{k: (float(v) if v not in ("", None) and k != "phase" else v) for k, v in h.items()} for h in hist],
        "best": [{"epoch": e["epoch"], "train_loss": e["train_loss"], "val_mse": e["val_mse"],
                  "collision_penalty": e["collision_penalty"], "val_pos_mean_mm": e["acc"]["val"]["pos_mean_mm"],
                  "val_within_50": e["acc"]["val"]["acc_pos"]["50"], "secs": e["secs"]} for e in best_ep],
        "best_collision": [{"epoch": e["epoch"], "val_acc": e["val_acc"], "train_loss": e["train_loss"]} for e in best_col],
        "case": {"target_pos": c["target_pos"], "target_orn": c["target_orn"], "q_true": R(wrap_near(c["q_true"], c["q_true"], lim)),
                 "epochs": [{"epoch": e["epoch"], "q": R(wrap_near(e["q_pred"], c["q_true"], lim)),
                             "pos_err_mm": e["pos_err_mm"], "orn_err_deg": e["orn_err_deg"]} for e in c["epochs"]]},
        "best_run": {"id": BEST_RUN, "gpu": meta.get("gpu"), "train_secs": meta.get("train_secs"),
                     "test_pos_mean_mm": meta["metrics"]["test"]["pos_mean_mm"]},
    })


# ------------------------------------------------------------------------------------------------ inference
def export_inference(sim, picks=None):
    d, P, O = load_eval()
    if picks is None:                          # clearly visible network miss, target in view, three different regions
        pe = d["residual_mlp_warm_pe"]
        cand = [k for k in range(len(pe)) if 25 < pe[k] < 60 and P[k][0] > 0.12 and 0.3 < P[k][2] < 0.7]
        picks = []
        for k in cand:
            if all(np.linalg.norm(P[k] - P[j]) > 0.35 for j in picks):
                picks.append(k)
            if len(picks) == 3:
                break
    cases = []
    for k in picks:
        seed = d["seed_warm"][k]
        qn = wrap_near(d["residual_mlp_warm_q"][k], seed, sim.lim)
        qh = wrap_near(d["hybrid_lm_warm_q"][k], seed, sim.lim)
        cases.append({"k": int(k), "target_pos": R(P[k]), "target_orn": R(O[k]), "seed": R(wrap_near(seed, seed, sim.lim)),
                      "x": R(np.concatenate([P[k], O[k], seed]), 4),
                      "network": sim.check(qn, P[k], O[k]), "hybrid": sim.check(qh, P[k], O[k])})
        print(f"    case {k}: network {cases[-1]['network']['pos_err_mm']:.1f} mm -> hybrid {cases[-1]['hybrid']['pos_err_mm']:.5f} mm")
    import csv
    rows = {f"{r['method']}|{r['seed_mode']}": r for r in csv.DictReader(open(RESULTS / "evaluation_metrics.csv"))}
    timing = {"network_ms": float(rows["residual_mlp|warm"]["time_ms_mean"]),
              "hybrid_ms": float(rows["hybrid_lm|warm"]["time_ms_mean"])}
    save("inference", {"cases": cases, "thresholds": {"pos_mm": 1.0, "orn_deg": 1.0}, "timing": timing})


# ------------------------------------------------------------------------------------------------ comparison
def export_comparison(sim, picks=None):
    """Identical targets, cold start from the home pose (zero seed): network alone, PyBullet DLS, network + LM."""
    d, P, O = load_eval()
    mode = "zero"
    pe_dls, col_dls, lim_dls = d[f"classical_dls_{mode}_pe"], d[f"classical_dls_{mode}_col"], d[f"classical_dls_{mode}_lim"]
    if picks is None:
        ok = np.nonzero((pe_dls < 1) & ~col_dls & ~lim_dls & (d[f"residual_mlp_{mode}_pe"] < 120))[0]
        sc = np.nonzero(col_dls & (pe_dls < 5))[0]
        lv = np.nonzero(lim_dls & ~col_dls & (pe_dls < 5))[0]
        picks = [int(ok[0]), int(sc[0]), int(lv[0])]
    cases = []
    for k in picks:
        z = np.zeros(6)
        cases.append({"k": int(k), "target_pos": R(P[k]), "target_orn": R(O[k]),
                      "network": sim.check(d[f"residual_mlp_{mode}_q"][k], P[k], O[k]),
                      "dls": sim.check(d[f"classical_dls_{mode}_q"][k], P[k], O[k]),
                      "hybrid": sim.check(wrap_near(d[f"hybrid_lm_{mode}_q"][k], z, sim.lim), P[k], O[k])})
    import csv
    rows = list(csv.DictReader(open(RESULTS / "evaluation_metrics.csv")))
    agg = {f"{r['method']}|{r['seed_mode']}": {k: r[k] for k in r if k not in ("notes",)} for r in rows}
    save("comparison", {"seed_mode": mode, "cases": cases, "aggregate": agg})
    for c in cases:
        print("    ", c["k"], {m: (round(c[m]["pos_err_mm"], 2), c[m]["self_collision"], c[m]["limit_violation"])
                              for m in ("network", "dls", "hybrid")})


# ------------------------------------------------------------------------------------------------ level 2
def export_level2(sim, policy=None, scan=120):
    """Recorded Level 2 episodes on identical seeds (scenario "everything at once"): the PPO agent, the
    straight joint-space path and the potential field. Picks the first evaluation seed where an obstacle
    appears, a joint slips, the straight path collides, the potential field arrives but takes at least 1.5x
    as many steps, and the agent arrives - the episode the narration describes."""
    from level2.env import ReachAvoidEnv
    from level2.evaluate import SCENARIOS, run_episode
    from level2.ppo import NumpyPolicy
    policy = policy or str(WORK / "level2_repro" / "policy.pt")
    pol = NumpyPolicy(policy)
    env = ReachAvoidEnv("xarm6", seed=0, **dict({"base": "straight"}, **(pol.meta.get("env_cfg") or {})))
    sc = SCENARIOS["combined"]
    pick = None
    for sd in range(50_000, 50_000 + scan):           # the evaluation seeds (level2/evaluate.py default 50 000)
        a = run_episode(env, pol, sd, sc)
        if a["outcome"] != "success":
            continue
        b = run_episode(env, None, sd, sc)
        types = {e["type"] for e in a["events"]}
        if b["outcome"] != "collision" or not {"obstacle_appeared", "joint_slip"} <= types or not 45 <= a["steps"] <= 110:
            continue
        c = run_episode(env, "apf", sd, sc)
        if c["outcome"] == "success" and c["steps"] >= 1.5 * a["steps"]:
            pick = sd
            break
    if pick is None:
        sys.exit(f"no episode matching the narration in seeds 50000-{50_000 + scan - 1}; raise scan")
    eps = {}
    for label, ctrl in (("policy", pol), ("baseline", None), ("apf", "apf")):
        ep = run_episode(env, ctrl, pick, sc, record=True)
        eps[label] = {"outcome": ep["outcome"], "collision": ep["collision"], "steps": ep["steps"],
                      "min_clearance_mm": round(1000 * ep["min_clearance_m"], 1), "events": ep["events"],
                      "frames": [{"t": fr["t"], "q": fr["q"], "obstacles": fr["obstacles"],
                                  "clearance": fr["clearance"]} for fr in ep["frames"]]}
    env.reset(seed=pick, scenario=sc)
    goal = {"q": R(env.q_goal), "ee": R(env.ee_goal)}
    env.close()
    team = read_json(REPO / "media" / "data" / "l2_20261013-185525_xarm6_meta_n200.json")["evaluation"]
    repro_path = RESULTS / "level2_repro_eval.json"
    save("level2", {"seed": pick, "scenario": "combined", "goal": goal, "episodes": eps,
                    "policy": {"path": Path(policy).name, "params": pol.n_params},
                    "team_eval": {k: {c: {m: v[c][m] for m in ("success_pct", "collision_pct", "timeout_pct")} for c in v}
                                  for k, v in team["scenarios"].items()},
                    "team_params": team["policy_params"], "team_decision_us": team["decision_us"],
                    "repro_eval": read_json(repro_path) if repro_path.exists() else None})
    print(f"    seed {pick}: " + ", ".join(f"{k} {v['outcome']} ({v['steps']} steps)" for k, v in eps.items()))


SCENES = {"robot": export_robot, "fk": export_fk, "ik": export_ik, "dataset": export_dataset,
          "network": export_network, "collision": export_collision, "training": export_training,
          "inference": export_inference, "comparison": export_comparison,
          "level2": export_level2}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("--scan-ik", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(1)
    sim = Sim()
    if a.scan_ik:
        scan_ik_targets(sim)
        sys.exit()
    for name in a.scenes or list(SCENES):
        print(name)
        SCENES[name](sim)
