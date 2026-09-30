"""
Shared accuracy metrics for the IK network (used by the train / eval runners
and by the server's inference endpoint).

Error is measured the same way evaluate.py measures it: predicted joint
angles -> PyBullet forward kinematics -> compare the end-effector pose that is
actually reached with the requested target pose.

"Accuracy" = percentage of predictions whose reached end-effector lands
within a threshold of the target (position in mm, orientation in degrees).
All thresholds are stored so the UI can switch between them.
"""

import numpy as np
import pybullet as p

POS_THRESH_MM = (1, 5, 10, 25, 50, 100)
ORN_THRESH_DEG = (1, 5, 15, 30)
SEED_NOISE_STD = 0.3   # train_ik.load_dataset's default seed noise
SEED_MODES = {
    "noisy": "true joints + N(0, 0.3) (warm start, as in training)",
    "zero": "zero pose (cold start - no prior knowledge of the answer)",
    "random": "random pose within the joint limits (cold start)",
}
# numerical refinement defaults (refine/lm_solver.py)
REFINE_DEFAULTS = {"iters": 30, "restarts": 16, "use_seed": True, "tol_mm": 1.0, "tol_deg": 1.0}


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def quat_angle_deg(q1, q2):
    q1 = np.atleast_2d(q1)
    q2 = np.atleast_2d(q2)
    dot = np.clip(np.abs(np.sum(q1 * q2, axis=1)), -1.0, 1.0)
    return np.degrees(2 * np.arccos(dot))


def decode(raw, dof):
    """[sin..., cos...] network output -> joint angles (same as IKNet.decode_angles)."""
    raw = np.asarray(raw)
    return np.arctan2(raw[..., :dof], raw[..., dof:])


def make_inputs(positions, orientations, seeds):
    return np.concatenate([positions, orientations, seeds], axis=1).astype(np.float32)


def noisy_seeds(joints, rng_seed):
    """Seed configurations built with train_ik.load_dataset's recipe
    (true joints + N(0, 0.3)), for held-out rows that train_ik never saw."""
    rng = np.random.default_rng(rng_seed)
    return (joints + rng.normal(0, SEED_NOISE_STD, size=joints.shape)).astype(np.float32)


def make_seeds(joints, mode, limits, rng_seed):
    """Seeds for an evaluation: see SEED_MODES."""
    if mode == "noisy":
        return noisy_seeds(joints, rng_seed)
    if mode == "zero":
        return np.zeros_like(joints, dtype=np.float32)
    if mode == "random":
        lim = np.asarray(limits, dtype=float)
        rng = np.random.default_rng(rng_seed)
        return rng.uniform(lim[:, 0], lim[:, 1], size=joints.shape).astype(np.float32)
    raise ValueError(f"unknown seed mode {mode!r}")


def unwrap_to_limits(Q, limits):
    """Shift angles by k * 360 deg into the joint limits where possible (same pose).
    Networks decode angles to -180..180 deg; e.g. xArm6 joint 3 spans -225..+11 deg."""
    Q = np.array(Q, dtype=float, copy=True)
    lim = np.asarray(limits, dtype=float)
    for j, (lo, hi) in enumerate(lim):
        for k in (0, -1, 1, -2, 2):
            c = Q[:, j] + 2 * np.pi * k
            bad = (Q[:, j] < lo - 1e-3) | (Q[:, j] > hi + 1e-3)
            fix = bad & (c >= lo - 1e-3) & (c <= hi + 1e-3)
            Q[fix, j] = c[fix]
    return Q


def fk_batch(body, joint_indices, ee_link, Q, collision=False):
    n = len(Q)
    pos = np.zeros((n, 3))
    orn = np.zeros((n, 4))
    col = np.zeros(n, dtype=bool) if collision else None
    for i, q in enumerate(Q):
        for idx, a in zip(joint_indices, q):
            p.resetJointState(body, idx, float(a))
        st = p.getLinkState(body, ee_link, computeForwardKinematics=True)
        pos[i] = st[4]
        orn[i] = st[5]
        if collision:
            p.performCollisionDetection()
            col[i] = len(p.getContactPoints(bodyA=body, bodyB=body)) > 0
    return pos, orn, col


def limit_violations(Q, limits):
    limits = np.asarray(limits)
    return np.any((Q < limits[:, 0] - 1e-3) | (Q > limits[:, 1] + 1e-3), axis=1)


def tkey(t):
    """Threshold -> dict key ("50", "12.5"); matches String(+t) on the JS side."""
    return f"{float(t):g}"


def summarize(pos_err_m, orn_err_deg, joint_err_rad=None, colliding=None, limit_viol=None, extra_pos_mm=()):
    """extra_pos_mm: user-set position tolerances reported next to the presets.
    "success" = strict success rate per "<mm>|<deg>" pair: position AND orientation
    within tolerance AND inside the joint limits AND collision-free (the last two
    whenever they are known)."""
    pe = np.asarray(pos_err_m, dtype=float) * 1000.0
    oe = np.asarray(orn_err_deg, dtype=float)
    if len(pe) == 0:
        return {"n": 0}
    pos_t = sorted({float(t) for t in POS_THRESH_MM} | {float(t) for t in extra_pos_mm if t})
    valid = np.ones(len(pe), dtype=bool)
    if colliding is not None:
        valid &= ~np.asarray(colliding, dtype=bool)
    if limit_viol is not None:
        valid &= ~np.asarray(limit_viol, dtype=bool)
    s = {
        "n": int(len(pe)),
        "pos_mean_mm": float(pe.mean()),
        "pos_median_mm": float(np.median(pe)),
        "pos_p90_mm": float(np.percentile(pe, 90)),
        "pos_p99_mm": float(np.percentile(pe, 99)),
        "orn_mean_deg": float(oe.mean()),
        "orn_median_deg": float(np.median(oe)),
        "acc_pos": {tkey(t): float((pe <= t).mean() * 100) for t in pos_t},
        "acc_orn": {tkey(t): float((oe <= t).mean() * 100) for t in ORN_THRESH_DEG},
        "success": {f"{tkey(pt)}|{tkey(ot)}": float(((pe <= pt) & (oe <= ot) & valid).mean() * 100)
                    for pt in pos_t for ot in ORN_THRESH_DEG},
        "success_checks": [k for k, v in (("collision", colliding), ("limits", limit_viol)) if v is not None],
    }
    if joint_err_rad is not None:
        s["joint_mae_deg"] = float(np.degrees(np.abs(wrap(np.asarray(joint_err_rad)))).mean())
    if colliding is not None:
        s["self_collision_pct"] = float(np.mean(colliding) * 100)
    if limit_viol is not None:
        s["limit_violation_pct"] = float(np.mean(limit_viol) * 100)
    return s


def predict(model, X, dof, batch=8192):
    """Network output -> joint angles, for any model following the IKNet contract.
    Runs in eval mode (dropout off) and restores the previous mode afterwards,
    so it is safe to call in the middle of training. During training the UI's
    observed model exposes raw_forward() - the plain forward pass without the
    training observer - which is used when present."""
    import torch
    device = next(model.parameters()).device
    fwd = getattr(model, "raw_forward", None) or model
    was_training = model.training
    model.eval()
    outs = []
    try:
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=device)
                outs.append(fwd(xb).cpu().numpy())
    finally:
        model.train(was_training)
    return decode(np.concatenate(outs) if outs else np.zeros((0, 2 * dof)), dof)


def evaluate_ik(model, body, cfg, limits, X, q_true, collision=True, extra_pos_mm=()):
    """Full IK evaluation of `model` on inputs X (target pose + seed) whose
    ground-truth joint solution is q_true. Returns (summary, per-sample)."""
    dof = cfg["dof"]
    q_pred = predict(model, X, dof)
    pos, orn, col = fk_batch(body, cfg["joint_indices"], cfg["ee_link_index"], q_pred, collision)
    pos_err = np.linalg.norm(pos - X[:, :3], axis=1)
    orn_err = quat_angle_deg(orn, X[:, 3:7])
    lim = limit_violations(unwrap_to_limits(q_pred, limits), limits)   # +-360 deg is the same pose
    summary = summarize(pos_err, orn_err, q_pred - q_true, col, lim, extra_pos_mm=extra_pos_mm)
    return summary, {"q_pred": q_pred, "pos_err": pos_err, "orn_err": orn_err, "colliding": col, "limit_viol": lim}


_CHAINS = {}


def chain_for(robot, device):
    """Cached refine.kinematics.Chain (float64) for a robot on a device."""
    key = (robot, str(device))
    if key not in _CHAINS:
        from refine.kinematics import Chain
        _CHAINS[key] = Chain(robot, device)
    return _CHAINS[key]


def refine_ik(robot, body, cfg, limits, X, q_start, settings=None, device=None, extra_pos_mm=(), q_true=None,
              collision=True):
    """Network + numerical refinement: refine/lm_solver.solve from the network's
    joints (and the seed, if settings["use_seed"]), random restarts for targets
    still outside tolerance, then an independent PyBullet check of the result
    (FK error, joint limits, self-collision). Returns (summary, per-sample)."""
    import time
    import torch
    from refine import lm_solver
    st = dict(REFINE_DEFAULTS, **(settings or {}))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    chain = chain_for(robot, device)
    dof = cfg["dof"]
    seeds = X[:, 7:7 + dof]
    starts = [q_start] + ([seeds] if st["use_seed"] else [])
    t0 = time.perf_counter()
    with torch.no_grad():
        res = lm_solver.solve(chain, X[:, :3], X[:, 3:7], starts, ref=seeds, iters=int(st["iters"]),
                              restarts=int(st["restarts"]), tol_mm=float(st["tol_mm"]), tol_deg=float(st["tol_deg"]))
    q = res["q"].copy()
    winner = res["winner"].copy()
    pos, orn, col = fk_batch(body, cfg["joint_indices"], cfg["ee_link_index"], q, collision)
    pos_err = np.linalg.norm(pos - X[:, :3], axis=1)
    orn_err = quat_angle_deg(orn, X[:, 3:7])
    lim = limit_violations(q, limits)
    if collision and st["restarts"] > 0:
        # The solver cannot see self-collisions: where the best solution collides (or failed),
        # look for another solution (IK branch) from fresh random starts and keep the first one
        # PyBullet confirms as within tolerance, inside the limits and collision-free.
        for rnd, n_random in enumerate((max(32, int(st["restarts"])), 128)):   # 2nd round: rare stubborn targets
            fail = np.nonzero((pos_err * 1000 > st["tol_mm"]) | (orn_err > st["tol_deg"]) | col | lim)[0]
            if not len(fail):
                break
            with torch.no_grad():
                cand = lm_solver.random_candidates(chain, X[fail, :3], X[fail, 3:7], n_random=n_random,
                                                   iters=max(50, int(st["iters"])), rng_seed=1 + rnd)
            conv = (cand["pos_err_mm"] <= st["tol_mm"]) & (cand["orn_err_deg"] <= st["tol_deg"])
            rr, cc = np.nonzero(conv)
            if len(rr):
                Qc = cand["q"][rr, cc]
                pc, oc, colc = fk_batch(body, cfg["joint_indices"], cfg["ee_link_index"], Qc, True)
                pec = np.linalg.norm(pc - X[fail[cc], :3], axis=1)
                oec = quat_angle_deg(oc, X[fail[cc], 3:7])
                good = (pec * 1000 <= st["tol_mm"]) & (oec <= st["tol_deg"]) & ~colc & ~limit_violations(Qc, limits)
                done = set()
                for k in np.nonzero(good)[0]:
                    t = fail[cc[k]]
                    if t in done:
                        continue
                    done.add(t)
                    q[t], pos_err[t], orn_err[t], col[t], lim[t], winner[t] = Qc[k], pec[k], oec[k], False, False, RETRY
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    summary = summarize(pos_err, orn_err, None if q_true is None else q - q_true, col, lim, extra_pos_mm=extra_pos_mm)
    summary["refine"] = {**st, "device": str(device), "ms_per_target": secs * 1000 / max(1, len(q)),
                         "won_by": won_by(winner, len(starts), st["use_seed"])}
    return summary, {"q_ref": q, "pos_err_ref": pos_err, "orn_err_ref": orn_err, "colliding_ref": col,
                     "limit_viol_ref": lim, "winner": winner}


RETRY = 999      # winner code: solution found by the collision-free retry


def won_by(winner, n_starts, use_seed):
    w = np.asarray(winner)
    return {"network": float((w == 0).mean() * 100),
            "seed": float((w == 1).mean() * 100) if use_seed else 0.0,
            "random restart": float(((w >= n_starts) & (w != RETRY)).mean() * 100),
            "collision-free retry": float((w == RETRY).mean() * 100)}


def collision_metrics(col_model, joints, labels, batch=8192):
    """Accuracy / precision / recall of the collision classifier (threshold 0.5)."""
    import torch
    if len(joints) == 0:
        return {"n": 0}
    device = next(col_model.parameters()).device
    logits = []
    with torch.no_grad():
        for i in range(0, len(joints), batch):
            qb = torch.as_tensor(joints[i:i + batch], dtype=torch.float32, device=device)
            logits.append(col_model.net(qb).squeeze(-1).cpu().numpy())
    pred = np.concatenate(logits) > 0
    lab = np.asarray(labels).astype(bool)
    tp = int((pred & lab).sum())
    fp = int((pred & ~lab).sum())
    fn = int((~pred & lab).sum())
    return {
        "n": int(len(lab)), "n_colliding": int(lab.sum()),
        "accuracy": float((pred == lab).mean() * 100),
        "precision": float(tp / (tp + fp) * 100) if tp + fp else None,
        "recall": float(tp / (tp + fn) * 100) if tp + fn else None,
    }


def infer_generation_seed(d, max_seed=1000):
    """generate_dataset.generate() draws the first sample as
    [rng.uniform(lo, hi) for each joint] with rng = default_rng(seed), so the
    seed of an existing dataset can be recovered from its first row."""
    lims = np.asarray(d["joint_limits"], dtype=float)
    q0 = np.asarray(d["joints"][0], dtype=float)
    for s in range(max_seed):
        rng = np.random.default_rng(s)
        cand = np.array([rng.uniform(lo, hi) for lo, hi in lims])
        if np.allclose(cand, q0, atol=1e-4):
            return s
    return None
