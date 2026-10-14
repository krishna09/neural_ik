#!/usr/bin/env python
"""Level 1 accuracy / speed / safety comparison of every IK method in the project, on IDENTICAL targets.

Targets: held-out test poses of the best recorded run (ui/workspace/library/runs/20260928-174703_xarm6,
200 000-sample dataset ds_20260928-163059_xarm6_n200000_s0). Its test split was never seen in training.

  A) the best network on its whole collision-free test split (25 090 targets, the run's own warm seeds)
     - re-checks the recorded run metrics on this machine;
  B) every method on a common subset of N targets (default 2 000), with three seed modes:
       warm   = true joints + N(0, 0.3 rad)   (training recipe; the run's own test seeds)
       zero   = zero pose ("robot at home")   (cold start)
       random = uniform random pose in limits (cold start)
     The subset only uses dataset rows >= 14 000: rows below that coincide with the baseline
     reproduction's 20 000-sample training set (same generator, same seed 0), so the baseline
     network is evaluated on unseen poses too.

Methods
  baseline_mlp     train_ik.IKNet 256-256-128 (plan architecture), retrained by run_level1_baseline.py
  residual_mlp     modelset/custom/model_set.py, 2.05 M parameters (best recorded run)
  classical_dls    PyBullet calculateInverseKinematics, 100 iterations, seed as currentPositions (evaluate.py)
  hybrid_dls       residual_mlp + 5 DLS iterations (evaluate.py's hybrid recipe)
  baseline_hybrid  baseline_mlp + 5 DLS iterations (the original Level 1 plan hybrid)
  hybrid_lm        residual_mlp + Levenberg-Marquardt refinement (refine/lm_solver.py via
                   ui/runners/ik_metrics.refine_ik, default settings: 30 iterations, 16 random restarts,
                   1 mm / 1 deg tolerance, collision-free retry) - the project's final Level 1 method
  lm_only          the same LM solver, restarts and retry, but started from the seed alone (no network):
                   isolates what the network contributes

Every answer is checked independently with PyBullet: FK pose error, joint limits, self-collision.
Joint-limit violations are counted after shifting angles by +-360 deg into the joint range where
possible (same pose; atan2 decodes to -180..180 deg while xArm6 joint 3 spans -225..+11 deg).

Outputs: deliverables/results/evaluation_metrics.csv, experiment_configuration.json,
         eval_summary.json, demo_cases.json ; per-target arrays in $NIK_WORK/eval_per_target.npz

    python deliverables/scripts/pipeline/evaluate_methods.py --n 2000
"""
import argparse
import csv
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BEST_RUN, BEST_RUN_DIR, DATASET_200K, REPO, RESULTS, WORK, read_json, write_json  # noqa: E402

import numpy as np  # noqa: E402
import pybullet as p  # noqa: E402
import torch  # noqa: E402

import ik_metrics as M  # noqa: E402  (ui/runners)
import model_registry as MR  # noqa: E402
import train_ik  # noqa: E402
from robot_config import connect, load_robot  # noqa: E402

ROBOT = "xarm6"
DEG = np.pi / 180


def fk_check(body, cfg, Q, X, limits):
    pos, orn, col = M.fk_batch(body, cfg["joint_indices"], cfg["ee_link_index"], Q, collision=True)
    pe = np.linalg.norm(pos - X[:, :3], axis=1) * 1000
    oe = M.quat_angle_deg(orn, X[:, 3:7])
    lim = M.limit_violations(M.unwrap_to_limits(Q, limits), limits)
    return pe, oe, col, lim


def dls(body, cfg, X, starts, iters):
    """PyBullet's classical damped-least-squares IK from each start (as evaluate.py does it)."""
    dof = cfg["dof"]
    out = np.zeros((len(X), dof))
    times = np.zeros(len(X))
    for i, (x, q0) in enumerate(zip(X, starts)):
        for j, a in zip(cfg["joint_indices"], q0):
            p.resetJointState(body, j, float(a))
        t0 = time.perf_counter()
        q = p.calculateInverseKinematics(body, cfg["ee_link_index"], x[:3].tolist(), x[3:7].tolist(),
                                         currentPositions=[float(v) for v in q0],
                                         maxNumIterations=iters, residualThreshold=1e-5)
        times[i] = time.perf_counter() - t0
        out[i] = q[:dof]
    return out, times


def net_single_latency(model, X, reps=300):
    """Mean wall time of one forward pass + atan2 decode for a single target (batch of 1), CPU."""
    xs = torch.as_tensor(X[:reps], dtype=torch.float32)
    with torch.no_grad():
        for k in range(20):
            model.decode_angles(model(xs[k:k + 1]))
        t = []
        for k in range(len(xs)):
            t0 = time.perf_counter()
            model.decode_angles(model(xs[k:k + 1]))
            t.append(time.perf_counter() - t0)
    return float(np.mean(t)), float(np.median(t))


def summary_row(method, seed_mode, pe, oe, col, lim, ms_mean, ms_median, notes="", won=None):
    ok = ~col & ~lim
    s = lambda pt, ot: float(((pe <= pt) & (oe <= ot) & ok).mean() * 100)
    return {
        "method": method, "seed_mode": seed_mode, "n_targets": int(len(pe)),
        "pos_mean_mm": float(pe.mean()), "pos_median_mm": float(np.median(pe)), "pos_p95_mm": float(np.percentile(pe, 95)),
        "orn_mean_deg": float(oe.mean()), "orn_median_deg": float(np.median(oe)), "orn_p95_deg": float(np.percentile(oe, 95)),
        "self_collision_pct": float(col.mean() * 100), "limit_violation_pct": float(lim.mean() * 100),
        "within_1mm_pct": float((pe <= 1).mean() * 100), "within_10mm_pct": float((pe <= 10).mean() * 100),
        "within_50mm_pct": float((pe <= 50).mean() * 100),
        "success_1mm_1deg_pct": s(1, 1), "success_10mm_5deg_pct": s(10, 5), "success_50mm_15deg_pct": s(50, 15),
        "time_ms_mean": ms_mean, "time_ms_median": ms_median,
        "won_network_pct": won["network"] if won else "", "won_seed_pct": won["seed"] if won else "",
        "won_restart_pct": won["random restart"] if won else "", "won_retry_pct": won["collision-free retry"] if won else "",
        "notes": notes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--subset-seed", type=int, default=2026)
    ap.add_argument("--timing-n", type=int, default=200, help="targets timed one by one for the LM hybrid")
    a = ap.parse_args()
    torch.set_num_threads(1)                 # single-core timings (inference is per target)
    t_start = time.time()

    meta = read_json(BEST_RUN_DIR / "meta.json")
    data = np.load(DATASET_200K / "data" / "xarm6_ik_dataset.npz")
    tr = np.load(BEST_RUN_DIR / "test_results.npz")
    test_idx, warm_all = tr["index"], tr["seed"]
    J, P, O = data["joints"], data["positions"], data["orientations"]

    connect(gui=False)
    body, cfg, limits = load_robot(ROBOT)
    limits = np.asarray(limits)
    dof = cfg["dof"]

    best, best_col = MR.load_run_models(str(BEST_RUN_DIR), meta, dof, "cpu")
    base = train_ik.IKNet(dof)
    base.load_state_dict(torch.load(RESULTS / "baseline_repro" / "xarm6_ik_net.pt", map_location="cpu"))
    base.eval()
    base_col = train_ik.CollisionNet(dof)
    base_col.load_state_dict(torch.load(RESULTS / "baseline_repro" / "xarm6_collision_net.pt", map_location="cpu"))
    base_col.eval()
    n_params = {"residual_mlp": MR.count_params(best), "baseline_mlp": MR.count_params(base),
                "collision_net": MR.count_params(best_col)}
    print("params", n_params)

    # ---------------- A) whole test split, best network, the run's own warm seeds ----------------
    Xa = M.make_inputs(P[test_idx], O[test_idx], warm_all)
    qa = M.predict(best, Xa, dof)
    pe_a, oe_a, col_a, lim_a = fk_check(body, cfg, qa, Xa, limits)
    full_test = summary_row("residual_mlp", "warm (run seeds)", pe_a, oe_a, col_a, lim_a, np.nan, np.nan,
                            "whole collision-free test split of the best run")
    print(f"A) full test split n={len(pe_a)}: {full_test['pos_mean_mm']:.2f} mm mean, "
          f"{full_test['orn_mean_deg']:.2f} deg (recorded: {meta['metrics']['test']['pos_mean_mm']:.2f} mm)")

    # collision classifiers on the whole test split (colliding rows included)
    test_all = np.load(BEST_RUN_DIR / "split.npz")["test"]
    lab = data["collision_labels"][test_all].astype(bool)
    cls = {"best_run_classifier_test": M.collision_metrics(best_col, J[test_all], lab),
           "baseline_repro_classifier_test": M.collision_metrics(base_col, J[test_all], lab)}
    print("classifiers", {k: round(v["accuracy"], 2) for k, v in cls.items()})

    # ---------------- B) common subset, every method, three seed modes ----------------
    pool = np.nonzero(test_idx >= 14000)[0]
    sel = np.sort(np.random.default_rng(a.subset_seed).choice(pool, a.n, replace=False))
    idx = test_idx[sel]
    q_true = J[idx].astype(float)
    seeds = {
        "warm": warm_all[sel].astype(float),
        "zero": np.zeros_like(q_true),
        "random": np.random.default_rng(a.subset_seed + 1).uniform(limits[:, 0], limits[:, 1], size=q_true.shape),
    }
    rows, per = [], {"dataset_index": idx, "q_true": q_true}
    for mode, sd in seeds.items():
        X = M.make_inputs(P[idx], O[idx], sd)
        per[f"seed_{mode}"] = sd
        # networks alone
        for name, net in (("baseline_mlp", base), ("residual_mlp", best)):
            q = M.predict(net, X, dof)
            pe, oe, col, lim = fk_check(body, cfg, q, X, limits)
            ms = net_single_latency(net, X)
            rows.append(summary_row(name, mode, pe, oe, col, lim, ms[0] * 1e3, ms[1] * 1e3))
            per[f"{name}_{mode}_q"], per[f"{name}_{mode}_pe"], per[f"{name}_{mode}_oe"] = q, pe, oe
            per[f"{name}_{mode}_col"], per[f"{name}_{mode}_lim"] = col, lim
        # classical DLS from the seed, 100 iterations
        q, tt = dls(body, cfg, X, sd, 100)
        pe, oe, col, lim = fk_check(body, cfg, q, X, limits)
        rows.append(summary_row("classical_dls", mode, pe, oe, col, lim, tt.mean() * 1e3, np.median(tt) * 1e3,
                                "PyBullet DLS, 100 iterations, seed as warm start"))
        per[f"classical_dls_{mode}_q"], per[f"classical_dls_{mode}_pe"], per[f"classical_dls_{mode}_oe"] = q, pe, oe
        per[f"classical_dls_{mode}_col"], per[f"classical_dls_{mode}_lim"] = col, lim
        # hybrids with 5 DLS iterations from the network output
        for name, net in (("baseline_hybrid", base), ("hybrid_dls", best)):
            qn = M.predict(net, X, dof)
            q, tt = dls(body, cfg, X, qn, 5)
            pe, oe, col, lim = fk_check(body, cfg, q, X, limits)
            t_net = net_single_latency(net, X)
            rows.append(summary_row(name, mode, pe, oe, col, lim, (tt.mean() + t_net[0]) * 1e3,
                                    (np.median(tt) + t_net[1]) * 1e3, "network + 5 DLS iterations"))
            per[f"{name}_{mode}_q"], per[f"{name}_{mode}_pe"], per[f"{name}_{mode}_oe"] = q, pe, oe
            per[f"{name}_{mode}_col"], per[f"{name}_{mode}_lim"] = col, lim
        # LM alone from the seed (no network): same solver, restarts and retry
        t0 = time.perf_counter()
        summ0, pr0 = M.refine_ik(ROBOT, body, cfg, limits, X, X[:, 7:7 + dof].astype(float),
                                 settings={"use_seed": False}, device="cpu", q_true=q_true)
        batch0 = (time.perf_counter() - t0) * 1000 / len(X)
        q = pr0["q_ref"]
        pe, oe, col, lim = fk_check(body, cfg, q, X, limits)
        lat = []
        for k in range(min(a.timing_n, len(X))):
            t0 = time.perf_counter()
            M.refine_ik(ROBOT, body, cfg, limits, X[k:k + 1], X[k:k + 1, 7:7 + dof].astype(float),
                        settings={"use_seed": False}, device="cpu")
            lat.append(time.perf_counter() - t0)
        w0 = summ0["refine"]["won_by"]
        rows.append(summary_row("lm_only", mode, pe, oe, col, lim, float(np.mean(lat) * 1e3), float(np.median(lat) * 1e3),
                                f"LM from the seed only (30 it, 16 restarts, collision-free retry); batched {batch0:.3f} ms/target",
                                won={"network": 0.0, "seed": w0["network"], "random restart": w0["random restart"],
                                     "collision-free retry": w0["collision-free retry"]}))
        per[f"lm_only_{mode}_q"], per[f"lm_only_{mode}_pe"], per[f"lm_only_{mode}_oe"] = q, pe, oe
        per[f"lm_only_{mode}_col"], per[f"lm_only_{mode}_lim"] = col, lim
        # hybrid: network + LM refinement (project's final method), batched over all targets
        qn = M.predict(best, X, dof)
        t0 = time.perf_counter()
        summ, pr = M.refine_ik(ROBOT, body, cfg, limits, X, qn, device="cpu", q_true=q_true)
        batch_ms = (time.perf_counter() - t0) * 1000 / len(X)
        q = pr["q_ref"]
        pe, oe, col, lim = fk_check(body, cfg, q, X, limits)
        # single-target latency (network + LM + retries, batch of 1) on the first timing_n targets
        lat = []
        for k in range(min(a.timing_n, len(X))):
            t0 = time.perf_counter()
            qk = M.predict(best, X[k:k + 1], dof)
            M.refine_ik(ROBOT, body, cfg, limits, X[k:k + 1], qk, device="cpu")
            lat.append(time.perf_counter() - t0)
        won = summ["refine"]["won_by"]
        rows.append(summary_row("hybrid_lm", mode, pe, oe, col, lim, float(np.mean(lat) * 1e3), float(np.median(lat) * 1e3),
                                f"network + LM (30 it, 16 restarts, collision-free retry); batched {batch_ms:.3f} ms/target",
                                won=won))
        per[f"hybrid_lm_{mode}_q"], per[f"hybrid_lm_{mode}_pe"], per[f"hybrid_lm_{mode}_oe"] = q, pe, oe
        per[f"hybrid_lm_{mode}_col"], per[f"hybrid_lm_{mode}_lim"] = col, lim
        per[f"hybrid_lm_{mode}_winner"] = pr["winner"]
        print(f"B) seed={mode} done ({time.time() - t_start:.0f} s)")
        for r in rows[-7:]:
            print(f"   {r['method']:16s} pos {r['pos_mean_mm']:9.3f} mm  orn {r['orn_mean_deg']:7.3f} deg  "
                  f"coll {r['self_collision_pct']:5.1f}%  lim {r['limit_violation_pct']:5.1f}%  "
                  f"ok@1mm/1deg {r['success_1mm_1deg_pct']:5.1f}%  {r['time_ms_mean']:.3f} ms")

    RESULTS.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(RESULTS / "evaluation_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in [full_test] + rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    WORK.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(WORK / "eval_per_target.npz", **per)
    write_json(RESULTS / "eval_summary.json", {"full_test_split": full_test, "common_subset": rows,
                                               "collision_classifiers": cls, "params": n_params})
    write_json(RESULTS / "experiment_configuration.json", {
        "robot": ROBOT, "urdf": "urdf/xarm/xarm6_robot.urdf", "dof": dof,
        "joint_limits_deg": np.degrees(limits).round(2).tolist(),
        "best_run": {"id": BEST_RUN, "dataset": DATASET_200K.name, "model": meta["model"], "params": meta["params"],
                     "split": meta["split"], "train_secs": meta.get("train_secs"), "gpu": meta.get("gpu")},
        "baseline_reproduction": read_json(RESULTS / "baseline_repro" / "summary.json"),
        "common_subset": {"n": int(a.n), "subset_seed": a.subset_seed, "rule": "test rows with dataset index >= 14000",
                          "seed_modes": {"warm": "true joints + N(0, 0.3 rad) (the run's own test seeds)",
                                         "zero": "all joints 0 (home pose)", "random": "uniform within joint limits"}},
        "methods": {"classical_dls": "PyBullet calculateInverseKinematics, maxNumIterations=100, residualThreshold=1e-5",
                    "hybrid_dls": "network output + PyBullet DLS, maxNumIterations=5",
                    "hybrid_lm": {**M.REFINE_DEFAULTS, "solver": "refine/lm_solver.py (Levenberg-Marquardt, lambda=|e|^2+1e-6)",
                                  "device": "cpu"}},
        "checks": "PyBullet FK pose error, joint limits (after +-360 deg shift), self-collision (URDF_USE_SELF_COLLISION | EXCLUDE_PARENT)",
        "timing": "single target (batch of 1), one CPU thread, mean over targets; hardware dependent",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "pybullet": "3.2.7",
                        "numpy": np.__version__, "platform": platform.platform(), "cpu_count": os.cpu_count(),
                        "cpu_model": next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")},
        "runtime_secs": round(time.time() - t_start, 1),
    })
    print(f"done in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
