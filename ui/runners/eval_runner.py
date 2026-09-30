"""
Evaluates a saved model (a library run) on any library dataset of the same
robot and appends the result to that run's meta.json -> "evaluations".

The saved state_dicts are loaded into the network class the run was trained
with (see model_registry.py): train_ik.IKNet for base runs - exactly what
evaluate.py would load - or the run's modelset model; collision net:
train_ik.CollisionNet.

Inputs: target pose from the dataset + a seed - by default train_ik.load_dataset's
recipe (true joints + N(0, 0.3), "warm start"), or the zero pose / a random pose
("cold start", seed_mode). Accuracy = predicted joints -> PyBullet FK ->
distance from the target pose (see ik_metrics.py).

With "refine" settings, the network's answer is also refined numerically
(refine/lm_solver.py: Levenberg-Marquardt from the network output and the seed,
random restarts for targets still outside tolerance) and checked again with
PyBullet; both results are stored (summary / summary_refined).
"""

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import ik_metrics as M  # noqa: E402
import collision_filter  # noqa: E402
import model_registry as MR  # noqa: E402
from libio import read_json, write_json  # noqa: E402
from robot_config import connect, load_robot  # noqa: E402


def emit(ev):
    sys.stdout.write("@@EV " + json.dumps(ev) + "\n")
    sys.stdout.flush()


def main():
    P = json.loads(sys.argv[1])
    run_dir, robot = P["run_dir"], P["robot"]
    max_samples = int(P.get("max_samples", 0))
    eval_seed = int(P.get("eval_seed", 1))
    seed_mode = P.get("seed_mode") or "noisy"
    refine = P.get("refine") or None          # None = network only; else refine/lm_solver settings
    meta_path = os.path.join(run_dir, "meta.json")
    meta = read_json(meta_path, {})
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with np.load(P["dataset_file"]) as z:
        d = {k: z[k] for k in z.files}
    dof = int(d["dof"])
    # the network class the run was trained with (its model_source.py snapshot;
    # older runs: train_ik.IKNet, rebuilt at the width stored in the checkpoint)
    ik, col = MR.load_run_models(run_dir, meta, dof, device)

    connect(gui=False)
    body, cfg, limits = load_robot(robot)
    collision_filter.apply(body, cfg, limits)       # ignore link pairs that touch by construction

    n = len(d["joints"])
    rows = np.arange(n)
    if max_samples and max_samples < n:
        rows = np.sort(np.random.default_rng(eval_seed).choice(n, max_samples, replace=False))
    free = rows[d["collision_labels"][rows] == 0]
    seeds = M.make_seeds(d["joints"][free], seed_mode, limits, eval_seed)
    X = M.make_inputs(d["positions"][free], d["orientations"][free], seeds)
    q_true = d["joints"][free]
    tol = (meta.get("params") or {}).get("pos_tol_mm")        # the run's own accuracy tolerance
    extra = (tol,) if tol else ()

    emit({"type": "eval_start", "robot": robot, "run_id": P["run_id"], "dataset_id": P["dataset_id"],
          "n": int(len(free)), "i": 0})
    t0 = time.time()
    chunks = {"q_pred": [], "pos_err": [], "orn_err": [], "colliding": [], "limit_viol": []}
    ref_keys = ("q_ref", "pos_err_ref", "orn_err_ref", "colliding_ref", "limit_viol_ref", "winner")
    ref_chunks = {k: [] for k in ref_keys}
    ref_secs = 0.0
    step = 2000 if not refine else 10000
    for i in range(0, len(X), step):
        _, per = M.evaluate_ik(ik, body, cfg, limits, X[i:i + step], q_true[i:i + step])
        for k in chunks:
            chunks[k].append(per[k])
        if refine:
            s_ref, per_ref = M.refine_ik(robot, body, cfg, limits, X[i:i + step], per["q_pred"], refine, device=device)
            ref_secs += s_ref["refine"]["ms_per_target"] * len(per["q_pred"]) / 1000
            for k in ref_keys:
                ref_chunks[k].append(per_ref[k])
        emit({"type": "eval_progress", "robot": robot, "run_id": P["run_id"],
              "i": int(min(len(X), i + step)), "n": int(len(X))})
    per = {k: np.concatenate(v) if v else np.zeros(0) for k, v in chunks.items()}
    summary = M.summarize(per["pos_err"], per["orn_err"], per["q_pred"] - q_true, per["colliding"], per["limit_viol"],
                          extra_pos_mm=extra)
    colm = M.collision_metrics(col, d["joints"][rows], d["collision_labels"][rows]) if col is not None else None
    summary_ref = None
    if refine:
        per.update({k: np.concatenate(v) for k, v in ref_chunks.items()})
        summary_ref = M.summarize(per["pos_err_ref"], per["orn_err_ref"], None, per["colliding_ref"],
                                  per["limit_viol_ref"], extra_pos_mm=extra)
        st = dict(M.REFINE_DEFAULTS, **refine)
        w = per["winner"]
        summary_ref["refine"] = {**st, "device": device, "ms_per_target": ref_secs * 1000 / max(1, len(w)),
                                 "won_by": M.won_by(w, 2 if st["use_seed"] else 1, st["use_seed"])}

    eval_id = "eval_" + time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(os.path.join(run_dir, "evals"), exist_ok=True)
    np.savez(os.path.join(run_dir, "evals", eval_id + ".npz"), index=free, seed=seeds, **per)
    rec = {
        "id": eval_id, "dataset_id": P["dataset_id"], "created": time.time(),
        "n_rows": int(len(rows)), "overlap": P.get("overlap"), "seed_mode": M.SEED_MODES[seed_mode],
        "seed_mode_id": seed_mode,
        "summary": summary, "summary_refined": summary_ref, "collision": colm, "secs": round(time.time() - t0, 2),
    }
    meta = read_json(meta_path, {})           # re-read: keep anything written meanwhile
    meta.setdefault("evaluations", []).append(rec)
    write_json(meta_path, meta)
    emit({"type": "eval_done", "robot": robot, "run_id": P["run_id"], **rec})


if __name__ == "__main__":
    main()
