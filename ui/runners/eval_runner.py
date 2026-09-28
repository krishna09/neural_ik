"""
Evaluates a saved model (a library run) on any library dataset of the same
robot and appends the result to that run's meta.json -> "evaluations".

The saved state_dicts are loaded into the network class the run was trained
with (see model_registry.py): train_ik.IKNet for base runs - exactly what
evaluate.py would load - or the run's modelset model; collision net:
train_ik.CollisionNet.

Inputs are built with train_ik.load_dataset's recipe: target pose from the
dataset + seed = true joints + N(0, 0.3). Accuracy = predicted joints ->
PyBullet FK -> distance from the target pose (see ik_metrics.py).
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

    n = len(d["joints"])
    rows = np.arange(n)
    if max_samples and max_samples < n:
        rows = np.sort(np.random.default_rng(eval_seed).choice(n, max_samples, replace=False))
    free = rows[d["collision_labels"][rows] == 0]
    seeds = M.noisy_seeds(d["joints"][free], eval_seed)
    X = M.make_inputs(d["positions"][free], d["orientations"][free], seeds)
    q_true = d["joints"][free]

    emit({"type": "eval_start", "robot": robot, "run_id": P["run_id"], "dataset_id": P["dataset_id"],
          "n": int(len(free)), "i": 0})
    t0 = time.time()
    chunks = {"q_pred": [], "pos_err": [], "orn_err": [], "colliding": [], "limit_viol": []}
    step = 2000
    for i in range(0, len(X), step):
        _, per = M.evaluate_ik(ik, body, cfg, limits, X[i:i + step], q_true[i:i + step])
        for k in chunks:
            chunks[k].append(per[k])
        emit({"type": "eval_progress", "robot": robot, "run_id": P["run_id"],
              "i": int(min(len(X), i + step)), "n": int(len(X))})
    per = {k: np.concatenate(v) if v else np.zeros(0) for k, v in chunks.items()}
    tol = (meta.get("params") or {}).get("pos_tol_mm")        # the run's own accuracy tolerance
    summary = M.summarize(per["pos_err"], per["orn_err"], per["q_pred"] - q_true, per["colliding"], per["limit_viol"],
                          extra_pos_mm=(tol,) if tol else ())
    colm = M.collision_metrics(col, d["joints"][rows], d["collision_labels"][rows]) if col is not None else None

    eval_id = "eval_" + time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(os.path.join(run_dir, "evals"), exist_ok=True)
    np.savez(os.path.join(run_dir, "evals", eval_id + ".npz"), index=free, seed=seeds, **per)
    rec = {
        "id": eval_id, "dataset_id": P["dataset_id"], "created": time.time(),
        "n_rows": int(len(rows)), "overlap": P.get("overlap"), "seed_mode": "true joints + N(0, 0.3)",
        "summary": summary, "collision": colm, "secs": round(time.time() - t0, 2),
    }
    meta = read_json(meta_path, {})           # re-read: keep anything written meanwhile
    meta.setdefault("evaluations", []).append(rec)
    write_json(meta_path, meta)
    emit({"type": "eval_done", "robot": robot, "run_id": P["run_id"], **rec})


if __name__ == "__main__":
    main()
