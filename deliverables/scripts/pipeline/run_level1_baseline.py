#!/usr/bin/env python
"""Level 1 baseline reproduction for the xArm6 (README section "Running it end to end").

1. Copies robot_config.py, generate_dataset.py, train_ik.py and evaluate.py (plus urdf/) into a
   scratch work folder, so the project's own data/ and models/ are never overwritten.
2. Runs them UNCHANGED, exactly as the README says:
       python generate_dataset.py --robot xarm6 --n 20000
       python train_ik.py --robot xarm6 --epochs 60
       python evaluate.py --robot xarm6 --n 200
3. Runs an instrumented copy of train_ik.py's two training loops on the same dataset: same model
   classes (imported from train_ik.py), same data preparation (train_ik.load_dataset), same
   optimiser, learning rate, batch size, epochs and collision weight. The only additions are fixed
   torch seeds and logging: per-epoch training loss, collision penalty, validation MSE, classifier
   accuracy, FK-checked pose error on a fixed validation subset, and the predictions for a few
   fixed validation cases (used for the "network learning" animation).
4. Evaluates the instrumented model with the unchanged evaluate.py too, so both runs are comparable.

Outputs: deliverables/results/baseline_repro/ and deliverables/results/training_history.csv

    python deliverables/scripts/pipeline/run_level1_baseline.py
"""
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import REPO, RESULTS, WORK, write_json  # noqa: E402

OUT = RESULTS / "baseline_repro"
WD = WORK / "level1_baseline"
ROBOT = "xarm6"
N_SAMPLES, EPOCHS, COLLISION_EPOCHS, EVAL_N = 20000, 60, 15, 200
SEED = 0                    # torch seed of the instrumented run
N_ACC = 1000                # validation rows whose FK pose error is measured every epoch
CASES = 4                   # fixed validation cases whose predictions are stored every epoch


def run(cmd, log_name):
    t0 = time.time()
    r = subprocess.run(cmd, cwd=WD, capture_output=True, text=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    (OUT / "logs" / log_name).write_text(r.stdout + ("\n[stderr]\n" + r.stderr if r.stderr.strip() else ""))
    if r.returncode:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{r.stderr[-3000:]}")
    print(f"  {' '.join(cmd[1:])}  ({time.time() - t0:.1f} s)")
    return r.stdout, time.time() - t0


def parse_evaluate(text):
    """evaluate.py's printed summary -> {method: {metric: value}}."""
    out, cur = {}, None
    for line in text.splitlines():
        m = re.match(r"-- (.+) --", line.strip())
        if m:
            cur = m.group(1)
            out[cur] = {}
            continue
        m = re.match(r"\s*(mean position error|mean orientation error|self-collision rate|joint-limit violation|mean inference time):\s+([\d.]+)", line)
        if m and cur:
            key = {"mean position error": "pos_mean_mm", "mean orientation error": "orn_mean_deg",
                   "self-collision rate": "self_collision_pct", "joint-limit violation": "limit_violation_pct",
                   "mean inference time": "time_ms"}[m.group(1)]
            out[cur][key] = float(m.group(2))
    return out


def prepare_workdir():
    if WD.exists():
        shutil.rmtree(WD)
    (WD / "data").mkdir(parents=True)
    (WD / "models").mkdir()
    for f in ("robot_config.py", "generate_dataset.py", "train_ik.py", "evaluate.py"):
        shutil.copy2(REPO / f, WD / f)
    os.symlink(REPO / "urdf", WD / "urdf")


def instrumented_training():
    """train_ik.py's train_collision_net + train_ik_net loops, line for line, plus logging."""
    import numpy as np
    import pybullet as p
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    os.chdir(WD)
    sys.path.insert(0, str(WD))
    import train_ik as ti                      # the unchanged copy in the work folder
    from robot_config import connect, load_robot

    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    device = "cpu"
    X, y, dof, raw = ti.load_dataset(ROBOT)
    hist = []

    # ---------------- collision classifier (train_ik.train_collision_net) ----------------
    joints = torch.tensor(raw["joints"], dtype=torch.float32)
    labels = torch.tensor(raw["collision_labels"], dtype=torch.float32)
    ds = TensorDataset(joints, labels)
    n_val = max(1, int(0.15 * len(ds)))
    train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)
    cnet = ti.CollisionNet(dof).to(device)
    opt = torch.optim.Adam(cnet.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(COLLISION_EPOCHS):
        t0 = time.time()
        cnet.train()
        losses, corr_tr, n_tr = [], 0, 0
        for qb, yb in train_loader:
            opt.zero_grad()
            logits = cnet(qb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            corr_tr += ((torch.sigmoid(logits) > 0.5).float() == yb).sum().item()
            n_tr += yb.numel()
        cnet.eval()
        correct, total, vloss = 0, 0, []
        with torch.no_grad():
            for qb, yb in val_loader:
                lg = cnet(qb)
                vloss.append(loss_fn(lg, yb).item())
                correct += ((torch.sigmoid(lg) > 0.5).float() == yb).sum().item()
                total += yb.numel()
        hist.append({"phase": "collision", "epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                     "train_acc_pct": 100 * corr_tr / n_tr, "val_loss": float(np.mean(vloss)),
                     "val_acc_pct": 100 * correct / total, "secs": time.time() - t0})
        print(f"  [collision-net] epoch {epoch + 1}/{COLLISION_EPOCHS}  val_acc={correct / total:.3f}")

    # collision classifier: precision / recall on its validation split (threshold 0.5)
    vi = np.array(val_ds.indices)
    with torch.no_grad():
        pred = (torch.sigmoid(cnet(joints[vi])) > 0.5).numpy()
    lab = raw["collision_labels"][vi].astype(bool)
    tp, fp, fn = int((pred & lab).sum()), int((pred & ~lab).sum()), int((~pred & lab).sum())
    cls_metrics = {"n_val": int(len(vi)), "n_val_colliding": int(lab.sum()),
                   "accuracy_pct": float((pred == lab).mean() * 100),
                   "precision_pct": 100 * tp / max(1, tp + fp), "recall_pct": 100 * tp / max(1, tp + fn)}

    # ---------------- IK network (train_ik.train_ik_net) ----------------
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    ds = TensorDataset(X_t, y_t)
    n_val = max(1, int(0.15 * len(ds)))
    train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)
    model = ti.IKNet(dof).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    collision_weight = 0.05
    cnet.eval()
    for prm in cnet.parameters():
        prm.requires_grad_(False)

    # fixed validation rows: FK-checked pose error every epoch + a few cases for the animation
    connect(gui=False)
    body, cfg, limits = load_robot(ROBOT)
    vidx = np.array(val_ds.indices)
    rng = np.random.default_rng(7)
    acc_rows = np.sort(rng.choice(vidx, min(N_ACC, len(vidx)), replace=False))
    case_rows = acc_rows[:CASES]
    q_true_free = raw["joints"][raw["collision_labels"] == 0]

    def fk(q):
        for j, a in zip(cfg["joint_indices"], q):
            p.resetJointState(body, j, float(a))
        st = p.getLinkState(body, cfg["ee_link_index"], computeForwardKinematics=True)
        return np.array(st[4]), np.array(st[5])

    def pose_errors(rows):
        with torch.no_grad():
            q = model.decode_angles(model(X_t[rows])).numpy()
        pe, oe = [], []
        for r, qi in zip(rows, q):
            pos, orn = fk(qi)
            pe.append(np.linalg.norm(pos - X[r, :3]) * 1000)
            oe.append(np.degrees(2 * np.arccos(np.clip(abs(np.dot(orn, X[r, 3:7])), -1, 1))))
        return q, np.array(pe), np.array(oe)

    cases = [{"row": int(r), "target_pos": X[r, :3].tolist(), "target_orn": X[r, 3:7].tolist(),
              "seed": X[r, 7:].tolist(), "q_true": q_true_free[r].tolist(), "epochs": []} for r in case_rows]
    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        fit_l, col_l, tot_l = [], [], []
        for xb, yb in train_loader:
            opt.zero_grad()
            raw_out = model(xb)
            fit_loss = mse(raw_out, yb)
            angles = model.decode_angles(raw_out)
            collision_logits = cnet(angles)
            collision_penalty = torch.sigmoid(collision_logits).mean()
            loss = fit_loss + collision_weight * collision_penalty
            loss.backward()
            opt.step()
            fit_l.append(fit_loss.item())
            col_l.append(collision_penalty.item())
            tot_l.append(loss.item())
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                val_losses.append(mse(model(xb), yb).item())
        train_secs = time.time() - t0
        q_acc, pe, oe = pose_errors(acc_rows)
        rec = {"phase": "ik", "epoch": epoch + 1, "train_loss": float(np.mean(tot_l)),
               "train_fit_mse": float(np.mean(fit_l)), "train_collision_penalty": float(np.mean(col_l)),
               "val_mse": float(np.mean(val_losses)), "val_pos_mean_mm": float(pe.mean()),
               "val_pos_median_mm": float(np.median(pe)), "val_orn_mean_deg": float(oe.mean()),
               "val_within_50mm_pct": float((pe <= 50).mean() * 100), "secs": train_secs}
        hist.append(rec)
        for c, qi, e_p, e_o in zip(cases, q_acc[:CASES], pe[:CASES], oe[:CASES]):
            c["epochs"].append({"epoch": epoch + 1, "q_pred": [round(float(v), 5) for v in qi],
                                "pos_err_mm": round(float(e_p), 2), "orn_err_deg": round(float(e_o), 2)})
        if epoch == EPOCHS - 1 or epoch % 10 == 0:
            print(f"  [ik-net] epoch {epoch + 1}/{EPOCHS}  val_mse={rec['val_mse']:.5f}  "
                  f"pos_err={rec['val_pos_mean_mm']:.1f} mm")
    p.disconnect()
    return model, cnet, hist, cases, cls_metrics, {"n_dataset": int(len(raw["joints"])),
                                                    "n_colliding": int(raw["collision_labels"].sum()),
                                                    "n_ik_rows": int(len(X)), "n_ik_train": len(train_ds),
                                                    "n_ik_val": len(val_ds), "n_acc_rows": int(len(acc_rows))}


def main():
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    prepare_workdir()
    print("1) unchanged project scripts (README recipe)")
    gen_out, gen_s = run([py, "generate_dataset.py", "--robot", ROBOT, "--n", str(N_SAMPLES)], "1_generate_dataset.txt")
    _, train_s = run([py, "train_ik.py", "--robot", ROBOT, "--epochs", str(EPOCHS)], "2_train_ik_unchanged.txt")
    ev_a, _ = run([py, "evaluate.py", "--robot", ROBOT, "--n", str(EVAL_N)], "3_evaluate_unchanged_model.txt")
    shutil.copy2(WD / "models" / f"{ROBOT}_ik_net.pt", OUT / "unchanged_run_ik_net.pt")

    print("2) instrumented training (same loops + logging, fixed seeds)")
    t0 = time.time()
    model, cnet, hist, cases, cls_metrics, split = instrumented_training()
    instr_s = time.time() - t0
    torch.save(model.state_dict(), OUT / f"{ROBOT}_ik_net.pt")
    torch.save(cnet.state_dict(), OUT / f"{ROBOT}_collision_net.pt")
    shutil.copy2(OUT / f"{ROBOT}_ik_net.pt", WD / "models" / f"{ROBOT}_ik_net.pt")
    shutil.copy2(OUT / f"{ROBOT}_collision_net.pt", WD / "models" / f"{ROBOT}_collision_net.pt")
    ev_b, _ = run([py, "evaluate.py", "--robot", ROBOT, "--n", str(EVAL_N)], "4_evaluate_instrumented_model.txt")

    keys = ["phase", "epoch", "train_loss", "train_fit_mse", "train_collision_penalty", "train_acc_pct",
            "val_loss", "val_mse", "val_acc_pct", "val_pos_mean_mm", "val_pos_median_mm", "val_orn_mean_deg",
            "val_within_50mm_pct", "secs"]
    with open(RESULTS / "training_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for h in hist:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in h.items() if k in keys})
    m = re.search(r"generated (\d+) samples in ([\d.]+)s \((\d+) colliding", gen_out)
    summary = {
        "robot": ROBOT, "samples": N_SAMPLES, "generator_seed": 0, "epochs": EPOCHS,
        "collision_epochs": COLLISION_EPOCHS, "torch_seed_instrumented": SEED,
        "dataset": {"n": int(m.group(1)), "n_colliding": int(m.group(3)), "generation_secs": float(m.group(2))} if m else None,
        "split_instrumented": split, "collision_classifier_val": cls_metrics,
        "final_epoch": hist[-1], "seconds": {"generate": gen_s, "train_unchanged": train_s, "train_instrumented": instr_s},
        "evaluate_py_unchanged_model": parse_evaluate(ev_a),
        "evaluate_py_instrumented_model": parse_evaluate(ev_b),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "platform": platform.platform(), "cpu_count": os.cpu_count(), "device": "cpu"},
    }
    write_json(OUT / "summary.json", summary)
    write_json(OUT / "case_predictions.json", {"robot": ROBOT, "cases": cases})
    print(json.dumps({k: summary[k] for k in ("evaluate_py_unchanged_model", "evaluate_py_instrumented_model")}, indent=1))


if __name__ == "__main__":
    main()
