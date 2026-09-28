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

POS_THRESH_MM = (10, 25, 50, 100)
ORN_THRESH_DEG = (5, 15, 30)
SEED_NOISE_STD = 0.3   # train_ik.load_dataset's default seed noise


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
    """extra_pos_mm: user-set position tolerances reported next to the presets."""
    pe = np.asarray(pos_err_m, dtype=float) * 1000.0
    oe = np.asarray(orn_err_deg, dtype=float)
    if len(pe) == 0:
        return {"n": 0}
    pos_t = sorted({float(t) for t in POS_THRESH_MM} | {float(t) for t in extra_pos_mm if t})
    s = {
        "n": int(len(pe)),
        "pos_mean_mm": float(pe.mean()),
        "pos_median_mm": float(np.median(pe)),
        "pos_p90_mm": float(np.percentile(pe, 90)),
        "orn_mean_deg": float(oe.mean()),
        "orn_median_deg": float(np.median(oe)),
        "acc_pos": {tkey(t): float((pe <= t).mean() * 100) for t in pos_t},
        "acc_orn": {tkey(t): float((oe <= t).mean() * 100) for t in ORN_THRESH_DEG},
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
    lim = limit_violations(q_pred, limits)
    summary = summarize(pos_err, orn_err, q_pred - q_true, col, lim, extra_pos_mm=extra_pos_mm)
    return summary, {"q_pred": q_pred, "pos_err": pos_err, "orn_err": orn_err, "colliding": col, "limit_viol": lim}


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
