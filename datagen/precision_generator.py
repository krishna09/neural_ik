"""
Precision IK dataset generator - for the "< 1 mm and < 1 deg" goal.

What an IK dataset has to get right for a network that is followed by
numerical refinement (refine/lm_solver.py):

1. Correct labels. A pose the robot can reach safely must never be labelled
   colliding, or the network learns nothing there. The self-collision matrix of
   clean_generator.py is used (rigidly attached links, links one movable joint
   apart and pairs that touch in >= 95 % of random poses are not checked - the
   MoveIt Setup Assistant rule), so it works for the xArm6, KUKA iiwa and Panda.
   An optional clearance (clearance_mm > 0) also labels poses that come closer
   than that to a self-collision as colliding, so every IK target keeps a
   safety margin.
2. Even coverage of the joint space. Scrambled Sobol (low-discrepancy) points
   instead of independent uniform draws leave no holes, so the network is an
   equally good starting point everywhere; near_limit_frac of the samples have
   every joint within the outer 15 % of its range, where IK is hardest and
   joint-limit clamping matters.
3. One orientation representation per pose. The quaternions q and -q are the
   same rotation; PyBullet returns either sign. Every quaternion is stored with
   w >= 0 (canonical hemisphere), so the network input is consistent.
4. Exactly the keys of generate_dataset.py (joints, positions, orientations,
   collision_labels, joint_limits, dof): train_ik.py and the UI read it as is.
   Colliding samples stay in the file with label 1 - train_ik trains the
   collision classifier on them and the IK network only on label 0.

UI: generate(robot_name, n_samples, seed=0, out_path=None, progress=None,
             near_limit_frac=0.15, clearance_mm=0.0)
"""

import time

import numpy as np
import pybullet as p

from robot_config import connect, load_robot
from clean_generator import build_collision_matrix, sample_configs

DESCRIPTION = ("Precision generator for < 1 mm / < 1 deg IK: clean self-collision matrix (works for "
               "xArm6, KUKA, Panda), Sobol coverage + near-limit samples, canonical quaternions (w >= 0), "
               "optional safety clearance.")
ID_TAG = "precision"
OPTION_HELP = {
    "near_limit_frac": "share of samples drawn close to the joint limits (0 - 0.5)",
    "clearance_mm": "also label poses closer than this to a self-collision as colliding (0 = off; slower when on)",
}
LAST_REPORT = {}


def canonical_quat(q):
    """(N,4) xyzw -> same rotations with w >= 0."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    return np.where(q[:, 3:4] < 0, -q, q)


def _collides(body, clearance_m):
    p.performCollisionDetection()
    if p.getContactPoints(bodyA=body, bodyB=body):
        return True
    if clearance_m > 0:
        # getClosestPoints respects the collision filter set by build_collision_matrix
        return len(p.getClosestPoints(bodyA=body, bodyB=body, distance=clearance_m)) > 0
    return False


def generate(robot_name, n_samples, seed=0, out_path=None, progress=None, near_limit_frac=0.15, clearance_mm=0.0):
    out_path = out_path or f"data/{robot_name}_ik_dataset.npz"
    near_limit_frac = float(min(0.5, max(0.0, near_limit_frac)))
    clearance_m = max(0.0, float(clearance_mm)) / 1000.0
    t0 = time.time()
    cid = connect(gui=False)
    try:
        body, cfg, limits = load_robot(robot_name)
        report = build_collision_matrix(body, cfg, limits, seed)
        dof = cfg["dof"]
        Q, n_sobol = sample_configs(n_samples, limits, seed, edge_frac=near_limit_frac)
        joints = Q.astype(np.float32)
        positions = np.zeros((n_samples, 3), dtype=np.float64)
        orientations = np.zeros((n_samples, 4), dtype=np.float64)
        labels = np.zeros((n_samples,), dtype=np.int64)
        for i, q in enumerate(joints):
            for idx, a in zip(cfg["joint_indices"], q):
                p.resetJointState(body, idx, float(a))
            st = p.getLinkState(body, cfg["ee_link_index"], computeForwardKinematics=True)
            collided = _collides(body, clearance_m)
            positions[i], orientations[i], labels[i] = st[4], st[5], int(collided)
            if progress:
                progress(i + 1, n_samples, q, st[4], st[5], collided, "sobol" if i < n_sobol else "near-limits")
    finally:
        p.disconnect(cid)

    orientations = canonical_quat(orientations)
    np.savez(out_path, joints=joints, positions=positions.astype(np.float32),
             orientations=orientations.astype(np.float32), collision_labels=labels,
             joint_limits=np.array(limits, dtype=np.float32), dof=dof)
    free = labels == 0
    report.update(
        robot=robot_name, n_samples=n_samples, seed=seed, n_colliding=int(labels.sum()),
        sampling=f"scrambled Sobol ({n_sobol}) + near-limits ({n_samples - n_sobol})",
        clearance_mm=clearance_m * 1000, quaternion="canonical, w >= 0",
        workspace_m={"min": positions[free].min(0).round(3).tolist() if free.any() else None,
                     "max": positions[free].max(0).round(3).tolist() if free.any() else None},
        secs=round(time.time() - t0, 2))
    LAST_REPORT.clear()
    LAST_REPORT.update(report)
    print(f"[{robot_name}] precision generator: {n_samples} samples in {report['secs']:.1f}s "
          f"({labels.sum()} colliding, {labels.mean():.1%}); ignored {report['disabled_pair_count']} "
          f"structural link pairs; clearance {clearance_m * 1000:g} mm")
    print(f"saved -> {out_path}")
    return out_path
