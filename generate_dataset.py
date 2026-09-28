"""
Level 1 - Step 1: Dataset generation for neural inverse kinematics.

For a chosen robot (6DOF xarm6 or 7DOF kuka_iiwa/panda), this script:
  1. Samples random joint configurations within each joint's limits.
  2. Runs forward kinematics (PyBullet does this natively) to get the
     resulting end-effector pose (position + orientation quaternion).
  3. Runs PyBullet's built-in collision checker on that configuration to
     label it self-collision-free or colliding.
  4. Saves everything to a single .npz file: joint angles, EE pose, and
     the collision label.

This gives a (pose -> joint angles) supervised dataset "for free," without
needing an external dataset download, and lets us control the sampling
distribution (e.g. oversampling near joint limits, where self-collision is
most likely) which a fixed downloaded dataset would not let us do.

Usage:
    python generate_dataset.py --robot xarm6 --n 20000
    python generate_dataset.py --robot kuka_iiwa --n 20000
"""

import argparse
import time

import numpy as np
import pybullet as p

from robot_config import connect, load_robot


def sample_joint_config(joint_limits, rng):
    return np.array([rng.uniform(lo, hi) for (lo, hi) in joint_limits])


def set_joint_config(body_id, joint_indices, q):
    for idx, angle in zip(joint_indices, q):
        p.resetJointState(body_id, idx, angle)


def get_ee_pose(body_id, ee_link_index):
    state = p.getLinkState(body_id, ee_link_index, computeForwardKinematics=True)
    pos = state[4]      # worldLinkFramePosition
    orn = state[5]      # worldLinkFrameOrientation (quaternion xyzw)
    return np.array(pos), np.array(orn)


def is_self_colliding(body_id, margin=0.0):
    """
    Checks the robot against itself for self-collision.
    PyBullet's URDF loader marks adjacent links as non-colliding by default
    (via the URDF's own joint tree), so getContactPoints on itself after
    stepping collision detection correctly reports only *unexpected*
    self-collisions (e.g. wrist folding back into the base link), which is
    exactly what "self-collision prevention" in the project brief means.
    """
    p.performCollisionDetection()
    contacts = p.getContactPoints(bodyA=body_id, bodyB=body_id)
    return len(contacts) > 0


def generate(robot_name, n_samples, seed=0, oversample_frac=0.3):
    rng = np.random.default_rng(seed)
    connect(gui=False)
    body_id, cfg, joint_limits = load_robot(robot_name)
    joint_indices = cfg["joint_indices"]
    ee_link_index = cfg["ee_link_index"]
    dof = cfg["dof"]

    joints = np.zeros((n_samples, dof), dtype=np.float32)
    positions = np.zeros((n_samples, 3), dtype=np.float32)
    orientations = np.zeros((n_samples, 4), dtype=np.float32)
    collision_labels = np.zeros((n_samples,), dtype=np.int64)

    n_uniform = int(n_samples * (1 - oversample_frac))
    n_edge = n_samples - n_uniform

    t0 = time.time()
    idx = 0

    # Phase 1: uniform sampling across the full joint range.
    while idx < n_uniform:
        q = sample_joint_config(joint_limits, rng)
        set_joint_config(body_id, joint_indices, q)
        pos, orn = get_ee_pose(body_id, ee_link_index)
        collided = is_self_colliding(body_id)

        joints[idx] = q
        positions[idx] = pos
        orientations[idx] = orn
        collision_labels[idx] = int(collided)
        idx += 1

    # Phase 2: bias sampling toward joint-limit extremes, where
    # self-collisions are far more likely (the region a collision-aware
    # model most needs good examples from).
    while idx < n_samples:
        q = np.array([
            rng.choice([
                rng.uniform(lo, lo + 0.15 * (hi - lo)),
                rng.uniform(hi - 0.15 * (hi - lo), hi),
            ])
            for (lo, hi) in joint_limits
        ])
        set_joint_config(body_id, joint_indices, q)
        pos, orn = get_ee_pose(body_id, ee_link_index)
        collided = is_self_colliding(body_id)

        joints[idx] = q
        positions[idx] = pos
        orientations[idx] = orn
        collision_labels[idx] = int(collided)
        idx += 1

    elapsed = time.time() - t0
    n_colliding = collision_labels.sum()
    print(f"[{robot_name}] generated {n_samples} samples in {elapsed:.1f}s "
          f"({n_colliding} colliding, {n_colliding / n_samples:.1%})")

    out_path = f"data/{robot_name}_ik_dataset.npz"
    np.savez(
        out_path,
        joints=joints,
        positions=positions,
        orientations=orientations,
        collision_labels=collision_labels,
        joint_limits=np.array(joint_limits, dtype=np.float32),
        dof=dof,
    )
    print(f"saved -> {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="xarm6", choices=["xarm6", "kuka_iiwa", "panda"])
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    generate(args.robot, args.n, args.seed)
