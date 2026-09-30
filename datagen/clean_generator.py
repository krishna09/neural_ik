"""
Clean self-collision dataset generator.

Made for 7-axis arms with a gripper (e.g. the Franka Panda), and works for every
robot in robot_config.py. generate_dataset.py stays the base generator; this is
an alternative selectable in the UI (Generate Dataset -> Generator).

Why a second generator
----------------------
generate_dataset.py labels a configuration "colliding" if ANY two links of the
URDF touch, except parent/child pairs. On the Panda some parts touch in every
configuration by construction - panda_hand is bolted to panda_link7 through a
fixed joint, and the two closed gripper fingers touch each other - so every
sample is labelled colliding and the IK network has nothing to learn from.

What this generator does differently
------------------------------------
1. Self-collision matrix, built the way MoveIt's Setup Assistant builds one:
   * rigid   - links joined by fixed joints move as one body; never checked
               against each other;
   * adjacent - rigid groups connected by one movable joint (PyBullet's
               parent/child exclusion, generalised to rigid groups); not checked;
   * always  - pairs in contact in >= 95 % of 2,000 random configurations are
               structural overlaps of the meshes, not real collisions; not checked.
   Every other pair - real self-collisions such as the wrist folding into the
   base - is checked exactly as generate_dataset.py checks it.
2. Even coverage: the joint space is sampled with a scrambled Sobol
   (low-discrepancy) sequence instead of independent uniform draws, plus the same
   near-joint-limit oversampling idea as generate_dataset.py (20 % by default).
3. Same output: the .npz has exactly the keys generate_dataset.py writes
   (joints, positions, orientations, collision_labels, joint_limits, dof), so
   train_ik.py reads it unchanged.

Interface used by the UI (see README.md):
    generate(robot_name, n_samples, seed=0, out_path=None, progress=None) -> path
    progress(i, n, q, pos, orn, collided, phase) is called after every sample.
    LAST_REPORT holds the collision matrix of the last call.
"""

import itertools
import time
from collections import Counter

import numpy as np
import pybullet as p
import torch

from robot_config import connect, load_robot

DESCRIPTION = ("Clean self-collision generator: MoveIt-style collision matrix (rigid / adjacent / "
               "always-touching link pairs ignored) + Sobol sampling. Use for 7-axis arms with grippers.")
ID_TAG = "clean"

ALWAYS_THRESHOLD = 0.95    # a pair touching in >= 95 % of random poses is structural
ANALYSIS_SAMPLES = 2000    # random poses used to find always-touching pairs
EDGE_FRAC = 0.2            # share of samples drawn near the joint limits
EDGE_BAND = 0.15           # "near the limit" = within 15 % of the joint range

LAST_REPORT = {}


# --------------------------------------------------------------------------
# Self-collision matrix
# --------------------------------------------------------------------------

def _link_name(body, link):
    return p.getBodyInfo(body)[0].decode() if link < 0 else p.getJointInfo(body, link)[12].decode()


def _rigid_groups(body):
    """Union-find over fixed joints: links that can never move relative to each other."""
    n = p.getNumJoints(body)
    parent = {link: link for link in range(-1, n)}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for j in range(n):
        info = p.getJointInfo(body, j)
        if info[2] == p.JOINT_FIXED:
            parent[find(j)] = find(info[16])
    return {link: find(link) for link in parent}


def _set_joints(body, indices, q):
    for idx, a in zip(indices, q):
        p.resetJointState(body, idx, float(a))


def _contact_pairs(body):
    p.performCollisionDetection()
    return {tuple(sorted((c[3], c[4]))) for c in p.getContactPoints(bodyA=body, bodyB=body)}


def build_collision_matrix(body, cfg, limits, seed):
    """Disable structural link pairs on `body`; returns the report."""
    n = p.getNumJoints(body)
    links = list(range(-1, n))
    group = _rigid_groups(body)
    members = {}
    for link, g in group.items():
        members.setdefault(g, []).append(link)

    disabled = {}
    for a, b in itertools.combinations(links, 2):
        if group[a] == group[b]:
            disabled[(a, b)] = "rigid (fixed joint)"
    for j in range(n):
        info = p.getJointInfo(body, j)
        if info[2] == p.JOINT_FIXED:
            continue
        for a in members[group[info[16]]]:
            for b in members[group[j]]:
                disabled.setdefault(tuple(sorted((a, b))), "adjacent (one movable joint apart)")

    # Contacts over random poses with only PyBullet's parent/child exclusion active.
    rng = np.random.default_rng(seed + 12345)
    lo, hi = np.array(limits).T
    per_sample, counts = [], Counter()
    for _ in range(ANALYSIS_SAMPLES):
        _set_joints(body, cfg["joint_indices"], rng.uniform(lo, hi))
        pairs = _contact_pairs(body)
        per_sample.append(pairs)
        counts.update(pairs)
    for pair, c in counts.items():
        if c >= ALWAYS_THRESHOLD * ANALYSIS_SAMPLES:
            disabled.setdefault(pair, f"always touching ({c / ANALYSIS_SAMPLES:.0%} of random poses)")

    for a, b in disabled:
        p.setCollisionFilterPair(body, body, a, b, enableCollision=0)

    before = float(np.mean([len(s) > 0 for s in per_sample]))
    after = float(np.mean([len(s - disabled.keys()) > 0 for s in per_sample]))
    touching = sorted(((pr, reason) for pr, reason in disabled.items() if counts.get(pr)),
                      key=lambda x: -counts[x[0]])
    return {
        "analysis_samples": ANALYSIS_SAMPLES,
        "collision_rate_base_rule": before,      # generate_dataset.py's rule on the same poses
        "collision_rate_clean": after,
        "disabled_pair_count": len(disabled),
        "disabled_pairs": [list(pr) for pr in sorted(disabled)],     # link indices, e.g. for cfg ignore lists
        "disabled_touching_pairs": [
            {"a": _link_name(body, a), "b": _link_name(body, b), "reason": reason,
             "touching_in": counts[(a, b)] / ANALYSIS_SAMPLES}
            for (a, b), reason in touching
        ],
        "checked_pairs_seen_colliding": [
            {"a": _link_name(body, a), "b": _link_name(body, b), "touching_in": c / ANALYSIS_SAMPLES}
            for (a, b), c in counts.most_common() if (a, b) not in disabled
        ][:10],
    }


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

def sample_configs(n, limits, seed, edge_frac=EDGE_FRAC):
    """Scrambled Sobol points over the joint limits, then near-limit samples."""
    lo, hi = np.array(limits).T
    dof = len(lo)
    n_edge = int(n * edge_frac)
    n_sobol = n - n_edge
    sobol = torch.quasirandom.SobolEngine(dimension=dof, scramble=True, seed=seed)
    q_sobol = lo + (hi - lo) * sobol.draw(n_sobol).double().numpy()
    rng = np.random.default_rng(seed)
    band = EDGE_BAND * (hi - lo)
    low_side = rng.random((n_edge, dof)) < 0.5
    offset = rng.random((n_edge, dof)) * band
    q_edge = np.where(low_side, lo + offset, hi - offset)
    return np.concatenate([q_sobol, q_edge]), n_sobol


# --------------------------------------------------------------------------

def generate(robot_name, n_samples, seed=0, out_path=None, progress=None):
    out_path = out_path or f"data/{robot_name}_ik_dataset.npz"
    t0 = time.time()
    cid = connect(gui=False)
    try:
        body, cfg, limits = load_robot(robot_name)
        report = build_collision_matrix(body, cfg, limits, seed)
        dof = cfg["dof"]
        Q, n_sobol = sample_configs(n_samples, limits, seed)

        joints = np.zeros((n_samples, dof), dtype=np.float32)
        positions = np.zeros((n_samples, 3), dtype=np.float32)
        orientations = np.zeros((n_samples, 4), dtype=np.float32)
        labels = np.zeros((n_samples,), dtype=np.int64)
        for i, q in enumerate(Q):
            _set_joints(body, cfg["joint_indices"], q)
            st = p.getLinkState(body, cfg["ee_link_index"], computeForwardKinematics=True)
            p.performCollisionDetection()
            collided = len(p.getContactPoints(bodyA=body, bodyB=body)) > 0
            joints[i], positions[i], orientations[i], labels[i] = q, st[4], st[5], int(collided)
            if progress:
                progress(i + 1, n_samples, q, st[4], st[5], collided, "sobol" if i < n_sobol else "near-limits")
    finally:
        p.disconnect(cid)

    np.savez(out_path, joints=joints, positions=positions, orientations=orientations,
             collision_labels=labels, joint_limits=np.array(limits, dtype=np.float32), dof=dof)
    report.update(robot=robot_name, n_samples=n_samples, seed=seed, n_colliding=int(labels.sum()),
                  sampling=f"scrambled Sobol ({n_sobol}) + near-limits ({n_samples - n_sobol})",
                  secs=round(time.time() - t0, 2))
    LAST_REPORT.clear()
    LAST_REPORT.update(report)
    print(f"[{robot_name}] clean generator: {n_samples} samples in {report['secs']:.1f}s "
          f"({labels.sum()} colliding, {labels.mean():.1%}); ignored {report['disabled_pair_count']} "
          f"structural link pairs (base rule would label {report['collision_rate_base_rule']:.0%} colliding)")
    print(f"saved -> {out_path}")
    return out_path
