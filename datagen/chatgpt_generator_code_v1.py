"""
Dataset generation for neural inverse kinematics.

Compatible with the existing project structure:

    from robot_config import connect, load_robot

Expected load_robot() return:

    body_id, cfg, joint_limits

Expected cfg entries:

    cfg["joint_indices"]
    cfg["ee_link_index"]
    cfg["dof"]

Optional cfg entries:

    cfg["collision_link_indices"]
    cfg["self_collision_ignore_pairs"]

Example:
    cfg["self_collision_ignore_pairs"] = [
        (3, 4),
        (7, 8),
    ]

The MAIN dataset contains only collision-safe IK targets.

Rejected colliding / near-colliding samples are stored separately inside
the same NPZ and therefore cannot accidentally become IK ground truth.

Usage:

    python generate_dataset.py --robot xarm6 --n 200000
    python generate_dataset.py --robot kuka_iiwa --n 200000
    python generate_dataset.py --robot panda --n 200000

Recommended larger training set:

    python generate_dataset.py \
        --robot xarm6 \
        --n 500000 \
        --seed 42 \
        --edge-frac 0.25 \
        --clearance-mm 2.0
"""

import argparse
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pybullet as p

from robot_config import connect, load_robot


# ================================================================
# Basic robot utilities
# ================================================================

def sample_joint_config(joint_limits, rng):
    """Uniform sample over the physical joint ranges."""
    return np.asarray(
        [rng.uniform(lo, hi) for lo, hi in joint_limits],
        dtype=np.float64,
    )


def sample_edge_config(
    joint_limits,
    rng,
    edge_band=0.15,
    max_edge_fraction=0.35,
):
    """
    Sample a configuration where ONLY SOME joints are near their limits.

    This fixes the old implementation where every joint was pushed toward
    an extreme simultaneously.
    """

    q = sample_joint_config(joint_limits, rng)

    dof = len(joint_limits)

    max_edge_joints = max(
        1,
        int(math.ceil(dof * max_edge_fraction)),
    )

    n_edge_joints = int(
        rng.integers(
            1,
            max_edge_joints + 1,
        )
    )

    selected = rng.choice(
        dof,
        size=n_edge_joints,
        replace=False,
    )

    for j in selected:

        lo, hi = joint_limits[j]

        width = hi - lo

        if rng.random() < 0.5:

            # near lower limit
            q[j] = rng.uniform(
                lo,
                lo + edge_band * width,
            )

        else:

            # near upper limit
            q[j] = rng.uniform(
                hi - edge_band * width,
                hi,
            )

    return q


def set_joint_config(body_id, joint_indices, q):

    for idx, angle in zip(joint_indices, q):

        p.resetJointState(
            body_id,
            idx,
            targetValue=float(angle),
            targetVelocity=0.0,
        )


def canonicalize_quaternion(quat):
    """
    PyBullet quaternion order:

        x, y, z, w

    q and -q represent the same orientation.

    We choose one deterministic hemisphere so the dataset does not
    randomly contain both representations.
    """

    quat = np.asarray(
        quat,
        dtype=np.float64,
    )

    norm = np.linalg.norm(quat)

    if norm < 1e-12:
        raise RuntimeError("Invalid zero quaternion returned by FK.")

    quat = quat / norm

    # Canonical hemisphere
    if quat[3] < 0.0:
        quat = -quat

    return quat


def quaternion_to_6d(quat):
    """
    Continuous-ish rotation representation:
    first two columns of the 3x3 rotation matrix.

    Quaternion is still stored for backwards compatibility.
    """

    matrix = np.asarray(
        p.getMatrixFromQuaternion(quat.tolist()),
        dtype=np.float64,
    ).reshape(3, 3)

    return np.concatenate(
        [
            matrix[:, 0],
            matrix[:, 1],
        ],
        axis=0,
    )


def get_ee_pose(body_id, ee_link_index):

    state = p.getLinkState(
        body_id,
        ee_link_index,
        computeForwardKinematics=True,
    )

    # worldLinkFramePosition
    pos = np.asarray(
        state[4],
        dtype=np.float64,
    )

    # worldLinkFrameOrientation
    quat = canonicalize_quaternion(
        state[5]
    )

    return pos, quat


# ================================================================
# Collision handling
# ================================================================

def normalise_pair(a, b):

    a = int(a)
    b = int(b)

    if a <= b:
        return a, b

    return b, a


def link_has_collision_geometry(body_id, link_index):

    data = p.getCollisionShapeData(
        body_id,
        link_index,
    )

    return len(data) > 0


def build_self_collision_pairs(body_id, cfg):
    """
    Construct robot link pairs that should actually be tested.

    Excludes:
      - same link
      - parent/child neighbours
      - links without collision geometry
      - user-defined ignored pairs

    Includes base link (-1).
    """

    n_joints = p.getNumJoints(body_id)

    # ------------------------------------------------------------
    # Build parent map
    #
    # Link j corresponds to joint j's child link.
    # jointInfo[16] is the parent link index.
    # ------------------------------------------------------------

    parent = {-1: None}

    for j in range(n_joints):

        info = p.getJointInfo(
            body_id,
            j,
        )

        parent[j] = int(info[16])

    # ------------------------------------------------------------
    # Candidate links
    # ------------------------------------------------------------

    if "collision_link_indices" in cfg:

        candidate_links = [
            int(x)
            for x in cfg["collision_link_indices"]
        ]

    else:

        candidate_links = (
            [-1]
            + list(range(n_joints))
        )

    candidate_links = [
        link
        for link in candidate_links
        if link_has_collision_geometry(
            body_id,
            link,
        )
    ]

    # ------------------------------------------------------------
    # Optional URDF-specific exclusions
    # ------------------------------------------------------------

    ignored = set()

    for pair in cfg.get(
        "self_collision_ignore_pairs",
        [],
    ):

        ignored.add(
            normalise_pair(
                pair[0],
                pair[1],
            )
        )

    pairs = []

    for i in range(len(candidate_links)):

        a = candidate_links[i]

        for j in range(i + 1, len(candidate_links)):

            b = candidate_links[j]

            pair = normalise_pair(a, b)

            # User requested ignore
            if pair in ignored:
                continue

            # Direct parent-child pair
            if (
                parent.get(a) == b
                or parent.get(b) == a
            ):
                continue

            pairs.append(pair)

    return set(pairs)


def configure_self_collision_pairs(
    body_id,
    allowed_pairs,
):
    """
    Explicitly enable the link pairs we want checked.

    This means the generator does not rely solely on assumptions about
    how load_robot() called p.loadURDF().
    """

    for link_a, link_b in allowed_pairs:

        p.setCollisionFilterPair(
            bodyUniqueIdA=body_id,
            bodyUniqueIdB=body_id,
            linkIndexA=link_a,
            linkIndexB=link_b,
            enableCollision=1,
        )


def filter_valid_contacts(
    points,
    allowed_pairs,
):

    valid = []

    for point in points:

        # PyBullet contact/closest-point tuple:
        # index 3 = linkIndexA
        # index 4 = linkIndexB
        # index 8 = contactDistance

        pair = normalise_pair(
            point[3],
            point[4],
        )

        if pair in allowed_pairs:
            valid.append(point)

    return valid


def check_self_collision(
    body_id,
    allowed_pairs,
    clearance_m=0.002,
):
    """
    Returns:

        colliding:
            actual touching / penetration

        near_collision:
            distance <= clearance_m

        minimum_distance:
            minimum observed pair distance inside query radius.
            NaN if nothing is within clearance_m.

    Negative distance = penetration.
    Positive distance = separation.
    """

    p.performCollisionDetection()

    # ------------------------------------------------------------
    # Actual contacts
    # ------------------------------------------------------------

    contacts = p.getContactPoints(
        bodyA=body_id,
        bodyB=body_id,
    )

    contacts = filter_valid_contacts(
        contacts,
        allowed_pairs,
    )

    colliding = False

    minimum_distance = np.inf

    for contact in contacts:

        distance = float(
            contact[8]
        )

        minimum_distance = min(
            minimum_distance,
            distance,
        )

        if distance <= 0.0:
            colliding = True

    # ------------------------------------------------------------
    # Safety clearance
    # ------------------------------------------------------------

    near_collision = colliding

    if clearance_m > 0.0:

        closest = p.getClosestPoints(
            bodyA=body_id,
            bodyB=body_id,
            distance=clearance_m,
        )

        closest = filter_valid_contacts(
            closest,
            allowed_pairs,
        )

        for point in closest:

            distance = float(
                point[8]
            )

            minimum_distance = min(
                minimum_distance,
                distance,
            )

            if distance <= clearance_m:
                near_collision = True

    if np.isinf(minimum_distance):
        minimum_distance = np.nan

    return (
        colliding,
        near_collision,
        minimum_distance,
    )


# ================================================================
# Dataset quality utilities
# ================================================================

def calculate_workspace_weights(
    positions,
    orientations,
    position_bin_m=0.05,
    angle_bin_deg=30.0,
):
    """
    Produce optional training weights that reduce severe Cartesian
    workspace imbalance.

    This DOES NOT modify or discard samples.

    Training code can use these with WeightedRandomSampler.

    Binning Euler angles here is only for rough sampling statistics.
    Euler angles are NOT used as the model's orientation target.
    """

    angle_bin = math.radians(
        angle_bin_deg
    )

    keys = []

    for pos, quat in zip(
        positions,
        orientations,
    ):

        xyz_bin = tuple(
            np.floor(
                pos / position_bin_m
            ).astype(np.int64)
        )

        euler = np.asarray(
            p.getEulerFromQuaternion(
                quat.tolist()
            ),
            dtype=np.float64,
        )

        orientation_bin = tuple(
            np.floor(
                (euler + np.pi)
                / angle_bin
            ).astype(np.int64)
        )

        keys.append(
            xyz_bin
            + orientation_bin
        )

    counts = Counter(keys)

    weights = np.asarray(
        [
            1.0
            / math.sqrt(
                counts[key]
            )
            for key in keys
        ],
        dtype=np.float32,
    )

    # Avoid insane weights from extremely rare cells.
    weights /= max(
        float(weights.mean()),
        1e-8,
    )

    weights = np.clip(
        weights,
        0.25,
        4.0,
    )

    return weights


def make_stratified_split(
    sample_types,
    rng,
    train_frac=0.80,
    val_frac=0.10,
):
    """
    0 = train
    1 = validation
    2 = test

    Stratified by sample strategy so edge samples do not accidentally
    disappear from validation/test.
    """

    split = np.full(
        len(sample_types),
        2,
        dtype=np.uint8,
    )

    for sample_type in np.unique(
        sample_types
    ):

        indices = np.flatnonzero(
            sample_types
            == sample_type
        )

        rng.shuffle(indices)

        n = len(indices)

        n_train = int(
            n * train_frac
        )

        n_val = int(
            n * val_frac
        )

        split[
            indices[:n_train]
        ] = 0

        split[
            indices[
                n_train:
                n_train + n_val
            ]
        ] = 1

        split[
            indices[
                n_train + n_val:
            ]
        ] = 2

    return split


def validate_joint_limits(
    joint_limits,
    dof,
):

    limits = np.asarray(
        joint_limits,
        dtype=np.float64,
    )

    if limits.shape != (dof, 2):

        raise ValueError(
            f"Expected joint_limits shape "
            f"({dof}, 2), got {limits.shape}"
        )

    if not np.all(
        np.isfinite(limits)
    ):

        raise ValueError(
            "Dataset generation requires "
            "finite physical joint limits."
        )

    if np.any(
        limits[:, 1]
        <= limits[:, 0]
    ):

        raise ValueError(
            "Invalid joint limit: upper "
            "must be greater than lower."
        )

    return limits


# ================================================================
# Main generator
# ================================================================

def generate(
    robot_name,
    n_samples,
    seed=0,
    oversample_frac=0.25,
    clearance_mm=2.0,
    reject_near=True,
    collision_pool_size=50000,
):

    rng = np.random.default_rng(
        seed
    )

    connect(
        gui=False
    )

    body_id, cfg, joint_limits = load_robot(
        robot_name
    )

    joint_indices = list(
        cfg["joint_indices"]
    )

    ee_link_index = int(
        cfg["ee_link_index"]
    )

    dof = int(
        cfg["dof"]
    )

    if len(joint_indices) != dof:

        raise RuntimeError(
            f"{robot_name}: cfg['dof']={dof} "
            f"but len(joint_indices)="
            f"{len(joint_indices)}"
        )

    limits = validate_joint_limits(
        joint_limits,
        dof,
    )

    lower = limits[:, 0]
    upper = limits[:, 1]

    clearance_m = (
        clearance_mm / 1000.0
    )

    # ------------------------------------------------------------
    # Build explicit collision topology
    # ------------------------------------------------------------

    collision_pairs = (
        build_self_collision_pairs(
            body_id,
            cfg,
        )
    )

    if len(collision_pairs) == 0:

        raise RuntimeError(
            f"{robot_name}: no self-collision "
            f"link pairs were discovered. "
            f"Check the URDF and robot_config.py."
        )

    configure_self_collision_pairs(
        body_id,
        collision_pairs,
    )

    print(
        f"[{robot_name}] DOF={dof}, "
        f"self-collision pairs="
        f"{len(collision_pairs)}"
    )

    # ------------------------------------------------------------
    # Main SAFE IK dataset
    # ------------------------------------------------------------

    joints = np.zeros(
        (n_samples, dof),
        dtype=np.float32,
    )

    positions = np.zeros(
        (n_samples, 3),
        dtype=np.float32,
    )

    orientations = np.zeros(
        (n_samples, 4),
        dtype=np.float32,
    )

    orientations_6d = np.zeros(
        (n_samples, 6),
        dtype=np.float32,
    )

    sample_types = np.zeros(
        n_samples,
        dtype=np.uint8,
    )

    min_self_distance = np.full(
        n_samples,
        np.nan,
        dtype=np.float32,
    )

    # ------------------------------------------------------------
    # Rejected collision pool
    # ------------------------------------------------------------

    rejected_q = []
    rejected_pos = []
    rejected_quat = []
    rejected_reason = []

    accepted = 0
    attempted = 0

    exact_collision_count = 0
    near_collision_count = 0

    t0 = time.time()

    report_interval = max(
        1,
        n_samples // 10,
    )

    next_report = report_interval

    max_attempts = max(
        n_samples * 100,
        10000,
    )

    while (
        accepted < n_samples
        and attempted < max_attempts
    ):

        attempted += 1

        # --------------------------------------------------------
        # Sampling mode
        #
        # type 0 = uniform joint space
        # type 1 = partial near-limit sampling
        # --------------------------------------------------------

        if rng.random() < oversample_frac:

            q = sample_edge_config(
                limits,
                rng,
            )

            sample_type = 1

        else:

            q = sample_joint_config(
                limits,
                rng,
            )

            sample_type = 0

        # Safety against numerical errors
        q = np.clip(
            q,
            lower,
            upper,
        )

        set_joint_config(
            body_id,
            joint_indices,
            q,
        )

        pos, quat = get_ee_pose(
            body_id,
            ee_link_index,
        )

        (
            colliding,
            near_collision,
            minimum_distance,
        ) = check_self_collision(
            body_id,
            collision_pairs,
            clearance_m=clearance_m,
        )

        # --------------------------------------------------------
        # Collision statistics
        # --------------------------------------------------------

        if colliding:
            exact_collision_count += 1

        elif near_collision:
            near_collision_count += 1

        unsafe_for_ik = (
            colliding
            or (
                reject_near
                and near_collision
            )
        )

        # --------------------------------------------------------
        # Never put unsafe q into IK ground truth
        # --------------------------------------------------------

        if unsafe_for_ik:

            if (
                len(rejected_q)
                < collision_pool_size
            ):

                rejected_q.append(
                    q.astype(
                        np.float32
                    )
                )

                rejected_pos.append(
                    pos.astype(
                        np.float32
                    )
                )

                rejected_quat.append(
                    quat.astype(
                        np.float32
                    )
                )

                # 1 = true collision
                # 2 = clearance violation only
                rejected_reason.append(
                    1
                    if colliding
                    else 2
                )

            continue

        # --------------------------------------------------------
        # SAFE IK sample
        # --------------------------------------------------------

        joints[accepted] = q

        positions[accepted] = pos

        orientations[accepted] = quat

        orientations_6d[accepted] = (
            quaternion_to_6d(
                quat
            )
        )

        sample_types[
            accepted
        ] = sample_type

        min_self_distance[
            accepted
        ] = minimum_distance

        accepted += 1

        if accepted >= next_report:

            elapsed = (
                time.time()
                - t0
            )

            print(
                f"[{robot_name}] "
                f"{accepted:,}/"
                f"{n_samples:,} accepted | "
                f"{attempted:,} attempted | "
                f"{elapsed:.1f}s"
            )

            next_report += (
                report_interval
            )

    if accepted != n_samples:

        raise RuntimeError(
            f"Could only generate "
            f"{accepted}/{n_samples} safe "
            f"samples after "
            f"{attempted} attempts. "
            f"Check collision exclusions "
            f"or reduce clearance."
        )

    # ============================================================
    # Joint normalization
    # ============================================================

    joint_center = (
        lower + upper
    ) / 2.0

    joint_half_range = (
        upper - lower
    ) / 2.0

    joints_normalized = (
        (
            joints.astype(
                np.float64
            )
            - joint_center
        )
        / joint_half_range
    ).astype(
        np.float32
    )

    joints_normalized = np.clip(
        joints_normalized,
        -1.0,
        1.0,
    )

    # ============================================================
    # Workspace balancing weights
    # ============================================================

    print(
        f"[{robot_name}] calculating "
        f"workspace sampling weights..."
    )

    sample_weights = (
        calculate_workspace_weights(
            positions,
            orientations,
        )
    )

    # ============================================================
    # Fixed train / validation / test split
    # ============================================================

    split = make_stratified_split(
        sample_types,
        rng,
    )

    # ============================================================
    # Rejected pool conversion
    # ============================================================

    if len(rejected_q) > 0:

        collision_joints = np.stack(
            rejected_q
        )

        collision_positions = np.stack(
            rejected_pos
        )

        collision_orientations = np.stack(
            rejected_quat
        )

        collision_reason = np.asarray(
            rejected_reason,
            dtype=np.uint8,
        )

    else:

        collision_joints = np.empty(
            (0, dof),
            dtype=np.float32,
        )

        collision_positions = np.empty(
            (0, 3),
            dtype=np.float32,
        )

        collision_orientations = np.empty(
            (0, 4),
            dtype=np.float32,
        )

        collision_reason = np.empty(
            (0,),
            dtype=np.uint8,
        )

    # ============================================================
    # Backward-compatible labels
    #
    # Main arrays are guaranteed collision-free.
    # Therefore every main IK row gets collision_labels = 0.
    # ============================================================

    collision_labels = np.zeros(
        n_samples,
        dtype=np.int64,
    )

    # ============================================================
    # Save
    # ============================================================

    data_dir = Path("data")

    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path = (
        data_dir
        / f"{robot_name}_ik_dataset.npz"
    )

    np.savez(
        out_path,

        # --------------------------------------------------------
        # Existing / backward-compatible fields
        # --------------------------------------------------------

        joints=joints,

        positions=positions,

        orientations=orientations,

        collision_labels=collision_labels,

        joint_limits=limits.astype(
            np.float32
        ),

        dof=np.int64(
            dof
        ),

        # --------------------------------------------------------
        # New training-safe fields
        # --------------------------------------------------------

        joints_normalized=(
            joints_normalized
        ),

        orientations_6d=(
            orientations_6d
        ),

        sample_weights=(
            sample_weights
        ),

        sample_type=(
            sample_types
        ),

        split=split,

        min_self_distance=(
            min_self_distance
        ),

        joint_center=(
            joint_center.astype(
                np.float32
            )
        ),

        joint_half_range=(
            joint_half_range.astype(
                np.float32
            )
        ),

        # --------------------------------------------------------
        # Collision data kept OUTSIDE IK ground truth
        # --------------------------------------------------------

        rejected_joints=(
            collision_joints
        ),

        rejected_positions=(
            collision_positions
        ),

        rejected_orientations=(
            collision_orientations
        ),

        rejected_reason=(
            collision_reason
        ),

        # --------------------------------------------------------
        # Metadata
        # --------------------------------------------------------

        dataset_version=np.array(
            "ik_dataset_v2"
        ),

        robot_name=np.array(
            robot_name
        ),

        quaternion_order=np.array(
            "xyzw"
        ),

        clearance_m=np.float32(
            clearance_m
        ),

        reject_near_collision=np.uint8(
            int(reject_near)
        ),
    )

    elapsed = (
        time.time()
        - t0
    )

    print()
    print("=" * 70)

    print(
        f"[{robot_name}] DATASET COMPLETE"
    )

    print(
        f"Safe IK samples       : "
        f"{n_samples:,}"
    )

    print(
        f"Candidate attempts     : "
        f"{attempted:,}"
    )

    print(
        f"Exact collisions seen  : "
        f"{exact_collision_count:,}"
    )

    print(
        f"Near collisions seen   : "
        f"{near_collision_count:,}"
    )

    print(
        f"Rejected pool saved    : "
        f"{len(collision_joints):,}"
    )

    print(
        f"Uniform samples        : "
        f"{np.sum(sample_types == 0):,}"
    )

    print(
        f"Near-limit samples     : "
        f"{np.sum(sample_types == 1):,}"
    )

    print(
        f"Train                  : "
        f"{np.sum(split == 0):,}"
    )

    print(
        f"Validation             : "
        f"{np.sum(split == 1):,}"
    )

    print(
        f"Test                   : "
        f"{np.sum(split == 2):,}"
    )

    print(
        f"Elapsed                : "
        f"{elapsed:.1f}s"
    )

    print(
        f"Saved                  : "
        f"{out_path}"
    )

    print("=" * 70)

    return str(out_path)


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--robot",
        default="xarm6",
        choices=[
            "xarm6",
            "kuka_iiwa",
            "panda",
        ],
    )

    parser.add_argument(
        "--n",
        type=int,
        default=20000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--edge-frac",
        type=float,
        default=0.25,
        help=(
            "Fraction of samples using "
            "partial near-joint-limit sampling."
        ),
    )

    parser.add_argument(
        "--clearance-mm",
        type=float,
        default=2.0,
        help=(
            "Minimum self-clearance used "
            "for rejecting near-collision "
            "IK training samples."
        ),
    )

    parser.add_argument(
        "--allow-near",
        action="store_true",
        help=(
            "Allow non-colliding configurations "
            "that fall inside clearance-mm. "
            "Normally leave this OFF."
        ),
    )

    parser.add_argument(
        "--collision-pool-size",
        type=int,
        default=50000,
    )

    args = parser.parse_args()

    if not (
        0.0
        <= args.edge_frac
        <= 1.0
    ):
        raise ValueError(
            "--edge-frac must be "
            "between 0 and 1"
        )

    if args.clearance_mm < 0.0:

        raise ValueError(
            "--clearance-mm cannot "
            "be negative"
        )

    generate(
        robot_name=args.robot,
        n_samples=args.n,
        seed=args.seed,
        oversample_frac=args.edge_frac,
        clearance_mm=args.clearance_mm,
        reject_near=not args.allow_near,
        collision_pool_size=(
            args.collision_pool_size
        ),
    )