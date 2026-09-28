"""
Robot configuration registry for the Neural IK project (STP Team 06, Level 1).

Keeping robot-specific details (URDF path, which joint indices are the
actuated arm joints, the end-effector link) in one place is what makes the
data-generation and training scripts below robot-agnostic: swapping in a
different 6DOF/7DOF arm later (including, eventually, the actual space
manipulator's URDF) means adding one entry here, not rewriting the pipeline.
"""

import os

import pybullet as p
import pybullet_data

# Local copy of the xarm6 URDF + meshes (downloaded from the bullet3 repo) so
# xarm6 doesn't depend on the pip-installed pybullet_data folder being intact.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Each entry:
#   urdf: path pybullet can resolve (pybullet_data path is added automatically)
#   joint_names: URDF joint names of the actuated arm joints, in kinematic
#                order from base to wrist. Resolved to pybullet joint
#                indices BY NAME at load time (see load_robot below) rather
#                than hardcoded as raw integers - some URDFs (xarm6, for
#                one) insert an extra fixed "world_joint" at index 0, which
#                silently shifts every subsequent joint index by one if you
#                assume indices 0..dof-1. Resolving by name sidesteps that
#                whole class of bug regardless of how a given URDF is laid
#                out, which matters once you swap in a new robot (including,
#                eventually, the actual space manipulator's URDF).
#   dof: convenience count, must equal len(joint_names)
ROBOTS = {
    "xarm6": {
        "urdf": os.path.join(PROJECT_DIR, "urdf", "xarm", "xarm6_robot.urdf"),
        "joint_names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        "dof": 6,
        "description": "UFactory xArm6 - standard 6DOF industrial arm",
    },
    "kuka_iiwa": {
        "urdf": "kuka_iiwa/model.urdf",
        "joint_names": [f"lbr_iiwa_joint_{i}" for i in range(1, 8)],
        "dof": 7,
        "description": "KUKA LBR iiwa - redundant 7DOF arm",
    },
    "panda": {
        "urdf": "franka_panda/panda.urdf",
        "joint_names": [f"panda_joint{i}" for i in range(1, 8)],
        "dof": 7,
        "description": "Franka Emika Panda arm joints only (gripper fingers excluded)",
    },
}


def connect(gui=False):
    """Start a PyBullet physics client (DIRECT = headless, no rendering)."""
    mode = p.GUI if gui else p.DIRECT
    cid = p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    return cid


def load_robot(robot_name):
    """
    Load a robot by name from the registry and return (body_id, cfg, joint_limits).

    cfg is enriched in-place with resolved 'joint_indices' and 'ee_link_index'
    (in pybullet, a link shares its parent joint's index, so the last arm
    joint's index doubles as the end-effector link index).
    """
    if robot_name not in ROBOTS:
        raise ValueError(f"Unknown robot '{robot_name}'. Options: {list(ROBOTS)}")
    cfg = dict(ROBOTS[robot_name])  # copy - don't mutate the shared registry entry
    # URDF_USE_SELF_COLLISION is required, or PyBullet silently skips
    # self-collision checks between a robot's own links entirely -
    # EXCLUDE_PARENT additionally skips the (always-touching) parent/child
    # pair at each joint so it doesn't spuriously flag every configuration.
    body_id = p.loadURDF(
        cfg["urdf"],
        useFixedBase=True,
        flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
    )

    name_to_index = {}
    for j in range(p.getNumJoints(body_id)):
        info = p.getJointInfo(body_id, j)
        name_to_index[info[1].decode()] = j

    missing = [n for n in cfg["joint_names"] if n not in name_to_index]
    if missing:
        raise ValueError(f"Joint name(s) {missing} not found in {cfg['urdf']}")

    joint_indices = [name_to_index[n] for n in cfg["joint_names"]]
    cfg["joint_indices"] = joint_indices
    cfg["ee_link_index"] = joint_indices[-1]

    joint_limits = []
    for j in joint_indices:
        info = p.getJointInfo(body_id, j)
        lower, upper = info[8], info[9]
        # Some URDFs report degenerate limits (lower > upper, or both 0)
        # for continuous joints; fall back to +-pi in that case.
        if lower >= upper:
            lower, upper = -3.14159, 3.14159
        joint_limits.append((lower, upper))

    return body_id, cfg, joint_limits
