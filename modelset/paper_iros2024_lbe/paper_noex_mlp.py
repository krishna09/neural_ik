"""
Paper reproduction - Plain MLP WITHOUT example (the paper's Fig. 1a baseline).

Demby's, Farag & DeSouza, "Inverse Kinematics of Robotic Manipulators Using a
New Learning-by-Example Method" (IROS 2024) - "Learning without Example".

Network (exactly as the authors' code, github.com/jacketdembys/iksolver,
utils.py: MLP, with the paper's 12 hidden layers of 1024 neurons):
  input_fc Linear(6 -> 1024) + ReLU
  11 x hidden_fc Linear(1024 -> 1024) + ReLU
  output_fc Linear(1024 -> dof)  = joint angles Q in radians

Input: only the query pose D = [x, y, z, roll, pitch, yaw], min-max scaled to
[0, 1]. The seed joints train_ik provides are deliberately ignored - this is the
scheme the paper shows to fail (tens of mm error) and improves on with
Learning-by-Example (paper_lbe_*.py).

How it maps onto this project (see README.md in this folder): query quaternion converted to
roll-pitch-yaw ("XYZ"); min-max ranges fitted once from 20,000 random
configurations; Q returned as [sin Q, cos Q] for train_ik's loss.
"""

import math
import os

import numpy as np
import pybullet as p
import pybullet_data
import torch
import torch.nn as nn

from robot_config import ROBOTS

DESCRIPTION = "Paper (IROS 2024) baseline: Plain MLP WITHOUT example, D -> Q; 12 hidden layers x 1024, ReLU."

DEFAULT_ROBOT_FOR_DOF = {6: "xarm6", 7: "kuka_iiwa"}   # used when the UI does not pass IK_ROBOT
SCALER_SAMPLES = 20000


# --------------------------------------------------------------------------
# Robot kinematics + input encoding (shared by all paper_* models)
# --------------------------------------------------------------------------

def resolve_robot(dof):
    robot = os.environ.get("IK_ROBOT")
    if robot in ROBOTS and ROBOTS[robot]["dof"] == dof:
        return robot
    return DEFAULT_ROBOT_FOR_DOF[dof]


def _frame(pos, orn):
    T = np.eye(4)
    T[:3, :3] = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    T[:3, 3] = pos
    return T


def robot_chain(robot):
    """Zero-configuration kinematic chain (root frame, joint-to-joint transforms,
    joint axes) and joint limits of `robot`, read from its URDF with PyBullet."""
    cfg = ROBOTS[robot]
    cid = p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
        body = p.loadURDF(cfg["urdf"], useFixedBase=True, physicsClientId=cid)
        names = {p.getJointInfo(body, j, physicsClientId=cid)[1].decode(): j
                 for j in range(p.getNumJoints(body, physicsClientId=cid))}
        idx = [names[n] for n in cfg["joint_names"]]
        limits = []
        for j in idx:
            info = p.getJointInfo(body, j, physicsClientId=cid)
            limits.append((info[8], info[9]) if info[8] < info[9] else (-math.pi, math.pi))
            p.resetJointState(body, j, 0.0, physicsClientId=cid)

        def link_frame(link):
            if link < 0:
                return np.eye(4)                  # fixed base loaded at the origin
            st = p.getLinkState(body, link, computeForwardKinematics=True, physicsClientId=cid)
            return _frame(st[4], st[5])

        prev = link_frame(p.getJointInfo(body, idx[0], physicsClientId=cid)[16])   # parent of joint 1
        root, rel, axes = prev.copy(), [], []
        for j in idx:
            Tj = link_frame(j)
            rel.append(np.linalg.inv(prev) @ Tj)
            axes.append(p.getJointInfo(body, j, physicsClientId=cid)[13])
            prev = Tj
    finally:
        p.disconnect(cid)
    return root, np.stack(rel), np.array(axes, dtype=float), np.array(limits, dtype=float)


def quat_to_matrix(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z, w = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], dim=-1).view(-1, 3, 3)


def matrix_to_rpy(R):
    """Euler angles in the "XYZ" convention of pytorch3d's matrix_to_euler_angles
    (what the paper's dataset generator uses)."""
    return torch.stack([
        torch.atan2(-R[:, 1, 2], R[:, 2, 2]),
        torch.asin(R[:, 0, 2].clamp(-1.0, 1.0)),
        torch.atan2(-R[:, 0, 1], R[:, 0, 0]),
    ], dim=-1)


def _f32(a):
    return torch.tensor(np.asarray(a), dtype=torch.float32)


class PoseCodec(nn.Module):
    """Forward kinematics (to fit the pose ranges) and the [0, 1] min-max input
    scaling. All constants are buffers, so they are saved with the model."""

    def __init__(self, dof):
        super().__init__()
        root, rel, axes, limits = robot_chain(resolve_robot(dof))
        K = np.zeros((dof, 3, 3))
        for i, (ax, ay, az) in enumerate(axes):
            K[i] = [[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]]
        self.register_buffer("root", _f32(root))
        self.register_buffer("rel", _f32(rel))
        self.register_buffer("K", _f32(K))
        self.register_buffer("K2", _f32(K @ K))
        self.register_buffer("joint_min", _f32(limits[:, 0]))
        self.register_buffer("joint_max", _f32(limits[:, 1]))
        gen = torch.Generator().manual_seed(0)
        q = self.joint_min + (self.joint_max - self.joint_min) * torch.rand(SCALER_SAMPLES, dof, generator=gen)
        with torch.no_grad():
            poses = self.fk_pose(q)
        self.register_buffer("pose_min", poses.min(dim=0).values)
        self.register_buffer("pose_max", poses.max(dim=0).values)

    def fk_pose(self, q):
        """Joint angles (B, dof) -> end-effector pose (B, 6) = [x, y, z, roll, pitch, yaw]."""
        T = self.root.expand(q.shape[0], 4, 4)
        eye = torch.eye(3, device=q.device)
        for i in range(q.shape[1]):
            s = torch.sin(q[:, i])[:, None, None]
            c = torch.cos(q[:, i])[:, None, None]
            R = eye + s * self.K[i] + (1 - c) * self.K2[i]
            T = T @ self.rel[i]
            T = torch.cat([torch.cat([T[:, :3, :3] @ R, T[:, :3, 3:]], dim=2), T[:, 3:]], dim=1)
        return torch.cat([T[:, :3, 3], matrix_to_rpy(T[:, :3, :3])], dim=-1)

    def query_pose(self, x):
        """Network input x = [xyz, quaternion, seed] -> query pose [xyz, roll, pitch, yaw]."""
        return torch.cat([x[:, :3], matrix_to_rpy(quat_to_matrix(x[:, 3:7]))], dim=-1)

    def scale_pose(self, pose):
        return (pose - self.pose_min) / (self.pose_max - self.pose_min).clamp_min(1e-6)

    def scale_joints(self, q):
        return (q - self.joint_min) / (self.joint_max - self.joint_min).clamp_min(1e-6)


# --------------------------------------------------------------------------
# Network - utils.py MLP from the authors' repository
# --------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim, h_sizes, output_dim):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, h_sizes[0])
        self.hidden_fc = nn.ModuleList()
        for i in range(len(h_sizes) - 1):
            self.hidden_fc.append(nn.Linear(h_sizes[i], h_sizes[i + 1]))
        self.output_fc = nn.Linear(h_sizes[-1], output_dim)
        self.relu_activation = nn.ReLU()

    def forward(self, x):
        x = self.input_fc(x)
        x = self.relu_activation(x)
        for fc in self.hidden_fc:
            x = fc(x)
            x = self.relu_activation(x)
        return self.output_fc(x)


# --------------------------------------------------------------------------

class IKNet(nn.Module):
    def __init__(self, dof, hidden_size=1024, hidden_layers=12):
        super().__init__()
        self.dof = dof
        self.codec = PoseCodec(dof)
        self.model = MLP(6, [hidden_size] * hidden_layers, dof)

    def forward(self, x):
        with torch.no_grad():                                       # input encoding, no weights involved
            inp = self.codec.scale_pose(self.codec.query_pose(x))  # D only - the seed is ignored
        q = self.model(inp)                                         # Q (radians), as in the paper
        return torch.cat([torch.sin(q), torch.cos(q)], dim=-1)

    def decode_angles(self, raw_out):
        return torch.atan2(raw_out[:, :self.dof], raw_out[:, self.dof:])
