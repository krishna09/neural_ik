"""
Paper reproduction - DMLP (dense MLP) with Learning-by-Example.

Demby's, Farag & DeSouza, "Inverse Kinematics of Robotic Manipulators Using a
New Learning-by-Example Method" (IROS 2024).

Network (exactly as the authors' code, github.com/jacketdembys/iksolver,
models_2.py: DenseNet = "DenseMLP3", configured in train.yaml with 1024 neurons,
NUM_BLOCKS 2, NUM_HIDDEN_LAYERS 5 - i.e. two dense blocks of five layers):
  input Linear(19 -> 1024) + ReLU
  dense block 1: 5 layers, layer k = Linear(concat(block input, layers 1..k-1) -> 1024) + ReLU
  transition Linear(6144 -> 1024) + ReLU
  dense block 2: 5 layers (same wiring)
  classifier Linear(6144 -> dof)  = joint angles Q in radians
= 12 hidden layers, the same count as the paper's plain MLP. (The reference
forward() calls self.classifier; its definition is the output Linear from the
final concatenated width, as used here.)

Learning-by-Example input (column order of the paper's "seq" dataset):
  [ De (example pose, 6), Qe (example joints, dof), D (query pose, 6) ]
  pose = [x, y, z, roll, pitch, yaw]; everything min-max scaled to [0, 1].

How it maps onto this project (see README.md in this folder):
  * Qe = the seed joints train_ik already feeds the network (true joints + noise);
  * De = FK(Qe), computed here from the robot's URDF (the paper's De is exactly the
    pose of the example joints);
  * D  = query position + quaternion converted to roll-pitch-yaw ("XYZ", as the
    paper's generator does);
  * min-max ranges are fitted once from 20,000 random configurations (the paper
    fits sklearn's MinMaxScaler on its training set);
  * Q is returned as [sin Q, cos Q] because train_ik's loss compares against
    [sin, cos] targets (the paper's loss is MSE on Q itself).
"""

import math
import os

import numpy as np
import pybullet as p
import pybullet_data
import torch
import torch.nn as nn

from robot_config import ROBOTS

DESCRIPTION = "Paper (IROS 2024): DMLP (DenseMLP3), Learning-by-Example [De, Qe, D] -> Q; 2 dense blocks x 5 layers x 1024."

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
    """Forward kinematics (example pose De = FK(Qe)) and the [0, 1] min-max
    input scaling. All constants are buffers, so they are saved with the model."""

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
# Network - models_2.py DenseBlock / TransitionLayer / DenseNet from the authors' repository
# --------------------------------------------------------------------------

class DenseBlock(nn.Module):
    def __init__(self, in_features, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self.make_dense_layer(in_features + i * growth_rate, growth_rate))

    def make_dense_layer(self, in_features, growth_rate):
        return nn.Sequential(nn.Linear(in_features, growth_rate), nn.ReLU())

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        return torch.cat(features, 1)


class TransitionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(in_features, out_features), nn.ReLU())

    def forward(self, x):
        return self.layer(x)


class DenseNet(nn.Module):
    def __init__(self, in_features, growth_rate, block_config, out_features):
        super().__init__()
        self.input = nn.Linear(in_features, growth_rate)
        self.input_relu = nn.ReLU()
        self.features = nn.Sequential()
        in_features = growth_rate
        for i, num_layers in enumerate(block_config):
            self.features.add_module(f"denseblock_{i + 1}", DenseBlock(in_features, growth_rate, num_layers))
            in_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                self.features.add_module(f"transition_{i + 1}", TransitionLayer(in_features, growth_rate))
                in_features = growth_rate
        self.classifier = nn.Linear(in_features, out_features)

    def forward(self, x):
        out = self.input(x)
        out = self.input_relu(out)
        out = self.features(out)
        return self.classifier(out)


# --------------------------------------------------------------------------

class IKNet(nn.Module):
    def __init__(self, dof, hidden_size=1024, num_blocks=2, layers_per_block=5):
        super().__init__()
        self.dof = dof
        self.codec = PoseCodec(dof)
        self.model = DenseNet(6 + dof + 6, hidden_size, [layers_per_block] * num_blocks, dof)

    def forward(self, x):
        qe = x[:, 7:7 + self.dof]                                   # example joints = seed
        with torch.no_grad():                                       # input encoding, no weights involved
            inp = torch.cat([
                self.codec.scale_pose(self.codec.fk_pose(qe)),      # De
                self.codec.scale_joints(qe),                        # Qe
                self.codec.scale_pose(self.codec.query_pose(x)),    # D
            ], dim=-1)
        q = self.model(inp)                                         # Q (radians), as in the paper
        return torch.cat([torch.sin(q), torch.cos(q)], dim=-1)

    def decode_angles(self, raw_out):
        return torch.atan2(raw_out[:, :self.dof], raw_out[:, self.dof:])
