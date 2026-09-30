"""
Forward kinematics and the geometric Jacobian of a robot arm in PyTorch,
built from its URDF (via robot_config.ROBOTS), for batches of joint vectors.

The chain is read once from PyBullet in the zero configuration - the pose of
each joint frame relative to the previous one and each joint's axis - so the
end-effector pose is exactly PyBullet's URDF link frame of the end-effector
link (getLinkState(...)[4:6], what generate_dataset.py stores). Revolute
joints only (all arms in robot_config.py are revolute).
"""

import math
import os
import sys

import numpy as np
import pybullet as p
import pybullet_data
import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from robot_config import ROBOTS  # noqa: E402


def _frame(pos, orn):
    T = np.eye(4)
    T[:3, :3] = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    T[:3, 3] = pos
    return T


def robot_chain(robot):
    """(root 4x4, joint-to-joint transforms (n,4,4), joint axes (n,3), limits (n,2))
    of `robot`, read with a private PyBullet client (the caller's is untouched)."""
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
            if info[2] != p.JOINT_REVOLUTE:
                raise ValueError(f"{robot}: joint {info[1].decode()} is not revolute")
            # same fallback as robot_config.load_robot for degenerate limits
            limits.append((info[8], info[9]) if info[8] < info[9] else (-3.14159, 3.14159))
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
        # robot_config.load_robot: the end-effector link is the last arm joint's link
        rel_ee = np.eye(4)
    finally:
        p.disconnect(cid)
    return root, np.stack(rel), np.array(axes, dtype=float), np.array(limits, dtype=float), rel_ee


def quat_to_matrix(q):
    """(B,4) quaternions x, y, z, w -> (B,3,3) rotation matrices."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z, w = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], dim=-1).view(-1, 3, 3)


def rot_err(R_target, R):
    """Rotation error as a vector (axis * angle, world frame) of R_target R^T; its norm is the angle."""
    E = R_target @ R.transpose(1, 2)
    cos = ((E.diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2).clamp(-1.0, 1.0)
    th = torch.acos(cos)
    v = torch.stack([E[:, 2, 1] - E[:, 1, 2], E[:, 0, 2] - E[:, 2, 0], E[:, 1, 0] - E[:, 0, 1]], dim=-1)
    s = torch.sin(th)
    scale = torch.where(th < 1e-6, torch.full_like(th, 0.5), th / (2 * s.clamp_min(1e-12)))
    out = v * scale[:, None]
    # near 180 deg the antisymmetric part vanishes: take the axis from the symmetric part
    near_pi = th > math.pi - 1e-3
    if near_pi.any():
        B = (E[near_pi] + E[near_pi].transpose(1, 2)) / 2 - torch.eye(3, dtype=E.dtype, device=E.device) * cos[near_pi, None, None]
        axis = B / (1 - cos[near_pi, None, None]).clamp_min(1e-12)
        k = axis.diagonal(dim1=1, dim2=2).argmax(-1)
        a = axis[torch.arange(len(k)), k]
        a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        out[near_pi] = a * th[near_pi, None]
    return out


class Chain:
    """Batched FK / Jacobian of one robot on one device and dtype."""

    def __init__(self, robot, device="cpu", dtype=torch.float64):
        root, rel, axes, limits, rel_ee = robot_chain(robot)
        self.robot = robot
        self.device, self.dtype = torch.device(device), dtype
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=self.device)
        self.root, self.rel, self.axes, self.rel_ee = t(root), t(rel), t(axes), t(rel_ee)
        self.limits = t(limits)
        self.lo, self.hi = self.limits[:, 0], self.limits[:, 1]
        self.dof = len(axes)
        K = torch.zeros(self.dof, 3, 3, dtype=dtype, device=self.device)
        a = self.axes
        K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -a[:, 2], a[:, 1], a[:, 2]
        K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -a[:, 0], -a[:, 1], a[:, 0]
        self.K, self.KK = K, K @ K
        self.eye3 = torch.eye(3, dtype=dtype, device=self.device)

    def to(self, device=None, dtype=None):
        c = object.__new__(Chain)
        c.__dict__.update(self.__dict__)
        c.device = torch.device(device) if device is not None else self.device
        c.dtype = dtype or self.dtype
        for k in ("root", "rel", "axes", "rel_ee", "limits", "lo", "hi", "K", "KK", "eye3"):
            setattr(c, k, getattr(self, k).to(device=c.device, dtype=c.dtype))
        return c

    def _walk(self, q, jac):
        B = q.shape[0]
        R = self.root[:3, :3].expand(B, 3, 3)
        t = self.root[:3, 3].expand(B, 3)
        zs, ps = [], []
        s, c = torch.sin(q), torch.cos(q)
        for i in range(self.dof):
            Ri, ti = self.rel[i, :3, :3], self.rel[i, :3, 3]
            t = t + R @ ti
            R = R @ Ri
            if jac:
                zs.append(R @ self.axes[i])
                ps.append(t)
            rot = self.eye3 + s[:, i, None, None] * self.K[i] + (1 - c[:, i, None, None]) * self.KK[i]
            R = R @ rot
        t = t + R @ self.rel_ee[:3, 3]
        R = R @ self.rel_ee[:3, :3]
        return t, R, zs, ps

    def fk(self, q):
        """q (B,dof) -> end-effector position (B,3), rotation (B,3,3). Differentiable."""
        t, R, _, _ = self._walk(q, False)
        return t, R

    def fk_jac(self, q):
        """-> position, rotation and the 6 x dof geometric Jacobian [z_i x (p - p_i); z_i]."""
        t, R, zs, ps = self._walk(q, True)
        Z, P = torch.stack(zs, 2), torch.stack(ps, 2)
        J = torch.cat([torch.cross(Z, t[:, :, None] - P, dim=1), Z], dim=1)
        return t, R, J
