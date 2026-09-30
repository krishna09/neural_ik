"""
Precision IK network - the learned half of "network + numerical refinement".

Why this model exists
---------------------
A regression network alone does not reach < 1 mm / < 1 deg (the best network
here, M10 on 1M samples, stops at a 3.8 mm / 1.1 deg median; published
networks stop at a few mm too). The published systems that do reach it let a
network give a starting point and finish with a numerical solver (Lu, Zou &
Jiang 2022 on the xArm6: network + Newton-Raphson). In this project the solver
is refine/lm_solver.py, which the UI runs after the network in evaluation,
inference and the final test of every training run.

The solver converges when its start is in the right basin: the right IK branch
(elbow up / down, wrist flip, ...) and roughly the right angles. So this
network is trained to be a robust *starting point* for any seed, not only for
seeds that are already close to the answer:

  * Trunk: the M10 design (8 pre-norm SwiGLU residual blocks, width 384), the
    best-measured network in this project. Inputs: position x 1.25, 6D rotation
    (continuous, same for q and -q), sin / cos of the seed.
  * Two heads blended by a learned per-joint gate: seed + correction (precise
    when the seed is close) and a direct angle prediction (usable when the
    seed is far or meaningless).
  * Seed augmentation during training (train_ik always gives seed = truth +
    N(0, 0.3)); the model replaces the seed of each training row with
      60 %  the given seed                      (warm start, as in tracking)
      25 %  the given seed + N(0, 0.7) more      (far seed, ~45 deg per joint)
      15 %  a random pose within the joint limits (no information - cold start)
    so the gate learns when to trust the seed. Validation / evaluation /
    inference (eval mode) use the seed exactly as given.

Robot: the joint limits for the random seeds come from the robot's URDF; the
UI passes the robot in IK_ROBOT (default by DOF: 6 -> xarm6, 7 -> kuka_iiwa).
"""

import os

import pybullet as p
import pybullet_data
import torch
import torch.nn as nn
import torch.nn.functional as F

DESCRIPTION = ("Precision IK starting-point network for LM refinement: M10 SwiGLU trunk (8 x 384), seed+delta / direct "
               "heads with a learned gate, trained with seed augmentation (60 % given / 25 % far / 15 % random seeds).")

POS_SCALE = 1.25
SEED_KEEP, SEED_FAR = 0.60, 0.25          # the remaining 15 % get a random seed
FAR_NOISE_STD = 0.7                        # rad, added on top of train_ik's N(0, 0.3)
DEFAULT_ROBOT_FOR_DOF = {6: "xarm6", 7: "kuka_iiwa"}


def resolve_robot(dof):
    from robot_config import ROBOTS
    robot = os.environ.get("IK_ROBOT")
    if robot in ROBOTS and ROBOTS[robot]["dof"] == dof:
        return robot
    return DEFAULT_ROBOT_FOR_DOF[dof]


def joint_limits(robot):
    """(lo, hi) lists from the URDF, with robot_config.load_robot's fallback, via a private PyBullet client."""
    from robot_config import ROBOTS
    cfg = ROBOTS[robot]
    cid = p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
        body = p.loadURDF(cfg["urdf"], useFixedBase=True, physicsClientId=cid)
        names = {p.getJointInfo(body, j, physicsClientId=cid)[1].decode(): j
                 for j in range(p.getNumJoints(body, physicsClientId=cid))}
        lims = []
        for n in cfg["joint_names"]:
            info = p.getJointInfo(body, names[n], physicsClientId=cid)
            lims.append((info[8], info[9]) if info[8] < info[9] else (-3.14159, 3.14159))
    finally:
        p.disconnect(cid)
    return [a for a, _ in lims], [b for _, b in lims]


def quat_to_rot6d(q):
    """Quaternion (x, y, z, w) -> first two rotation-matrix columns (continuous, sign-invariant)."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z, w = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w),
        2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w),
    ], dim=-1)


def zero_linear(n_in, n_out):
    layer = nn.Linear(n_in, n_out)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class SwiGLUBlock(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc_in = nn.Linear(width, 2 * hidden)
        self.fc_out = nn.Linear(hidden, width)

    def forward(self, h):
        a, b = self.fc_in(self.norm(h)).chunk(2, dim=-1)
        return h + self.fc_out(F.silu(a) * b)


class IKNet(nn.Module):
    def __init__(self, dof, width=384, hidden=768, depth=8):
        super().__init__()
        self.dof = dof
        self.robot = resolve_robot(dof)
        self._lims = None                           # joint limits, loaded on first training step
        self.stem = nn.Linear(9 + 2 * dof, width)
        self.blocks = nn.Sequential(*[SwiGLUBlock(width, hidden) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.delta_head = zero_linear(width, dof)
        self.direct_head = nn.Linear(width, 2 * dof)
        self.gate_head = nn.Linear(width, dof)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -3.0)

    def _augment(self, seed):
        if self._lims is None or self._lims[0].device != seed.device:
            lo, hi = joint_limits(self.robot)
            self._lims = (torch.tensor(lo, device=seed.device, dtype=seed.dtype),
                          torch.tensor(hi, device=seed.device, dtype=seed.dtype))
        lo, hi = self._lims
        u = torch.rand(seed.shape[0], 1, device=seed.device)
        far = seed + FAR_NOISE_STD * torch.randn_like(seed)
        rnd = lo + (hi - lo) * torch.rand_like(seed)
        return torch.where(u < SEED_KEEP, seed, torch.where(u < SEED_KEEP + SEED_FAR, far, rnd))

    def forward(self, x):
        seed = x[:, 7:7 + self.dof]
        if self.training:
            seed = self._augment(seed)
        feats = torch.cat([x[:, :3] * POS_SCALE, quat_to_rot6d(x[:, 3:7]), torch.sin(seed), torch.cos(seed)], dim=-1)
        h = self.norm(self.blocks(self.stem(feats)))
        q_res = seed + self.delta_head(h)
        s_r, c_r = torch.sin(q_res), torch.cos(q_res)
        raw = self.direct_head(h)
        s_d, c_d = raw[:, :self.dof], raw[:, self.dof:]
        n_d = torch.sqrt(s_d * s_d + c_d * c_d + 1e-6)
        g = torch.sigmoid(self.gate_head(h))
        s = (1 - g) * s_r + g * s_d / n_d
        c = (1 - g) * c_r + g * c_d / n_d
        n = torch.sqrt(s * s + c * c + 1e-6)
        return torch.cat([s / n, c / n], dim=-1)

    def decode_angles(self, raw_out):
        return torch.atan2(raw_out[:, :self.dof], raw_out[:, self.dof:])
