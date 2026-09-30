"""
M08 - Mixture of experts.

IK is piecewise: different regions of the workspace (and elbow-up / elbow-down
branches) need different local corrections. A shared encoder feeds a softmax
gate and 8 small expert networks; the output correction is the gate-weighted
sum of the experts' corrections. Experts are zero-initialised, the gate starts
uniform, so training begins from "output the seed".
"""

import torch
import torch.nn as nn

DESCRIPTION = "Mixture of 8 experts (512 -> 256 -> dof each) with a softmax gate on a shared ResMLP encoder; seed-residual."

POS_SCALE = 1.25


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


class PreNormBlock(nn.Module):
    def __init__(self, width, mult=2):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width * mult)
        self.fc2 = nn.Linear(width * mult, width)
        self.act = nn.SiLU()

    def forward(self, h):
        return h + self.fc2(self.act(self.fc1(self.norm(h))))


class IKNet(nn.Module):
    def __init__(self, dof, width=512, depth=2, n_experts=8, expert_width=256):
        super().__init__()
        self.dof = dof
        self.stem = nn.Linear(9 + 2 * dof, width)
        self.encoder = nn.Sequential(*[PreNormBlock(width) for _ in range(depth)], nn.LayerNorm(width))
        self.gate = zero_linear(width, n_experts)          # uniform gate at start
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(width, expert_width), nn.SiLU(),
                          nn.Linear(expert_width, expert_width), nn.SiLU(),
                          zero_linear(expert_width, dof))
            for _ in range(n_experts)
        ])

    def forward(self, x):
        seed = x[:, 7:7 + self.dof]
        feats = torch.cat([x[:, :3] * POS_SCALE, quat_to_rot6d(x[:, 3:7]), torch.sin(seed), torch.cos(seed)], dim=-1)
        h = self.encoder(self.stem(feats))
        weights = torch.softmax(self.gate(h), dim=-1)                       # (B, E)
        deltas = torch.stack([e(h) for e in self.experts], dim=1)           # (B, E, dof)
        q = seed + (weights.unsqueeze(-1) * deltas).sum(dim=1)
        return torch.cat([torch.sin(q), torch.cos(q)], dim=-1)

    def decode_angles(self, raw_out):
        return torch.atan2(raw_out[:, :self.dof], raw_out[:, self.dof:])
