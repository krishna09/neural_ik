"""
M06 - Kinematic-chain (autoregressive) joint heads.

A serial arm is a chain: where the wrist has to be depends on what the
shoulder and elbow already did. A shared trunk encodes pose + seed into a
context vector; then joint i is predicted by its own small head that sees the
context AND the already-predicted joints 1..i-1 (as sin/cos). Each head
outputs a correction to seed_i (zero-initialised).
"""

import torch
import torch.nn as nn

DESCRIPTION = "Kinematic-chain heads: trunk (4 x 512) + one head per joint conditioned on the joints predicted before it."

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
    def __init__(self, dof, width=512, depth=4, head_width=256):
        super().__init__()
        self.dof = dof
        self.stem = nn.Linear(9 + 2 * dof, width)
        self.trunk = nn.Sequential(*[PreNormBlock(width) for _ in range(depth)], nn.LayerNorm(width))
        # head i input: context + sin/cos of joints 1..i-1 (predicted) + sin/cos of seed_i
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(width + 2 * i + 2, head_width), nn.SiLU(),
                nn.Linear(head_width, head_width), nn.SiLU(),
                zero_linear(head_width, 1),
            )
            for i in range(dof)
        ])

    def forward(self, x):
        seed = x[:, 7:7 + self.dof]
        feats = torch.cat([x[:, :3] * POS_SCALE, quat_to_rot6d(x[:, 3:7]), torch.sin(seed), torch.cos(seed)], dim=-1)
        ctx = self.trunk(self.stem(feats))
        prev = []                                           # sin/cos of joints already predicted
        qs = []
        for i, head in enumerate(self.heads):
            s_i = seed[:, i:i + 1]
            inp = torch.cat([ctx, *prev, torch.sin(s_i), torch.cos(s_i)], dim=-1)
            q_i = s_i + head(inp)
            qs.append(q_i)
            prev += [torch.sin(q_i), torch.cos(q_i)]
        q = torch.cat(qs, dim=-1)
        return torch.cat([torch.sin(q), torch.cos(q)], dim=-1)

    def decode_angles(self, raw_out):
        return torch.atan2(raw_out[:, :self.dof], raw_out[:, self.dof:])
