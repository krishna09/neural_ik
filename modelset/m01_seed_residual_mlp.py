"""
M01 - Seed-residual pre-norm ResMLP.

Predicts a *correction* to the seed configuration instead of the joint angles
themselves: q = seed + delta. The delta head starts at exactly zero, so an
untrained network already outputs the seed and training only has to learn how
to remove the seed's error given the target pose.
"""

import torch
import torch.nn as nn

DESCRIPTION = "Seed-residual pre-norm ResMLP (6 blocks x 384): q = seed + delta, delta head zero-initialised."

POS_SCALE = 1.25  # metres -> roughly unit range (arm reach 0.7-1.3 m)


def quat_to_rot6d(q):
    """Quaternion (x, y, z, w) -> first two columns of the rotation matrix.
    Continuous and sign-invariant: q and -q give the same 6 numbers."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z, w = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w),
        2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w),
    ], dim=-1)


def encode(x, dof):
    """Raw input [xyz, quat, seed] -> (features, seed). Features use the 6D
    rotation and sin/cos of the seed so nothing jumps at +-180 / +-360 deg."""
    seed = x[:, 7:7 + dof]
    feats = torch.cat([x[:, :3] * POS_SCALE, quat_to_rot6d(x[:, 3:7]), torch.sin(seed), torch.cos(seed)], dim=-1)
    return feats, seed


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
    def __init__(self, dof, width=384, depth=6):
        super().__init__()
        self.dof = dof
        self.stem = nn.Linear(9 + 2 * dof, width)
        self.blocks = nn.Sequential(*[PreNormBlock(width) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = zero_linear(width, dof)

    def forward(self, x):
        feats, seed = encode(x, self.dof)
        delta = self.head(self.norm(self.blocks(self.stem(feats))))
        q = seed + delta
        return torch.cat([torch.sin(q), torch.cos(q)], dim=-1)

    def decode_angles(self, raw_out):
        return torch.atan2(raw_out[:, :self.dof], raw_out[:, self.dof:])
