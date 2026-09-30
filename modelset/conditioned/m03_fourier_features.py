"""
M03 - Fourier-feature pose encoding + seed-residual ResMLP.

Plain MLPs are biased towards smooth, low-frequency functions ("spectral
bias"), but millimetre-level IK needs fine detail. The target pose (position
+ 6D rotation) is first mapped through fixed random Fourier features
[sin(2*pi*Bv), cos(2*pi*Bv)] (Tancik et al., 2020), which lets the MLP fit
much sharper functions. B is a fixed buffer (saved with the model).
"""

import math

import torch
import torch.nn as nn

DESCRIPTION = "Random Fourier features of the pose (128 freq, sigma 2) + seed-residual ResMLP (5 x 512)."

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


class FourierFeatures(nn.Module):
    def __init__(self, in_dim, n_freq=128, sigma=2.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.randn(in_dim, n_freq, generator=g) * sigma)

    def forward(self, v):
        proj = 2 * math.pi * v @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


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
    def __init__(self, dof, width=512, depth=5, n_freq=128, sigma=2.0):
        super().__init__()
        self.dof = dof
        self.fourier = FourierFeatures(9, n_freq, sigma)
        self.stem = nn.Linear(9 + 2 * n_freq + 2 * dof, width)
        self.blocks = nn.Sequential(*[PreNormBlock(width) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = zero_linear(width, dof)

    def forward(self, x):
        seed = x[:, 7:7 + self.dof]
        pose = torch.cat([x[:, :3] * POS_SCALE, quat_to_rot6d(x[:, 3:7])], dim=-1)
        feats = torch.cat([pose, self.fourier(pose), torch.sin(seed), torch.cos(seed)], dim=-1)
        delta = self.head(self.norm(self.blocks(self.stem(feats))))
        q = seed + delta
        return torch.cat([torch.sin(q), torch.cos(q)], dim=-1)

    def decode_angles(self, raw_out):
        return torch.atan2(raw_out[:, :self.dof], raw_out[:, self.dof:])
