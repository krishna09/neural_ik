"""
M04 - FiLM: the target pose modulates a network that corrects the seed.

Two streams: a pose encoder (position + 6D rotation) and a trunk that works
on the seed configuration. Every trunk block is conditioned on the pose with
Feature-wise Linear Modulation (Perez et al., 2018): h -> LN(h) * (1 + gamma)
+ beta, where gamma / beta come from the pose. The question "how should this
seed be corrected?" is thereby asked explicitly *for this target pose*.
FiLM layers start at identity (zero-initialised), the delta head at zero.
"""

import torch
import torch.nn as nn

DESCRIPTION = "FiLM: pose encoder modulates every block of a seed-correcting trunk (6 x 384), seed-residual output."

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


class FiLMBlock(nn.Module):
    def __init__(self, width, cond_dim, mult=2):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.film = zero_linear(cond_dim, 2 * width)      # -> gamma, beta (identity at start)
        self.fc1 = nn.Linear(width, width * mult)
        self.fc2 = nn.Linear(width * mult, width)
        self.act = nn.SiLU()

    def forward(self, h, cond):
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        z = self.norm(h) * (1 + gamma) + beta
        return h + self.fc2(self.act(self.fc1(z)))


class IKNet(nn.Module):
    def __init__(self, dof, width=384, depth=6, cond_dim=256):
        super().__init__()
        self.dof = dof
        self.pose_enc = nn.Sequential(
            nn.Linear(9, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.stem = nn.Linear(9 + 2 * dof, width)
        self.blocks = nn.ModuleList([FiLMBlock(width, cond_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = zero_linear(width, dof)

    def forward(self, x):
        seed = x[:, 7:7 + self.dof]
        pose = torch.cat([x[:, :3] * POS_SCALE, quat_to_rot6d(x[:, 3:7])], dim=-1)
        cond = self.pose_enc(pose)
        h = self.stem(torch.cat([pose, torch.sin(seed), torch.cos(seed)], dim=-1))
        for block in self.blocks:
            h = block(h, cond)
        q = seed + self.head(self.norm(h))
        return torch.cat([torch.sin(q), torch.cos(q)], dim=-1)

    def decode_angles(self, raw_out):
        return torch.atan2(raw_out[:, :self.dof], raw_out[:, self.dof:])
