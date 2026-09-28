"""
M02 - Direct wide MLP with continuous input features (no seed residual).

The control experiment for the seed-residual models: same continuous inputs
(6D rotation, sin/cos seed), but the network predicts the joint angles
directly as unit-circle [sin, cos] pairs. Less dependent on a good seed, so it
degrades more gracefully when the seed is far from the answer.
"""

import torch
import torch.nn as nn

DESCRIPTION = "Direct wide MLP (4 x 1024, LayerNorm + SiLU), continuous inputs, unit-circle output - no seed residual."

POS_SCALE = 1.25


def quat_to_rot6d(q):
    """Quaternion (x, y, z, w) -> first two rotation-matrix columns (continuous, sign-invariant)."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z, w = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w),
        2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w),
    ], dim=-1)


def encode(x, dof):
    seed = x[:, 7:7 + dof]
    return torch.cat([x[:, :3] * POS_SCALE, quat_to_rot6d(x[:, 3:7]), torch.sin(seed), torch.cos(seed)], dim=-1)


class IKNet(nn.Module):
    def __init__(self, dof, width=1024, depth=4):
        super().__init__()
        self.dof = dof
        layers = [nn.Linear(9 + 2 * dof, width)]
        for _ in range(depth - 1):
            layers += [nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, width)]
        layers += [nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, 2 * dof)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        raw = self.net(encode(x, self.dof))
        s, c = raw[:, :self.dof], raw[:, self.dof:]
        norm = torch.sqrt(s * s + c * c + 1e-6)          # project each joint onto the unit circle
        return torch.cat([s / norm, c / norm], dim=-1)

    def decode_angles(self, raw_out):
        return torch.atan2(raw_out[:, :self.dof], raw_out[:, self.dof:])
