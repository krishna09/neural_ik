"""
M09 - DenseNet-style MLP with input re-injection.

Every layer receives the concatenation of the input features and the outputs
of ALL previous layers (Huang et al., 2017, applied to an MLP). The raw pose
and seed stay directly visible at every depth, which helps precise regression
(the same idea as NeRF's skip connection), and gradients reach early layers
easily. Output: a correction to the seed (zero-initialised).
"""

import torch
import torch.nn as nn

DESCRIPTION = "Dense MLP: 8 layers x 128 growth, each layer sees the input + all earlier layers; seed-residual."

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


class DenseLayer(nn.Module):
    def __init__(self, n_in, growth):
        super().__init__()
        self.norm = nn.LayerNorm(n_in)
        self.fc = nn.Linear(n_in, growth)
        self.act = nn.SiLU()

    def forward(self, h):
        return self.act(self.fc(self.norm(h)))


class IKNet(nn.Module):
    def __init__(self, dof, n_layers=8, growth=128, head_width=256):
        super().__init__()
        self.dof = dof
        in_dim = 9 + 2 * dof
        self.layers = nn.ModuleList([DenseLayer(in_dim + i * growth, growth) for i in range(n_layers)])
        total = in_dim + n_layers * growth
        self.head = nn.Sequential(nn.LayerNorm(total), nn.Linear(total, head_width), nn.SiLU(),
                                  zero_linear(head_width, dof))

    def forward(self, x):
        seed = x[:, 7:7 + self.dof]
        h = torch.cat([x[:, :3] * POS_SCALE, quat_to_rot6d(x[:, 3:7]), torch.sin(seed), torch.cos(seed)], dim=-1)
        for layer in self.layers:
            h = torch.cat([h, layer(h)], dim=-1)
        q = seed + self.head(h)
        return torch.cat([torch.sin(q), torch.cos(q)], dim=-1)

    def decode_angles(self, raw_out):
        return torch.atan2(raw_out[:, :self.dof], raw_out[:, self.dof:])
