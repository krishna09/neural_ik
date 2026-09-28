"""
M10 - SwiGLU residual network with a learned "trust the seed" gate.

Deep pre-norm residual blocks with SwiGLU activations (Shazeer, 2020 - the
MLP block of modern transformers). Two answers are produced per joint:
  (a) seed + correction          - accurate when the seed is close
  (b) a direct prediction        - still usable when the seed is poor
and a per-joint gate g in (0, 1) blends their unit vectors:
  v = (1 - g) * unit(seed + delta) + g * unit(direct),  then renormalised.
The gate starts near 0 (bias -3 -> g ~ 0.05), i.e. mostly trusting the seed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

DESCRIPTION = "SwiGLU ResNet (8 x 384) with two heads (seed + delta, direct angle) blended by a learned per-joint gate."

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
        self.stem = nn.Linear(9 + 2 * dof, width)
        self.blocks = nn.Sequential(*[SwiGLUBlock(width, hidden) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.delta_head = zero_linear(width, dof)
        self.direct_head = nn.Linear(width, 2 * dof)
        self.gate_head = nn.Linear(width, dof)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -3.0)

    def forward(self, x):
        seed = x[:, 7:7 + self.dof]
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
