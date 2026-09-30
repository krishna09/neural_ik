"""
M07 - Transformer over joint tokens.

Every joint becomes a token (its seed as sin/cos + a learned joint
embedding), plus one token for the target pose. A small pre-norm Transformer
encoder lets every joint attend to the pose and to all other joints, so
couplings like "the elbow compensates the shoulder" are learned explicitly.
Each joint token outputs a correction to its seed (zero-initialised).
"""

import torch
import torch.nn as nn

DESCRIPTION = "Transformer (4 layers, d=128, 4 heads) over 1 pose token + 1 token per joint; per-joint seed correction."

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


class IKNet(nn.Module):
    def __init__(self, dof, d_model=128, n_heads=4, n_layers=4, ff=512):
        super().__init__()
        self.dof = dof
        self.pose_tok = nn.Sequential(nn.Linear(9, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.joint_in = nn.Linear(2, d_model)
        self.joint_emb = nn.Parameter(torch.randn(dof, d_model) * 0.02)
        self.pose_emb = nn.Parameter(torch.randn(1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ff, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = zero_linear(d_model, 1)

    def forward(self, x):
        seed = x[:, 7:7 + self.dof]
        pose = torch.cat([x[:, :3] * POS_SCALE, quat_to_rot6d(x[:, 3:7])], dim=-1)
        p_tok = (self.pose_tok(pose) + self.pose_emb).unsqueeze(1)                       # (B, 1, d)
        j_tok = self.joint_in(torch.stack([torch.sin(seed), torch.cos(seed)], dim=-1)) + self.joint_emb  # (B, dof, d)
        h = self.encoder(torch.cat([p_tok, j_tok], dim=1))
        delta = self.head(self.norm(h[:, 1:])).squeeze(-1)                              # (B, dof)
        q = seed + delta
        return torch.cat([torch.sin(q), torch.cos(q)], dim=-1)

    def decode_angles(self, raw_out):
        return torch.atan2(raw_out[:, :self.dof], raw_out[:, self.dof:])
