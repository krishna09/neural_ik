"""
M05 - Learned iterative solver (unrolled, weight-shared refinement).

Classical IK solvers (e.g. damped least squares) start at a guess and apply
several small corrections. This network does the same with a learned update:
q0 = seed, then K times q_{k+1} = q_k + f(pose, q_k, seed, k), where f is one
shared ResMLP and k is a learned step embedding. Each step sees where the
previous step ended, so errors left by one step can be fixed by the next.
The whole unrolled chain is trained end to end by train_ik's loss.
"""

import torch
import torch.nn as nn

DESCRIPTION = "Learned iterative solver: 4 weight-shared refinement steps q <- q + f(pose, q, seed, step) (ResMLP 3 x 512)."

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
    def __init__(self, dof, width=512, depth=3, steps=4, pose_dim=256, step_dim=64):
        super().__init__()
        self.dof = dof
        self.steps = steps
        self.pose_enc = nn.Sequential(nn.Linear(9, pose_dim), nn.SiLU(), nn.Linear(pose_dim, pose_dim))
        self.step_emb = nn.Embedding(steps, step_dim)
        self.stem = nn.Linear(pose_dim + 4 * dof + step_dim, width)   # pose, sin/cos(q_k), sin/cos(seed), step
        self.blocks = nn.Sequential(*[PreNormBlock(width) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = zero_linear(width, dof)

    def forward(self, x):
        seed = x[:, 7:7 + self.dof]
        pose = self.pose_enc(torch.cat([x[:, :3] * POS_SCALE, quat_to_rot6d(x[:, 3:7])], dim=-1))
        seed_feat = torch.cat([torch.sin(seed), torch.cos(seed)], dim=-1)
        q = seed
        for k in range(self.steps):
            step = self.step_emb.weight[k].expand(x.shape[0], -1)
            h = self.stem(torch.cat([pose, torch.sin(q), torch.cos(q), seed_feat, step], dim=-1))
            q = q + self.head(self.norm(self.blocks(h)))
        return torch.cat([torch.sin(q), torch.cos(q)], dim=-1)

    def decode_angles(self, raw_out):
        return torch.atan2(raw_out[:, :self.dof], raw_out[:, self.dof:])
