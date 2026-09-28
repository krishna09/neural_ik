import torch
import torch.nn as nn
class IKNet(nn.Module):
    """Pose (+ seed config) -> joint angles, as [sin, cos] pairs."""

    def __init__(self, dof, hidden=256*2):
        super().__init__()
        in_dim = 3 + 4 + dof  # position + quaternion + seed
        out_dim = 2 * dof
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, out_dim),
        )
        self.dof = dof

    def forward(self, x):
        return self.net(x)

    def decode_angles(self, raw_out):
        """[sin,cos] pairs -> joint angles in radians via atan2."""
        sin_part = raw_out[:, : self.dof]
        cos_part = raw_out[:, self.dof:]
        return torch.atan2(sin_part, cos_part)