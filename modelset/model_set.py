import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(
        self,
        dim,
        dropout=0.05,
        activation="silu"
    ):
        super().__init__()

        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)

        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

        if activation == "silu":
            self.act = nn.SiLU()

        elif activation == "gelu":
            self.act = nn.GELU()

        elif activation == "leakyrelu":
            self.act = nn.LeakyReLU(0.1)

        else:
            self.act = nn.ReLU()

    def forward(self, x):
        residual = x

        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.norm2(x)

        x = x + residual
        x = self.act(x)

        return x


class IKNet(nn.Module):
    """
    Pose + seed joint configuration
            ↓
    [sin(theta), cos(theta)] for each joint
    """

    def __init__(self, dof=7):
        super().__init__()

        self.dof = dof

        # XYZ + quaternion + seed joints
        in_dim = 3 + 4 + dof

        # sin + cos for every joint
        out_dim = 2 * dof

        # --------------------------------------------------
        # INPUT FEATURE EXTRACTION
        # --------------------------------------------------

        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU()
        )

        # --------------------------------------------------
        # EXPANSION
        # --------------------------------------------------

        self.expand = nn.Sequential(
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.05)
        )

        # --------------------------------------------------
        # DEEP FEATURE LEARNING
        # --------------------------------------------------

        self.res1 = ResidualBlock(
            512,
            dropout=0.05,
            activation="silu"
        )

        self.res2 = ResidualBlock(
            512,
            dropout=0.05,
            activation="gelu"
        )

        self.res3 = ResidualBlock(
            512,
            dropout=0.05,
            activation="leakyrelu"
        )

        # --------------------------------------------------
        # FEATURE COMPRESSION
        # --------------------------------------------------

        self.compress1 = nn.Sequential(
            nn.Linear(512, 384),
            nn.LayerNorm(384),
            nn.SiLU(),
            nn.Dropout(0.05)
        )

        self.compress2 = nn.Sequential(
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.03)
        )

        self.compress3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU()
        )

        # --------------------------------------------------
        # OUTPUT
        # --------------------------------------------------

        self.output_layer = nn.Linear(
            128,
            out_dim
        )

    def forward(self, x):

        x = self.input_layer(x)

        x = self.expand(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        x = self.compress1(x)
        x = self.compress2(x)
        x = self.compress3(x)

        x = self.output_layer(x)

        return x

    def decode_angles(self, raw_out):

        sin_part = raw_out[:, :self.dof]
        cos_part = raw_out[:, self.dof:]

        # Normalize sin/cos pairs
        magnitude = torch.sqrt(
            sin_part ** 2 +
            cos_part ** 2 +
            1e-8
        )

        sin_part = sin_part / magnitude
        cos_part = cos_part / magnitude

        angles = torch.atan2(
            sin_part,
            cos_part
        )

        return angles