"""
Level 1 - Step 2: Neural inverse kinematics model.

Learns target_pose (position + quaternion) -> joint_angles for a given
robot, using the dataset produced by generate_dataset.py.

Design choices (see the write-up for the reasoning):

  - Output is [sin(theta), cos(theta)] per joint instead of theta directly,
    which avoids the angle-wraparound discontinuity (e.g. -pi and +pi being
    the same pose but numerically far apart) that would otherwise confuse
    an MSE loss.

  - For the 7DOF case (redundant arm: infinitely many joint solutions map
    to the same end-effector pose) the network is additionally conditioned
    on a "seed" configuration - typically the arm's *previous* joint state
    in a trajectory - so it learns "the solution near this seed" instead of
    an ill-defined average over all valid solutions. For the 6DOF case
    (usually not redundant for a general 6D pose) this conditioning is
    optional but harmless.

  - A collision-aware auxiliary loss term penalises predicted joint angles
    that a separately trained collision classifier flags as self-colliding,
    which is what ties this back to the project's "self-collision
    prevention" requirement rather than treating IK and collision-checking
    as two unrelated steps.
"""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
# from model_set import IKNet

# --------------------------------------------------------------------------
# Data loading / preprocessing
# --------------------------------------------------------------------------

def load_dataset(robot_name, seed_noise_std=0.3, rng=None):
    """
    Loads the .npz dataset and builds model inputs/targets.

    Input features: [position(3), quaternion(4), seed_joint_angles(dof)]
    The "seed" is a noisy version of the true joint solution, standing in
    for "the arm's previous configuration" that a real trajectory-following
    controller would supply. This is what lets the network pick a
    consistent branch of the redundant solution set instead of averaging.

    Target: joint angles, encoded as [sin, cos] pairs -> 2*dof outputs.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    data = np.load(f"data/{robot_name}_ik_dataset.npz")
    joints = data["joints"]              # (N, dof)
    positions = data["positions"]        # (N, 3)
    orientations = data["orientations"]  # (N, 4)
    collision_labels = data["collision_labels"]  # (N,)
    dof = int(data["dof"])

    # Only train the IK regressor on collision-free examples - we don't
    # want it to learn to reproduce colliding solutions. Colliding examples
    # are still used below to train the auxiliary collision classifier.
    free_mask = collision_labels == 0
    joints_free = joints[free_mask]
    positions_free = positions[free_mask]
    orientations_free = orientations[free_mask]

    seed = joints_free + rng.normal(0, seed_noise_std, size=joints_free.shape).astype(np.float32)

    X = np.concatenate([positions_free, orientations_free, seed], axis=1).astype(np.float32)
    y = np.concatenate([np.sin(joints_free), np.cos(joints_free)], axis=1).astype(np.float32)

    return X, y, dof, data


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
#dont chnage this or remove this
# class IKNet(nn.Module):
#     """Pose (+ seed config) -> joint angles, as [sin, cos] pairs."""

#     def __init__(self, dof, hidden=256):
#         super().__init__()
#         in_dim = 3 + 4 + dof  # position + quaternion + seed
#         out_dim = 2 * dof
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, hidden),
#             nn.ReLU(),
#             nn.Linear(hidden, hidden),
#             nn.ReLU(),
#             nn.Linear(hidden, hidden // 2),
#             nn.ReLU(),
#             nn.Linear(hidden // 2, out_dim),
#         )
#         self.dof = dof

#     def forward(self, x):
#         return self.net(x)

#     def decode_angles(self, raw_out):
#         """[sin,cos] pairs -> joint angles in radians via atan2."""
#         sin_part = raw_out[:, : self.dof]
#         cos_part = raw_out[:, self.dof:]
#         return torch.atan2(sin_part, cos_part)

class IKNet(nn.Module):
    """Pose (+ seed config) -> joint angles, as [sin, cos] pairs."""

    def __init__(self, dof, hidden=256):
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


class CollisionNet(nn.Module):
    """Joint configuration -> P(self-collision). Small, fast classifier
    used both standalone (quick collision pre-check) and as an auxiliary
    loss term during IKNet training."""

    def __init__(self, dof, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dof, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, q):
        return self.net(q).squeeze(-1)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train_collision_net(data, dof, epochs=15, device="cpu"):
    joints = torch.tensor(data["joints"], dtype=torch.float32)
    labels = torch.tensor(data["collision_labels"], dtype=torch.float32)

    ds = TensorDataset(joints, labels)
    n_val = max(1, int(0.15 * len(ds)))
    train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)

    model = CollisionNet(dof).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        for qb, yb in train_loader:
            qb, yb = qb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(qb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for qb, yb in val_loader:
                qb, yb = qb.to(device), yb.to(device)
                pred = (torch.sigmoid(model(qb)) > 0.5).float()
                correct += (pred == yb).sum().item()
                total += yb.numel()
        if epoch == epochs - 1 or epoch % 5 == 0:
            print(f"  [collision-net] epoch {epoch+1}/{epochs}  val_acc={correct/total:.3f}")

    return model


def train_ik_net(X, y, dof, collision_net, epochs=60, device="cpu",
                  collision_weight=0.05):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    ds = TensorDataset(X_t, y_t)
    n_val = max(1, int(0.15 * len(ds)))
    train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)

    model = IKNet(dof).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    collision_net.eval()
    for p in collision_net.parameters():
        p.requires_grad_(False)

    history = []
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            raw_out = model(xb)
            fit_loss = mse(raw_out, yb)

            angles = model.decode_angles(raw_out)
            collision_logits = collision_net(angles)
            # Penalise configurations the collision-net thinks are risky.
            # sigmoid -> soft probability, differentiable w.r.t. the IK net's
            # own output, so this genuinely shapes training rather than
            # just filtering after the fact.
            collision_penalty = torch.sigmoid(collision_logits).mean()

            loss = fit_loss + collision_weight * collision_penalty
            loss.backward()
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                raw_out = model(xb)
                val_losses.append(mse(raw_out, yb).item())
        val_loss = float(np.mean(val_losses))
        history.append(val_loss)
        if epoch == epochs - 1 or epoch % 10 == 0:
            print(f"  [ik-net] epoch {epoch+1}/{epochs}  val_mse={val_loss:.5f}")

    return model, history


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main(robot_name, epochs, collision_epochs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    X, y, dof, raw_data = load_dataset(robot_name)
    print(f"[{robot_name}] dof={dof}  training examples={len(X)}")

    print("training collision classifier...")
    collision_net = train_collision_net(raw_data, dof, epochs=collision_epochs, device=device)

    print("training IK network (with collision-aware loss)...")
    ik_net, history = train_ik_net(X, y, dof, collision_net, epochs=epochs, device=device)

    torch.save(ik_net.state_dict(), f"models/{robot_name}_ik_net.pt")
    torch.save(collision_net.state_dict(), f"models/{robot_name}_collision_net.pt")
    with open(f"models/{robot_name}_history.json", "w") as f:
        json.dump({"val_mse": history}, f)

    print(f"saved models/{robot_name}_ik_net.pt and models/{robot_name}_collision_net.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="xarm6", choices=["xarm6", "kuka_iiwa", "panda"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--collision-epochs", type=int, default=15)
    args = parser.parse_args()
    main(args.robot, args.epochs, args.collision_epochs)
