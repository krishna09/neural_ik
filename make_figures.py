"""Generates the two figures used in the write-up: training curves and the
three-way accuracy/speed/safety comparison. Run after train_ik.py."""

import json

import matplotlib.pyplot as plt
import numpy as np

from evaluate import run_eval


def plot_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, robot in zip(axes, ["xarm6", "kuka_iiwa"]):
        with open(f"models/{robot}_history.json") as f:
            hist = json.load(f)["val_mse"]
        ax.plot(hist, color="#2563eb")
        ax.set_title(f"{robot} - IK net validation MSE")
        ax.set_xlabel("epoch")
        ax.set_ylabel("val MSE (sin/cos targets)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/training_curves.png", dpi=150)
    print("saved figures/training_curves.png")


def plot_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    metrics = [("pos_err", "Position error (mm)", 1000), ("orn_err", "Orientation error (deg)", 1), ("time", "Inference time (ms)", 1000)]
    labels = ["Neural alone", "Classical DLS", "Hybrid"]
    colors = ["#f97316", "#64748b", "#16a34a"]

    for robot_idx, robot in enumerate(["xarm6"]):
        neural, classical, hybrid = run_eval(robot, n_targets=200, seed=1)
        results = [neural, classical, hybrid]
        for ax, (key, title, scale) in zip(axes, metrics):
            values = [np.mean(r[key]) * scale for r in results]
            ax.bar(labels, values, color=colors)
            ax.set_title(f"{robot}: {title}")
            ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig("figures/comparison.png", dpi=150)
    print("saved figures/comparison.png")


if __name__ == "__main__":
    plot_training_curves()
    plot_comparison()
