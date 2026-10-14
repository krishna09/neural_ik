#!/usr/bin/env python
"""Result figures from the measured data only (deliverables/results/*.csv|json, recorded run meta.json).

    python deliverables/scripts/pipeline/make_charts.py
Output: deliverables/results/accuracy_comparison.png, training_validation.png (1920 x 1080, dark theme
matching the deck and video; series colours = validated categorical slots).
"""
import csv
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BEST_RUN_DIR, RESULTS, read_json  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

BG, SURF, BORDER, GRID = "#0a1527", "#0f1b2d", "#23344f", "#1b2a40"
T1, T2, MUTED = "#eef3fa", "#aebbd0", "#74849e"
BLUE, ORANGE, AQUA, YELLOW, ACCENT = "#3987e5", "#d95926", "#199e70", "#c98500", "#4fd6f2"
METHODS = [("residual_mlp", "Network alone", AQUA), ("classical_dls", "Classical IK (DLS)", ORANGE),
           ("hybrid_lm", "Network + LM", BLUE)]
MODES = [("warm", "warm seed"), ("zero", "home pose"), ("random", "random pose")]
plt.rcParams.update({"font.family": "Inter", "font.size": 15, "text.color": T1, "axes.labelcolor": T2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": BORDER, "figure.facecolor": BG,
                     "axes.facecolor": SURF, "savefig.facecolor": BG})


def fx(v, d=1):
    """Decimal rounding, half away from zero (2.55 -> "2.6"): the rule the deck and the renderer use."""
    return str(Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=19, fontweight=600, color=T1, pad=14)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#3a4d6b")
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def grouped(ax, rows, key, title, unit, fmt, ymax=None):
    import numpy as np
    x = np.arange(len(MODES))
    w = 0.26
    for j, (m, name, col) in enumerate(METHODS):
        vals = [float(rows[f"{m}|{mode}"][key]) for mode, _ in MODES]
        bars = ax.bar(x + (j - 1) * w, [max(v, 0.0) for v in vals], w - 0.03, color=col, label=name, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + (ymax or max(vals) or 1) * 0.015, fmt(v),
                    ha="center", va="bottom", fontsize=13, color=T1, fontweight=600)
    ax.set_xticks(x, [lbl for _, lbl in MODES], color=T2, fontsize=15)
    if ymax:
        ax.set_ylim(0, ymax)
    ax.set_ylabel(unit, fontsize=14)
    style(ax, title)


def accuracy_figure():
    rows = {f"{r['method']}|{r['seed_mode']}": r for r in csv.DictReader(open(RESULTS / "evaluation_metrics.csv"))}
    fig, axes = plt.subplots(2, 2, figsize=(19.2, 10.8), dpi=100)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.79, bottom=0.08, hspace=0.42, wspace=0.14)
    pct = lambda v: f"{fx(v, 0)}%" if v >= 10 or v == 0 else f"{fx(v, 1)}%"
    grouped(axes[0, 0], rows, "success_1mm_1deg_pct", "Valid answers within 1 mm and 1°", "% of 2,000 targets", pct, 115)
    grouped(axes[0, 1], rows, "self_collision_pct", "Self-collision", "% of answers", pct, 40)
    grouped(axes[1, 0], rows, "limit_violation_pct", "Joint-limit violation", "% of answers", pct, 55)
    grouped(axes[1, 1], rows, "time_ms_mean", "Time per target (one CPU thread)", "milliseconds", lambda v: fx(v, 1) if v >= 1 else fx(v, 2), 60)
    fig.text(0.06, 0.95, "Neural IK vs classical IK vs hybrid — xArm6, 2,000 identical held-out targets", fontsize=28, fontweight=700, color=T1)
    fig.text(0.06, 0.905, "Seed = the joint configuration each method starts from: warm = true joints + N(0, 0.3 rad); "
             "home = all joints 0°; random = uniform in the limits.", fontsize=16, color=T2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in METHODS]
    fig.legend(handles, [n for _, n, _ in METHODS], loc="upper left", bbox_to_anchor=(0.055, 0.885), ncol=3, frameon=False,
               fontsize=16, labelcolor=T1, handlelength=1.2, columnspacing=2.0)
    fig.text(0.06, 0.02, "Source: deliverables/results/evaluation_metrics.csv (scripts/pipeline/evaluate_methods.py). "
             "Every answer re-checked in PyBullet: FK pose error, joint limits, self-collision.", fontsize=13, color=MUTED)
    fig.savefig(RESULTS / "accuracy_comparison.png")
    plt.close(fig)


def training_figure():
    hist = list(csv.DictReader(open(RESULTS / "training_history.csv")))
    ik = [h for h in hist if h["phase"] == "ik"]
    col = [h for h in hist if h["phase"] == "collision"]
    meta = read_json(BEST_RUN_DIR / "meta.json")
    best = [e for e in meta["epochs"] if e["phase"] == "ik"]
    bcol = [e for e in meta["epochs"] if e["phase"] == "collision"]
    fig, axes = plt.subplots(1, 3, figsize=(19.2, 10.8), dpi=100)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.78, bottom=0.14, wspace=0.22)
    ep = [int(float(h["epoch"])) for h in ik]
    ax = axes[0]
    ax.plot(ep, [float(h["train_fit_mse"]) for h in ik], color="#c8d3e3", lw=2.5, label="train (sin/cos MSE)")
    ax.plot(ep, [float(h["val_mse"]) for h in ik], color=ACCENT, lw=2.5, label="validation")
    ax.set_yscale("log")
    ax.set_yticks([0.03, 0.05, 0.1, 0.2, 0.3], ["0.03", "0.05", "0.1", "0.2", "0.3"])
    ax.minorticks_off()
    ax.set_xlabel("epoch")
    ax.legend(frameon=False, labelcolor=T1, fontsize=14)
    style(ax, "Baseline network: loss per epoch")
    ax = axes[1]
    ax.plot(ep, [float(h["val_pos_mean_mm"]) for h in ik], color=YELLOW, lw=2.5, label="baseline MLP · 20 k samples")
    ax.plot([e["epoch"] for e in best], [e["acc"]["val"]["pos_mean_mm"] for e in best], color=AQUA, lw=2.5,
            label="residual MLP · 200 k samples")
    ax.set_yscale("log")
    ax.set_yticks([20, 50, 100, 200, 400], ["20", "50", "100", "200", "400"])
    ax.minorticks_off()
    ax.set_xlabel("epoch")
    ax.set_ylabel("mm")
    ax.annotate(f"{fx(ik[-1]['val_pos_mean_mm'], 0)} mm", (ep[-1], float(ik[-1]["val_pos_mean_mm"])), xytext=(8, 8),
                textcoords="offset points", color=T1, fontsize=14, fontweight=600)
    ax.annotate(f"{fx(best[-1]['acc']['val']['pos_mean_mm'], 1)} mm", (best[-1]["epoch"], best[-1]["acc"]["val"]["pos_mean_mm"]),
                xytext=(-60, -24), textcoords="offset points", color=T1, fontsize=14, fontweight=600)
    ax.legend(frameon=False, labelcolor=T1, fontsize=14)
    style(ax, "Mean validation position error (FK-checked)")
    ax = axes[2]
    ax.plot([int(float(h["epoch"])) for h in col], [float(h["val_acc_pct"]) for h in col], color=YELLOW, lw=2.5, marker="o",
            ms=7, label="baseline reproduction")
    ax.plot([e["epoch"] for e in bcol], [e["val_acc"] for e in bcol], color=AQUA, lw=2.5, marker="o", ms=7, label="best run")
    ax.set_xlabel("epoch")
    ax.set_ylabel("% correct")
    ax.legend(frameon=False, labelcolor=T1, fontsize=14, loc="lower right")
    style(ax, "Collision classifier: validation accuracy")
    fig.text(0.05, 0.93, "Training and validation — recorded histories only", fontsize=28, fontweight=700, color=T1)
    fig.text(0.05, 0.885, "Adam, learning rate 0.001, batch 256, random 85 / 15 train / validation split; "
             "loss = MSE(sin, cos) + 0.05 × collision penalty.", fontsize=16, color=T2)
    fig.text(0.05, 0.04, "Baseline: deliverables/results/training_history.csv (instrumented train_ik.py loop, CPU). "
             f"Residual MLP: team run {meta['id']} ({meta['gpu']}, {meta['train_secs']:.0f} s), meta.json epochs.",
             fontsize=13, color=MUTED)
    fig.savefig(RESULTS / "training_validation.png")
    plt.close(fig)


if __name__ == "__main__":
    accuracy_figure()
    training_figure()
    print("wrote", RESULTS / "accuracy_comparison.png", RESULTS / "training_validation.png")
