#!/usr/bin/env python
"""Joint-axis diagram of the STP_Spacearm 7-DOF arm (teammate project), drawn from its configuration only.

Left: a 2-D textbook kinematic schematic (zero configuration, side view) with the standard joint symbols.
Right: a zoomed 3-D view of the arm alone in a bent pose, every joint axis drawn as a long line through the
joint with its rotation arrow. The joint chain is the one in deliverables/scripts/docs/spacearm_kinematics_diagram.py
(re-implemented from configs/default.yaml and kinematics.py of akashrayganguly/STP_Spacearm; nothing of that
project is run).

    python deliverables/scripts/docs/spacearm_joint_axes.py
Output: deliverables/documentation/STP_Spacearm_joint_axes.png (+ .pdf)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patches  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spacearm_kinematics_diagram import (AXES, LINKS, LOWER, MOUNT, PEDESTAL, RADII, TOOL, UPPER,  # noqa: E402
                                         cylinder, fk)

C_Z, C_Y, C_LINK = "#d95926", "#2f6fd1", "#8e9aab"
QDEG = [-37, 25, 15, -88, -55, 34, 0]        # chosen so all joints sit well apart on screen
OUT = Path(__file__).resolve().parents[2] / "documentation" / "STP_Spacearm_joint_axes"


# ------------------------------------------------------------------------------------- 2-D schematic
def schematic(ax):
    ax.set_title("A · Kinematic schematic (zero configuration, side view)", fontsize=13, fontweight="bold", loc="left")
    ax.set_aspect("equal"); ax.axis("off")
    S = 10.0                                            # drawing units per metre
    z = 0.0
    # spacecraft top face (ground symbol)
    ax.add_patch(patches.Rectangle((-1.6, -0.45), 3.2, 0.45, facecolor="#ead8a8", edgecolor="#8a7440", hatch="///", lw=1))
    ax.text(1.75, -0.25, "spacecraft top face\n(mount at x = 0.30 m)", fontsize=8.5, va="center", color="#6b5a2e")
    # pedestal
    ax.add_patch(patches.Rectangle((-0.35, 0), 0.7, PEDESTAL["length"] * S, facecolor="#9aa3ae", edgecolor="#555", lw=1))
    ax.text(-0.5, PEDESTAL["length"] * S / 2, "pedestal 10 cm", fontsize=8, ha="right", va="center", color="#444")
    z = PEDESTAL["length"] * S
    lengths = LINKS[:-1] + [None]
    for i, ax_name in enumerate(AXES):
        zc = z
        if ax_name == "z":                              # twist joint: axis along the link (in the page, vertical)
            ax.add_patch(patches.FancyBboxPatch((-0.45, zc - 0.22), 0.9, 0.44, boxstyle="round,pad=0.02",
                                                facecolor="white", edgecolor=C_Z, lw=2.2, zorder=4))
            ax.plot([0, 0], [zc - 0.75, zc + 0.75], color=C_Z, lw=1.4, ls=(0, (5, 3)), zorder=5)
            arc = patches.FancyArrowPatch((0.62, zc - 0.08), (-0.62, zc - 0.08), connectionstyle="arc3,rad=-0.45",
                                          arrowstyle="-|>", mutation_scale=12, color=C_Z, lw=1.6, zorder=6)
            ax.add_patch(arc)
            txt = f"J{i + 1}  twist · axis z (along the link)"
        else:                                           # bend joint: axis perpendicular to the page (y)
            ax.add_patch(patches.Circle((0, zc), 0.34, facecolor="white", edgecolor=C_Y, lw=2.2, zorder=4))
            ax.add_patch(patches.Circle((0, zc), 0.07, color=C_Y, zorder=5))
            arc = patches.FancyArrowPatch((0.48, zc + 0.3), (0.48, zc - 0.3), connectionstyle="arc3,rad=-0.9",
                                          arrowstyle="-|>", mutation_scale=12, color=C_Y, lw=1.6, zorder=6)
            ax.add_patch(arc)
            txt = f"J{i + 1}  bend · axis y (out of the page)"
        ax.text(1.25, zc, txt, fontsize=9.5, va="center", color=C_Z if ax_name == "z" else C_Y, fontweight="bold")
        L = (lengths[i] if lengths[i] is not None else TOOL) * S
        z_next = zc + L
        ax.plot([0, 0], [zc + (0.22 if ax_name == "z" else 0.34), z_next - 0.36], color="#3c4656", lw=4,
                solid_capstyle="butt", zorder=3)
        name = f"L{i + 1} = {round(100 * lengths[i])} cm" if lengths[i] is not None else "tool = 15 cm"
        ax.annotate("", (-1.15, zc), (-1.15, z_next), arrowprops=dict(arrowstyle="<->", color="#777", lw=0.9))
        ax.text(-1.3, (zc + z_next) / 2, name, fontsize=8.5, ha="right", va="center", color="#333")
        z = z_next
    # TCP
    ax.plot(0, z, marker="o", ms=9, color="#111", zorder=6)
    ax.text(0.3, z + 0.1, "TCP (tool tip)", fontsize=10, fontweight="bold")
    # frame legend
    ox, oz = -3.4, 0.6
    ax.annotate("", (ox + 0.9, oz), (ox, oz), arrowprops=dict(arrowstyle="-|>", color="#d03b3b", lw=1.8))
    ax.annotate("", (ox, oz + 0.9), (ox, oz), arrowprops=dict(arrowstyle="-|>", color="#2f6fd1", lw=1.8))
    ax.add_patch(patches.Circle((ox, oz), 0.14, facecolor="white", edgecolor="#0ca30c", lw=1.6))
    ax.add_patch(patches.Circle((ox, oz), 0.04, color="#0ca30c"))
    ax.text(ox + 1.0, oz, "x", color="#d03b3b", fontsize=10, va="center"); ax.text(ox, oz + 1.05, "z", color="#2f6fd1", fontsize=10, ha="center")
    ax.text(ox - 0.25, oz - 0.3, "y (out of page)", color="#0ca30c", fontsize=9, ha="right")
    ax.set_xlim(-4.2, 6.2); ax.set_ylim(-2.1, z + 0.8)
    # symbol key
    ky = -0.15
    kx, kz = -3.9, -1.55                               # symbol key, below the base
    ax.text(kx, kz + 0.75, "Symbols", fontsize=10, fontweight="bold")
    ax.add_patch(patches.FancyBboxPatch((kx + 0.05, kz + 0.25), 0.6, 0.3, boxstyle="round,pad=0.02", facecolor="white", edgecolor=C_Z, lw=2))
    ax.text(kx + 0.9, kz + 0.4, "twist joint: axis along the link", fontsize=8.5, va="center", color=C_Z)
    ax.add_patch(patches.Circle((kx + 4.85, kz + 0.4), 0.2, facecolor="white", edgecolor=C_Y, lw=2))
    ax.add_patch(patches.Circle((kx + 4.85, kz + 0.4), 0.05, color=C_Y))
    ax.text(kx + 5.3, kz + 0.4, "bend joint: axis out of the page", fontsize=8.5, va="center", color=C_Y)


# ------------------------------------------------------------------------------------------ 3-D view
def axes_3d(ax, q):
    O, A, F, tcp, Rt = fk(q)
    pts = np.vstack([O, tcp])
    ax.set_title("B · Every joint axis in 3-D (example pose)", fontsize=13, fontweight="bold", loc="left")
    cylinder(ax, MOUNT, MOUNT + [0, 0, PEDESTAL["length"]], PEDESTAL["radius"], "#7d8796", 0.6)
    for i in range(7):
        cylinder(ax, pts[i], pts[i + 1], RADII[i] * 0.45, C_LINK, 0.28)
    ax.plot(*pts.T, color="#3c4656", lw=1.8, alpha=0.8)
    offsets = [(0.05, -0.10, -0.02), (0.0, 0.10, 0.04), (0.06, 0.06, 0.0), (-0.03, -0.14, 0.03),
               (0.06, 0.05, 0.02), (-0.04, 0.14, 0.02), (0.05, -0.08, 0.04)]
    labels = []
    for i, (p, d) in enumerate(zip(O, A)):
        col = C_Z if AXES[i] == "z" else C_Y
        a, b = p - d * 0.16, p + d * 0.24                # long axis line through the joint
        ax.plot(*np.c_[a, b], color=col, lw=1.4, ls=(0, (4, 3)))
        ax.quiver(*p, *(d * 0.24), color=col, lw=3.0, arrow_length_ratio=0.25)
        # rotation arrow around the axis
        u = np.cross(d, [1, 0, 0] if abs(d[0]) < 0.9 else [0, 1, 0]); u /= np.linalg.norm(u)
        v = np.cross(d, u)
        th = np.linspace(0.3, 1.8 * np.pi, 50)
        c = p + d * 0.11
        ring = c[:, None] + 0.055 * (u[:, None] * np.cos(th) + v[:, None] * np.sin(th))
        ax.plot(*ring, color=col, lw=1.6)
        ax.quiver(*ring[:, -3], *(ring[:, -1] - ring[:, -3]), color=col, arrow_length_ratio=3, lw=1.6)
        ax.scatter(*p, s=60, color=col, edgecolor="white", linewidth=1, depthshade=False, zorder=10)
        labels.append((p, i, col))
    ax.scatter(*tcp, s=50, color="#111", depthshade=False)
    ax.text(*(tcp + [0.04, 0.02, 0.05]), "TCP", fontsize=10, fontweight="bold")
    lo, hi = pts.min(0) - 0.18, pts.max(0) + 0.2
    span = (hi - lo).max()
    mid = (hi + lo) / 2
    ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2); ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
    ax.set_zlim(mid[2] - span / 2, mid[2] + span / 2); ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=16, azim=-60)
    ax.set_xlabel("x (m)", fontsize=8, labelpad=-4); ax.set_ylabel("y (m)", fontsize=8, labelpad=-4)
    ax.set_zlabel("z (m)", fontsize=8, labelpad=-4); ax.tick_params(labelsize=6.5, pad=-1)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((0.97, 0.98, 0.99, 1)); a.pane.set_edgecolor("#dde3ea")
        a._axinfo["grid"]["color"] = (0.86, 0.89, 0.93, 1)
    # labels at fixed screen offsets with leader lines: twist joints to the left, bend joints to the right
    from mpl_toolkits.mplot3d import proj3d
    for p, i, col in labels:
        x2, y2, _ = proj3d.proj_transform(*p, ax.get_proj())
        dx = -95 if AXES[i] == "z" else 85
        ax.annotate(f"J{i + 1} · axis {AXES[i]}", (x2, y2), xytext=(dx, 6 if i % 2 else -6), textcoords="offset points",
                    color=col, fontsize=10.5, fontweight="bold", ha="center", va="center", zorder=30,
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=col, lw=1),
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.9, shrinkA=0, shrinkB=4))
    ax.text2D(0.02, 0.01, "Solid arrow = positive rotation axis (right-hand rule: thumb along the arrow, fingers show +q). "
              "Dashed line = the full axis through the joint.", transform=ax.transAxes, fontsize=8, color="#46546a")


def main():
    plt.rcParams.update({"font.family": "Inter"})
    fig = plt.figure(figsize=(16, 10.2), dpi=180)
    fig.text(0.03, 0.962, "STP_Spacearm 7-DOF arm — the axis of every joint", fontsize=19, fontweight="bold")
    fig.text(0.03, 0.935, "Axis order z · y · z · y · z · y · z  (shoulder J1–J3 · elbow J4 · wrist J5–J7). "
             "Each axis is expressed in the frame of the link before it; at q = 0 all z axes point up and all y axes point along +y.",
             fontsize=10.5, color="#46546a")
    schematic(fig.add_axes([0.02, 0.04, 0.42, 0.86]))
    q = np.radians(QDEG)
    ax3 = fig.add_axes([0.42, 0.04, 0.57, 0.86], projection="3d")
    axes_3d(ax3, q)
    fig.text(0.52, 0.875, "q = (" + ", ".join(f"{v:g}" for v in QDEG).replace("-", "−") + ")°", fontsize=10, color="#46546a")
    rows = "   ".join(f"J{i + 1}: {AXES[i]} [{LOWER[i]}°, {UPPER[i]}°]" for i in range(7))
    fig.text(0.03, 0.012, "Limits  " + rows + "      ·  Source: akashrayganguly/STP_Spacearm configs/default.yaml "
             "(commit a15a269); drawn from the configuration, no project code run.", fontsize=7.5, color="#7a879a")
    fig.savefig(OUT.with_suffix(".png"), dpi=180, facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), facecolor="white")
    print("wrote", OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
