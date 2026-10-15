#!/usr/bin/env python
"""3-D kinematics diagram of the STP_Spacearm robot (teammate project), drawn from its configuration only.

Geometry and the joint chain are taken from akashrayganguly/STP_Spacearm configs/default.yaml and the
frame rule in src/spacearm/kinematics.py (t <- t + R e_z * offset_i; R <- R Rot(axis_i, q_i);
TCP = t_7 + R_7 e_z * tool_offset). None of that project's code is run; the forward kinematics is
re-implemented here in NumPy for drawing.

    python deliverables/scripts/docs/spacearm_kinematics_diagram.py
Output: deliverables/documentation/STP_Spacearm_kinematics_3d.png (+ .pdf)
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

# ---- configuration (configs/default.yaml, commit a15a269) ------------------------------------------
BUS = (1.0, 1.0, 1.0)
PANEL, PANEL_GAP = (0.8, 1.6, 0.02), 0.05
PAYLOAD = dict(size=(0.40, 0.50, 0.30), xyz=(-0.20, 0.0, 0.65))
MOUNT = np.array([0.30, 0.0, 0.50])
PEDESTAL = dict(length=0.10, radius=0.07)
AXES = ["z", "y", "z", "y", "z", "y", "z"]
LINKS = [0.24, 0.20, 0.20, 0.20, 0.20, 0.14, 0.12]
RADII = [0.06, 0.06, 0.06, 0.06, 0.06, 0.05, 0.04]
LOWER = [-170, -120, -170, -120, -170, -120, -175]
UPPER = [170, 120, 170, 120, 170, 120, 175]
TOOL = 0.15

C_ROLL, C_PITCH = "#d95926", "#2f6fd1"          # z joints (twist) / y joints (bend)
C_LINK, C_BUS, C_PANEL, C_PAY = "#9aa7b8", "#d8b25a", "#2b4f9e", "#b9bfc8"
OUT = Path(__file__).resolve().parents[2] / "documentation" / "STP_Spacearm_kinematics_3d"


def rot(axis, q):
    c, s = np.cos(q), np.sin(q)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def fk(q):
    """Joint origins, joint axes (world), frames after each joint, TCP position and frame."""
    t, R = MOUNT.astype(float).copy(), np.eye(3)
    offsets = [PEDESTAL["length"]] + LINKS[:-1]
    origins, axes, frames = [], [], []
    for i, (ax, off) in enumerate(zip(AXES, offsets)):
        t = t + R[:, 2] * off
        axes.append(R[:, "xyz".index(ax)].copy())        # rotation axis of joint i, before it turns
        R = R @ rot(ax, q[i])
        origins.append(t.copy())
        frames.append(R.copy())
    tcp = t + R[:, 2] * TOOL
    return np.array(origins), np.array(axes), frames, tcp, R


def box(ax, centre, size, color, alpha, edge="#555555", lw=0.4):
    c, s = np.asarray(centre, float), np.asarray(size, float) / 2
    v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * s + c
    faces = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    ax.add_collection3d(Poly3DCollection([v[f] for f in faces], facecolor=color, edgecolor=edge, linewidths=lw, alpha=alpha))


def cylinder(ax, p0, p1, r, color, alpha=0.9, n=24):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = np.linalg.norm(d)
    w = d / L
    a = np.array([1, 0, 0]) if abs(w[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(w, a); u /= np.linalg.norm(u)
    v = np.cross(w, u)
    th = np.linspace(0, 2 * np.pi, n)
    s = np.linspace(0, 1, 2)
    TH, S = np.meshgrid(th, s)
    P = p0[:, None, None] + d[:, None, None] * S + r * (u[:, None, None] * np.cos(TH) + v[:, None, None] * np.sin(TH))
    ax.plot_surface(P[0], P[1], P[2], color=color, alpha=alpha, linewidth=0, shade=True)


def arrow(ax, p, d, color, length, lw=2.0, label=None, fs=8):
    q = p + d * length
    ax.plot(*np.c_[p, q], color=color, lw=lw)
    ax.quiver(*(q - d * 0.035), *(d * 0.035), color=color, arrow_length_ratio=1.0, lw=lw)
    if label:
        ax.text(*(q + d * 0.03), label, color=color, fontsize=fs, fontweight="bold", ha="center", va="center")


def rotation_arc(ax, p, d, color, r=0.075):
    a = np.array([1, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(d, a); u /= np.linalg.norm(u)
    v = np.cross(d, u)
    th = np.linspace(0.2, 1.75 * np.pi, 40)
    pts = p[:, None] + r * (u[:, None] * np.cos(th) + v[:, None] * np.sin(th))
    ax.plot(*pts, color=color, lw=1.3)
    tip, prev = pts[:, -1], pts[:, -4]
    ax.quiver(*prev, *(tip - prev), color=color, arrow_length_ratio=2.5, lw=1.3)


def triad(ax, p, R, length, labels, fs=8, lw=1.6):
    for k, (c, lab) in enumerate(zip(("#d03b3b", "#0ca30c", "#2f6fd1"), labels)):
        arrow(ax, p, R[:, k], c, length, lw=lw, label=lab, fs=fs)


def spacecraft(ax):
    box(ax, (0, 0, 0), BUS, C_BUS, 0.25)
    yc = BUS[1] / 2 + PANEL_GAP + PANEL[1] / 2
    for sgn in (-1, 1):
        box(ax, (0, sgn * yc, 0), PANEL, C_PANEL, 0.35, edge="#1d3566")
        ax.plot([0, 0], [sgn * BUS[1] / 2, sgn * (yc - PANEL[1] / 2)], [0, 0], color="#555", lw=1.2)
    box(ax, PAYLOAD["xyz"], PAYLOAD["size"], C_PAY, 0.45)
    cylinder(ax, MOUNT, MOUNT + [0, 0, PEDESTAL["length"]], PEDESTAL["radius"], "#7d8796", 0.95)


def arm(ax, q, labels=True, dims=False):
    O, A, F, tcp, Rt = fk(q)
    pts = np.vstack([O, tcp])
    for i in range(7):
        p1 = pts[i + 1]
        cylinder(ax, pts[i], p1, RADII[i] * 0.55, C_LINK, 0.55)
    ax.plot(*pts.T, color="#3c4656", lw=2.2)
    for i, (p, d) in enumerate(zip(O, A)):
        col = C_ROLL if AXES[i] == "z" else C_PITCH
        ax.scatter(*p, s=46, color=col, edgecolor="white", linewidth=0.8, depthshade=False, zorder=5)
        arrow(ax, p, d, col, 0.16, lw=1.8)
        rotation_arc(ax, p + d * 0.05, d, col)
        if labels:
            off = np.array([0.10, 0.06, 0.0]) if AXES[i] == "z" else np.array([-0.10, -0.20, 0.02])
            ax.text(*(p + off), f"J{i + 1}", color=col, fontsize=10, fontweight="bold", zorder=20,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=col, lw=0.8, alpha=0.95))
    triad(ax, tcp, Rt, 0.14, ("x", "y", "z"), fs=7, lw=1.3)
    ax.scatter(*tcp, s=40, color="#111", depthshade=False)
    ax.text(*(tcp + [0.05, 0.0, 0.05]), "TCP", fontsize=9, fontweight="bold")
    if dims:                                           # link lengths along the straight arm
        x = MOUNT[0] + 0.32
        zs = [MOUNT[2]] + list(O[:, 2]) + [tcp[2]]
        names = ["pedestal 10"] + [f"L{i + 1} {round(100 * l)}" for i, l in enumerate(LINKS[:-1])] + ["tool 15"]
        for k in range(len(zs) - 1):
            ax.plot([x, x], [0, 0], [zs[k], zs[k + 1]], color="#444", lw=0.9)
            ax.plot([x - 0.02, x + 0.02], [0, 0], [zs[k], zs[k]], color="#444", lw=0.9)
            ax.text(x + 0.04, 0, (zs[k] + zs[k + 1]) / 2, names[k] + " cm", fontsize=7, color="#333", va="center")
        ax.plot([x - 0.02, x + 0.02], [0, 0], [zs[-1], zs[-1]], color="#444", lw=0.9)
    return O, tcp


def setup(ax, title, elev, azim, lim):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=2)
    (x0, x1), (y0, y1), (z0, z1) = lim
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(z0, z1)
    ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x (m)", fontsize=8, labelpad=-6); ax.set_ylabel("y (m)", fontsize=8, labelpad=-6)
    ax.set_zlabel("z (m)", fontsize=8, labelpad=-6)
    ax.tick_params(labelsize=6, pad=-2)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((0.97, 0.98, 0.99, 1)); a.pane.set_edgecolor("#dde3ea")
        a._axinfo["grid"]["color"] = (0.85, 0.88, 0.92, 1)


def main():
    plt.rcParams.update({"font.family": "Inter"})
    fig = plt.figure(figsize=(16, 11.2), dpi=180)
    fig.text(0.03, 0.965, "STP_Spacearm robot — 3-D kinematics diagram (7 DOF: joints J1–J7)", fontsize=19, fontweight="bold")
    fig.text(0.03, 0.94, "7-DOF S-R-S arm (shoulder 3 · elbow 1 · wrist 3) on a free-floating spacecraft · "
             "body frame B at the bus centre (= centre of mass), z through the arm-mount face, y along the panels",
             fontsize=10.5, color="#46546a")

    ax1 = fig.add_axes([0.0, 0.24, 0.52, 0.62], projection="3d")
    setup(ax1, "Zero configuration  q = 0  (arm straight up)", 18, -58, ((-0.9, 1.0), (-2.25, 2.25), (-0.6, 2.05)))
    spacecraft(ax1)
    triad(ax1, np.zeros(3), np.eye(3), 0.42, ("x_B", "y_B", "z_B"), fs=9, lw=2.0)
    ax1.text(0.04, 0.04, -0.12, "B", fontsize=11, fontweight="bold")
    arm(ax1, np.zeros(7), dims=True)

    ax2 = fig.add_axes([0.47, 0.24, 0.52, 0.62], projection="3d")
    q = np.radians([35, 50, -20, -85, 30, 55, 0])
    setup(ax2, "Example pose  q = (35, 50, −20, −85, 30, 55, 0)°", 20, -40, ((-0.6, 1.4), (-0.9, 1.1), (0.3, 1.7)))
    box(ax2, (0, 0, 0.25), (1.0, 1.0, 0.5), C_BUS, 0.25)              # upper half of the bus only (zoomed view)
    box(ax2, PAYLOAD["xyz"], PAYLOAD["size"], C_PAY, 0.45)
    cylinder(ax2, MOUNT, MOUNT + [0, 0, PEDESTAL["length"]], PEDESTAL["radius"], "#7d8796", 0.95)
    O, tcp = arm(ax2, q)
    ax2.text2D(0.02, 0.02, f"TCP in frame B: ({tcp[0]:.3f}, {tcp[1]:.3f}, {tcp[2]:.3f}) m  (computed with the chain below)",
               transform=ax2.transAxes, fontsize=8.5, color="#333")

    # legend
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=C_ROLL, marker="o", lw=2, label="z joint: twist about the link (J1, J3, J5, J7)"),
         Line2D([], [], color=C_PITCH, marker="o", lw=2, label="y joint: bend across the link (J2, J4, J6)"),
         Line2D([], [], color="#d03b3b", lw=2, label="x axis"), Line2D([], [], color="#0ca30c", lw=2, label="y axis"),
         Line2D([], [], color="#2f6fd1", lw=2, label="z axis")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.03, 0.93), ncol=5, frameon=False, fontsize=9)

    # joint table + chain equations
    rows = []
    offs = [PEDESTAL["length"]] + LINKS[:-1]
    for i in range(7):
        rows.append([f"J{i + 1}", AXES[i], "twist" if AXES[i] == "z" else "bend", f"{100 * offs[i]:.0f} cm",
                     f"{100 * LINKS[i]:.0f} cm", f"{100 * RADII[i]:.0f} cm", f"{LOWER[i]}° … {UPPER[i]}°"])
    tab = fig.add_axes([0.03, 0.025, 0.55, 0.2]); tab.axis("off")
    t = tab.table(cellText=rows, colLabels=["Joint", "Axis (own frame)", "Motion", "Offset from prev. joint",
                                            "Link length", "Capsule radius", "Limits"],
                  loc="upper left", cellLoc="center", colLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.25)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#d9e0ea")
        if r == 0:
            cell.set_facecolor("#eef2f7"); cell.set_text_props(fontweight="bold")
        elif c == 1:
            cell.set_text_props(color=C_ROLL if rows[r - 1][1] == "z" else C_PITCH, fontweight="bold")
    eq = fig.add_axes([0.61, 0.025, 0.37, 0.2]); eq.axis("off")
    eq.text(0, 0.98, "Forward kinematics (body frame B)", fontsize=11, fontweight="bold", va="top")
    eq.text(0, 0.80, r"$t_0 = (0.30,\ 0,\ 0.50)\,$m (mount),  $R_0 = I$", fontsize=10, va="top")
    eq.text(0, 0.62, r"for $i = 1..7$:   $t_i = t_{i-1} + R_{i-1}\,e_z\,\ell_{i-1}$,   "
                     r"$R_i = R_{i-1}\,\mathrm{Rot}(a_i,\ q_i)$", fontsize=10, va="top")
    eq.text(0, 0.44, r"$\ell_0 = 10$ cm (pedestal),  $\ell_1..\ell_6$ = link lengths,  $a_i \in \{z,y,z,y,z,y,z\}$",
            fontsize=10, va="top")
    eq.text(0, 0.26, r"$x_{\mathrm{TCP}} = t_7 + R_7\,e_z\cdot 15$ cm   ·   reach ≈ 1.1 m from the shoulder", fontsize=10, va="top")
    eq.text(0, 0.17, "7 joints but only 3 tip coordinates (position): 4 spare DOF; 7 − 6 = 1 spare even for a full pose (null space).",
            fontsize=8.5, color="#46546a", va="top")
    eq.text(0, 0.08, "Source: akashrayganguly/STP_Spacearm configs/default.yaml and kinematics.py (commit a15a269).\n"
                     "Drawn from the configuration; none of that project's code was run.", fontsize=7.5, color="#7a879a", va="top")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=180, facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), facecolor="white")
    print("wrote", OUT.with_suffix(".png"), "and .pdf;  example TCP", np.round(tcp, 3))


if __name__ == "__main__":
    main()
