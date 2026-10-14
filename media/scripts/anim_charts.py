#!/usr/bin/env python
"""Animated charts, counters and title cards for the marketing video (see media/PRODUCTION_BRIEF.md).

Every number on screen is read from the run files listed in section 6 of the brief - nothing is typed by
hand, except PLAN_BASELINE_MM (it exists only in the Level 1 plan PDF).

Each clip is a function draw(fig, t, F) that paints one frame at time t (seconds) in pixel coordinates
(0..1920 x 0..1080, y up). Frames are piped straight into ffmpeg as raw RGBA -> H.264, 30 fps.

    python media/scripts/anim_charts.py                    # render every clip into media/assets/
    python media/scripts/anim_charts.py ik_drop waffle     # only some clips
    python media/scripts/anim_charts.py --stills           # one PNG per clip (fast layout check)
    python media/scripts/anim_charts.py --list             # clip names, durations, storyboard scene
"""
import argparse
import json
import math
import multiprocessing as mp
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
WS = ROOT / "ui" / "workspace"
OUT = ROOT / "media" / "assets"
sys.path.insert(0, str(ROOT))

# ---- sources (brief section 6) ----------------------------------------------------------------------
L1_RUN = "20260928-174703_xarm6"                 # best network alone
L1_LM_RUN = "20261005-155310_xarm6"              # network + LM refinement
L2_POLICY = "l2_20261013-185525_xarm6"           # best Level 2 agent
MISSION_DYNAMIC = "mb_20261013-211225_xarm6_dynamic_n100"
MISSION_COMBINED = "mb_20261013-211258_xarm6_combined_n100"
PLAN_BASELINE_MM = 128.0                         # Level 1 plan PDF (network alone) - no data file holds it
# The fact sheet quotes the 200-episode evaluation. A later re-evaluation in the Studio (n = 100) overwrote the agent's
# meta.json, so the media read this snapshot of the original file instead.
L2_META_SNAPSHOT = Path(__file__).resolve().parents[1] / "data" / f"{L2_POLICY}_meta_n200.json"

# ---- look & feel (brief section 3) ------------------------------------------------------------------
W, H, FPS, DPI = 1920, 1080, 30, 100
BG, SURF, SURF2, BORDER = "#0e1116", "#151a21", "#1b2129", "#2a313c"
TXT, TXT2, MUTED = "#eef2f7", "#b3bdcb", "#7f8a9a"
C_L1, C_L2, C_MISSION = "#4b8ff0", "#7c5cff", "#1f9d63"
C_AGENT, C_APF, C_STRAIGHT = "#3987e5", "#d95926", "#199e70"
C_REACHED, C_COLL, C_STUCK = "#0ca30c", "#d03b3b", "#fab219"
LOGO_STOPS = ["#4b8ff0", "#7c5cff", "#199e70", "#4b8ff0"]      # conic-gradient(from 210deg, ...) as in ui/static/style.css
FOOTER = "in simulation · xArm6 · STP 2026 · Team 06"

FP = {}


def _setup_fonts():
    if FP:
        return
    files = {"light": "segoeuil.ttf", "regular": "segoeui.ttf", "semibold": "seguisb.ttf", "bold": "segoeuib.ttf"}
    win, cache = Path("/mnt/c/Windows/Fonts"), Path.home() / ".cache" / "neural_ik_fonts"
    if all((win / f).exists() or (cache / f).exists() for f in files.values()):
        cache.mkdir(parents=True, exist_ok=True)
        for w, f in files.items():
            if not (cache / f).exists():                 # FreeType reads fonts lazily; /mnt/c (9p) fails under load
                shutil.copy(win / f, cache / (f + ".part"))
                (cache / (f + ".part")).replace(cache / f)
            font_manager.fontManager.addfont(str(cache / f))
            FP[w] = FontProperties(fname=str(cache / f))
        plt.rcParams["font.family"] = "Segoe UI"
    else:                                          # Inter (fonts-inter) or DejaVu as a fallback
        fam = "Inter" if any("Inter" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
        weights = {"light": "light", "regular": "normal", "semibold": "semibold", "bold": "bold"}
        for w, mw in weights.items():
            FP[w] = FontProperties(family=fam, weight=mw)
    plt.rcParams["image.aspect"] = "auto"


# ================================================================================================ data
def _json(p):
    with open(p) as f:
        return json.load(f)


def load_facts():
    """Everything the clips show, read from the run files."""
    F = {}
    l1 = _json(WS / "library" / "runs" / L1_RUN / "meta.json")
    F["l1_net_mm"] = l1["metrics"]["test"]["pos_mean_mm"]
    F["l1_net_params"] = l1["model"]["params"]
    lm = _json(WS / "library" / "runs" / L1_LM_RUN / "meta.json")["metrics"]["test_refined"]
    F["l1_lm_mm"] = lm["pos_mean_mm"]
    F["l1_lm_within_1mm_1deg"] = lm["success"]["1|1"]
    F["l1_lm_self_coll"] = lm["self_collision_pct"]
    F["l1_lm_limits"] = lm["limit_violation_pct"]
    F["plan_mm"] = PLAN_BASELINE_MM

    pol = WS / "level2" / "policies" / L2_POLICY
    meta = _json(L2_META_SNAPSHOT if L2_META_SNAPSHOT.exists() else pol / "meta.json")
    ev = meta["evaluation"]
    F["l2_eval"] = ev["scenarios"]
    F["l2_n"] = ev["n_per_scenario"]
    F["l2_params"] = ev["policy_params"]
    F["l2_us"] = ev["decision_us"]
    F["l2_steps"] = meta["steps_done"]
    F["l2_minutes"] = meta["train_secs"] / 60
    hist = _json(pol / "history.json")
    F["l2_hist"] = hist
    F["l2_envs"], F["l2_workers"] = hist[-1]["envs"], hist[-1]["workers"]
    try:                                           # layer sizes for the network drawing
        import torch
        ck = torch.load(pol / "policy.pt", map_location="cpu", weights_only=False)
        F["l2_layers"] = (ck["obs_dim"], ck["hidden"], ck["hidden"], ck["act_dim"])
        o, h, _, a = F["l2_layers"]
        assert (o + 1) * h + (h + 1) * h + (h + 1) * a == F["l2_params"], "actor size != policy_params"
    except ImportError:
        F["l2_layers"] = None

    for key, mid in (("m_dyn", MISSION_DYNAMIC), ("m_comb", MISSION_COMBINED)):
        m = _json(WS / "missions" / mid / "mission.json")
        ok = [r["episodes"]["policy"]["final_pos_err_mm"] for r in m["rows"] if r["episodes"]["policy"]["outcome"] == "success"]
        m["median_final_mm"] = statistics.median(ok)
        F[key] = m
    try:
        from robot_config import ROBOTS
        names = {"xarm6": "xArm6", "kuka_iiwa": "KUKA LBR iiwa", "panda": "Franka Panda"}
        F["robots"] = [(names.get(k, k), v["dof"]) for k, v in ROBOTS.items()]
    except ImportError:
        F["robots"] = []
    return F


# ============================================================================================ helpers
def clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


def prog(t, t0, dur):
    return clamp((t - t0) / dur)


def ease_out(p):
    return 1 - (1 - p) ** 3


def ease_io(p):
    return 4 * p ** 3 if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2


def ease_back(p, s=1.6):
    return 1 + (s + 1) * (p - 1) ** 3 + s * (p - 1) ** 2


def appear(t, t0, dur=0.45):
    return ease_out(prog(t, t0, dur))


def lerp(a, b, k):
    return a + (b - a) * k


def mix(c0, c1, k):
    a, b = np.array(to_rgb(c0)), np.array(to_rgb(c1))
    return tuple(a + (b - a) * k)


def loglerp(a, b, k):
    return 10 ** lerp(math.log10(a), math.log10(b), k)


def fmt_mm(v):
    return f"{v:.0f}" if v >= 100 else f"{v:.1f}" if v >= 10 else f"{v:.2f}" if v >= 1 else f"{v:.3f}" if v >= 0.01 else f"{v:.4f}"


def fmt_pct(v):
    return f"{v:.0f} %" if abs(v - round(v)) < 0.05 else f"{v:.1f} %"


def _make_bg():
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.hypot((x - W * 0.5) / W, (y - H * 0.45) / H * 0.75)
    k = np.clip(d / 0.62, 0, 1) ** 1.4
    c0, c1 = np.array(to_rgb(SURF)) * 255, np.array(to_rgb(BG)) * 255
    img = c0 * (1 - k[..., None]) + c1 * k[..., None]
    img += np.random.default_rng(0).uniform(-0.6, 0.6, img.shape)          # dither: no banding after H.264
    return np.clip(img, 0, 255).astype(np.uint8)


def _make_logo(n=256):
    y, x = np.mgrid[0:n, 0:n] + 0.5
    ang = ((np.degrees(np.arctan2(x - n / 2, -(y - n / 2))) - 210) % 360) / 360   # clockwise from top, like CSS
    stops = np.array([to_rgb(c) for c in LOGO_STOPS])
    pos = np.linspace(0, 1, len(stops))
    rgb = np.stack([np.interp(ang, pos, stops[:, i]) for i in range(3)], -1)
    return rgb, ang


BG_IMG = _make_bg()
LOGO_RGB, LOGO_ANG = _make_logo()
_GL = np.exp(-(np.mgrid[-1:1:128j, -1:1:128j] ** 2).sum(0) / (2 * 0.33 ** 2))
_SR = np.random.default_rng(7)
STARS = {"x": _SR.uniform(0, W, 240), "y": _SR.uniform(0, H, 240), "r": _SR.uniform(0.6, 2.2, 240),
         "ph": _SR.uniform(0, 2 * np.pi, 240), "f": _SR.uniform(0.5, 1.6, 240)}


def canvas(fig):
    fig.clf()
    fig.figimage(BG_IMG, zorder=-10, origin="upper")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_autoscale_on(False)
    ax.axis("off")
    ax.patch.set_alpha(0)
    return ax


def T(ax, x, y, s, size=28, color=TXT, w="regular", ha="left", va="baseline", alpha=1.0, z=4, **kw):
    fp = FP[w].copy()
    fp.set_size(size)
    return ax.text(x, y, s, fontproperties=fp, color=color, ha=ha, va=va, alpha=clamp(alpha), zorder=z, **kw)


def text_w(ax, t):
    return t.get_window_extent(renderer=ax.figure.canvas.get_renderer()).width


def rbox(ax, x, y, w, h, r, fc=SURF, ec=BORDER, lw=1.2, alpha=1.0, z=1):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={min(r, w / 2, h / 2)}",
                       fc=fc, ec=ec, lw=lw, alpha=clamp(alpha), zorder=z)
    ax.add_patch(p)
    return p


def glow(ax, cx, cy, r, color, alpha, z=0.5):
    if alpha <= 0.002 or r <= 0:
        return
    img = np.zeros((128, 128, 4))
    img[..., :3] = to_rgb(color)
    img[..., 3] = _GL * clamp(alpha)
    ax.imshow(img, extent=(cx - r, cx + r, cy - r, cy + r), zorder=z, interpolation="bilinear")


def logo(ax, cx, cy, size, alpha=1.0, sweep=1.0, z=5):
    if alpha <= 0 or sweep <= 0 or size <= 1:
        return
    a = np.clip((sweep * 1.03 - LOGO_ANG) * 40, 0, 1) * clamp(alpha)        # soft clock-hand reveal
    img = np.dstack([LOGO_RGB, a])
    clip = FancyBboxPatch((cx - size / 2, cy - size / 2), size, size, boxstyle=f"round,pad=0,rounding_size={size * 0.32}",
                          fc="none", ec="none", transform=ax.transData)
    ax.add_patch(clip)
    im = ax.imshow(img, extent=(cx - size / 2, cx + size / 2, cy - size / 2, cy + size / 2), origin="upper",
                   zorder=z, interpolation="bilinear")
    im.set_clip_path(clip)


def stars(ax, t, alpha=1.0):
    r = STARS["r"]
    x = (STARS["x"] - t * 5 * r) % W                                        # bigger = closer = faster (parallax)
    a = (0.18 + 0.32 * (0.5 + 0.5 * np.sin(STARS["f"] * t * 2 + STARS["ph"]))) * alpha * (r / 2.2) ** 0.5
    col = np.zeros((len(r), 4))
    col[:, :3] = to_rgb(TXT2)
    col[:, 3] = np.clip(a, 0, 1)
    ax.scatter(x, STARS["y"], s=(r * 0.72 * 2) ** 2, c=col, linewidths=0, zorder=0.2)


def fade_overlay(ax, a, color="black"):
    if a > 0.002:
        ax.add_patch(Rectangle((0, 0), W, H, fc=color, ec="none", alpha=clamp(a), zorder=100))


def pill(ax, x, y, label, color, alpha=1.0, size=17):
    t = T(ax, x + 44, y, label, size, TXT, "semibold", va="center_baseline", alpha=alpha, z=6)
    rbox(ax, x, y - 23, text_w(ax, t) + 68, 46, 23, fc=SURF2, ec=BORDER, alpha=alpha, z=5)
    ax.add_patch(Circle((x + 24, y), 7, fc=color, ec="none", alpha=clamp(alpha), zorder=6))
    glow(ax, x + 24, y, 26, color, 0.45 * alpha, z=5.5)


def chrome(ax, t, chapter=None, color=None):
    a = appear(t, 0.05, 0.5)
    logo(ax, 136, 72, 30, alpha=a)
    T(ax, 164, 72, "Neural IK Studio", 17, TXT2, "semibold", va="center_baseline", alpha=a)
    T(ax, W - 120, 72, FOOTER, 16, MUTED, ha="right", va="center_baseline", alpha=a)
    if chapter:
        pill(ax, 120, 990, chapter, color, alpha=appear(t, 0.1, 0.5))


def heading(ax, t, title, sub=None, t0=0.25):
    a = appear(t, t0, 0.5)
    T(ax, 120, 878 - 22 * (1 - a), title, 46, TXT, "semibold", alpha=a)
    if sub:
        b = appear(t, t0 + 0.15, 0.5)
        T(ax, 122, 826 - 16 * (1 - b), sub, 21, MUTED, alpha=b)


def check(ax, cx, cy, r, alpha=1.0, color=C_REACHED):
    ax.add_patch(Circle((cx, cy), r, fc=color, ec="none", alpha=clamp(alpha), zorder=5))
    ax.plot([cx - 0.45 * r, cx - 0.1 * r, cx + 0.5 * r], [cy + 0.02 * r, cy - 0.35 * r, cy + 0.4 * r],
            color="white", lw=r * 0.17, solid_capstyle="round", solid_joinstyle="round", alpha=clamp(alpha), zorder=6)


def glow_line(ax, x, y, color, lw=3.0, alpha=1.0, z=4):
    for w, a in ((lw * 4.5, 0.06), (lw * 2.4, 0.14), (lw, 1.0)):
        ax.plot(x, y, color=color, lw=w, alpha=clamp(a * alpha), solid_capstyle="round", solid_joinstyle="round", zorder=z)


# ============================================================================================== clips
def draw_title(fig, t, F):
    """Scene 1 (0-6 s): logo builds out of black, title and the three beats."""
    ax = canvas(fig)
    stars(ax, t, appear(t, 0.2, 1.5))
    s = ease_io(prog(t, 0.35, 1.1))
    pop = 0.84 + 0.16 * ease_back(prog(t, 0.35, 0.8))
    cx, cy = W / 2, 690
    glow(ax, cx, cy, 330 * pop, C_L2, 0.30 * s)
    glow(ax, cx, cy, 200 * pop, C_L1, 0.25 * s)
    logo(ax, cx, cy, 170 * pop, sweep=s)
    a = appear(t, 1.15, 0.55)
    T(ax, cx, 470 - 26 * (1 - a), "Neural IK Studio", 80, TXT, "semibold", ha="center", alpha=a)
    a = appear(t, 1.5, 0.5)
    T(ax, cx, 404 - 18 * (1 - a), "Team 06 · STP 2026", 30, TXT2, ha="center", alpha=a)
    a = appear(t, 1.85, 0.5)
    T(ax, cx, 352 - 14 * (1 - a), "Neural kinematics & reinforcement learning for collision-free space manipulators",
      22, MUTED, ha="center", alpha=a)
    beats = [("1", "Solve", C_L1), ("2", "Steer", C_L2), ("3", "Together", C_MISSION)]
    for i, (n, label, col) in enumerate(beats):
        x = cx + (i - 1) * 300
        b = appear(t, 2.6 + 0.25 * i, 0.45)
        sc = 0.9 + 0.1 * ease_back(prog(t, 2.6 + 0.25 * i, 0.45))
        rbox(ax, x - 110 * sc, 230 - 30 * sc, 220 * sc, 60 * sc, 30 * sc, fc=SURF2, ec=BORDER, alpha=b, z=3)
        ax.add_patch(Circle((x - 70, 230), 15, fc=col, ec="none", alpha=b, zorder=4))
        glow(ax, x - 70, 230, 40, col, 0.5 * b, z=3.5)
        T(ax, x - 70, 230, n, 15, "white", "bold", ha="center", va="center_baseline", alpha=b, z=5)
        T(ax, x - 42, 230, label, 24, TXT, "semibold", va="center_baseline", alpha=b)
        if i < 2:                                                   # connector draws left to right
            k = ease_out(prog(t, 2.85 + 0.25 * i, 0.35))
            if k > 0:
                ax.plot([x + 112, x + 112 + 76 * k], [230, 230], color=BORDER, lw=2, zorder=2)
    fade_overlay(ax, 1 - ease_out(prog(t, 0.0, 0.7)))


def draw_end(fig, t, F):
    """Scene 9: logo, project name, the three supported arms, fade to black (long enough for the closing line)."""
    ax = canvas(fig)
    stars(ax, t + 10)
    a = appear(t, 0.1, 0.6)
    glow(ax, W / 2, 790, 220, C_L2, 0.25 * a)
    logo(ax, W / 2, 790, 110 * (0.9 + 0.1 * ease_back(prog(t, 0.1, 0.6))), sweep=1, alpha=a)
    a = appear(t, 0.4, 0.55)
    T(ax, W / 2, 610 - 22 * (1 - a), "Neural Kinematics & Reinforcement Learning", 58, TXT, "semibold", ha="center", alpha=a)
    a = appear(t, 0.65, 0.55)
    T(ax, W / 2, 540 - 16 * (1 - a), "for collision-free space manipulators", 38, TXT2, ha="center", alpha=a)
    robots = F["robots"]
    if robots:
        a = appear(t, 1.1, 0.5)
        T(ax, W / 2, 440, "SUPPORTED ARMS", 15, MUTED, "semibold", ha="center", alpha=a)
        gap, cw = 30, 360
        x0 = W / 2 - (len(robots) * cw + (len(robots) - 1) * gap) / 2
        for i, (name, dof) in enumerate(robots):
            b = appear(t, 1.25 + 0.18 * i, 0.45)
            x = x0 + i * (cw + gap)
            rbox(ax, x, 350 - 14 * (1 - b), cw, 64, 14, fc=SURF2, alpha=b, z=3)
            T(ax, x + 26, 382 - 14 * (1 - b), name, 23, TXT, "semibold", va="center_baseline", alpha=b)
            T(ax, x + cw - 26, 382 - 14 * (1 - b), f"{dof}-DOF", 20, MUTED, ha="right", va="center_baseline", alpha=b)
    a = appear(t, 1.9, 0.5)
    T(ax, W / 2, 232, "STP 2026 · Team 06", 28, TXT2, "semibold", ha="center", alpha=a)
    T(ax, W / 2, 190, "All results in simulation (PyBullet)", 18, MUTED, ha="center", alpha=a)
    fade_overlay(ax, ease_io(prog(t, CLIPS["end"][0] - 0.9, 0.9)))


def draw_ik_drop(fig, t, F):
    """Scene 3 (14-24 s): position error 128 mm -> 23.1 mm -> 0.0003 mm on a log axis, counter on the right."""
    ax = canvas(fig)
    chrome(ax, t, "LEVEL 1 · SOLVE", C_L1)
    heading(ax, t, "A network guesses. A solver makes it exact.",
            "Mean position error on unseen poses · xArm6 · log scale")
    vals = [F["plan_mm"], F["l1_net_mm"], F["l1_lm_mm"]]
    names = [("Original plan", "network alone"),
             ("Our network", f"alone · {F['l1_net_params'] / 1e6:.2f} M parameters"),
             ("Network + LM solver", "refined to the exact answer")]
    cols = ["#5b6573", mix(C_L1, BG, 0.35), C_L1]
    grow = [(1.0, 1.2), (2.5, 1.2), (4.3, 1.6)]

    x0, x1, y0, y1, d0, d1 = 230, 1080, 200, 710, -4, 3
    Y = lambda v: y0 + (math.log10(v) - d0) / (d1 - d0) * (y1 - y0)            # noqa: E731
    g = appear(t, 0.5, 0.6)
    for d in range(d0, d1 + 1):
        y = Y(10.0 ** d)
        one = d == 0
        ax.plot([x0, x1], [y, y], color=TXT2 if one else BORDER, lw=1.6 if one else 1, ls=(0, (6, 5)) if one else "-",
                alpha=g * (0.7 if one else 1), zorder=1)
        T(ax, x0 - 18, y, f"{10.0 ** d:g}", 16, TXT2 if one else MUTED, ha="right", va="center_baseline", alpha=g)
    T(ax, x0 - 18, y1 + 34, "mm", 16, MUTED, ha="right", alpha=g)
    T(ax, x1 + 10, Y(1) + 8, "1 mm", 17, TXT2, "semibold", alpha=g)

    centers, bw = [380, 655, 930], 180
    tops = []
    for i, (v, (n1, n2), col, (gs, gd)) in enumerate(zip(vals, names, cols, grow)):
        k = ease_out(prog(t, gs, gd)) if i < 2 else ease_io(prog(t, gs, gd))
        la = appear(t, gs - 0.3, 0.4)
        cx = centers[i]
        T(ax, cx, y0 - 44, n1, 21, TXT, "semibold", ha="center", alpha=la)
        T(ax, cx, y0 - 74, n2, 16, MUTED, ha="center", alpha=la)
        if k <= 0:
            tops.append(None)
            continue
        top = lerp(y0, Y(v), k)
        p = rbox(ax, cx - bw / 2, y0 - 14, bw, top - y0 + 14, 10, fc=col, ec="none", z=2)
        p.set_clip_path(Rectangle((0, y0), W, H, transform=ax.transData))
        if i == 2:
            glow(ax, cx, top, 150, C_L1, 0.55 * k, z=1.5)
        shown = v if k >= 1 else (lerp(0, v, k) if i == 0 else loglerp(vals[i - 1], v, k))
        T(ax, cx, top + 18, f"{fmt_mm(shown)} mm", 26, TXT, "semibold", ha="center", alpha=min(1, k * 3))
        tops.append(top)

    ring = prog(t, 5.9, 0.9)                                                   # arrival burst on the exact answer
    if 0 < ring < 1:
        ax.add_patch(Circle((centers[2], Y(vals[2])), 20 + 110 * ease_out(ring), fc="none", ec=C_L1, lw=3,
                            alpha=1 - ring, zorder=3))
    for i, (ts, label) in enumerate([(3.8, f"{math.floor(vals[0] / vals[1] * 10) / 10:g}× better"), (6.1, "LM solver")]):
        a = appear(t, ts, 0.5)
        if a <= 0 or tops[i] is None or tops[i + 1] is None:
            continue
        p0 = (centers[i] + 40, tops[i] + 62)
        p1 = (centers[i + 1] - 30, tops[i + 1] + 62)
        rad = -0.28
        ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>,head_length=10,head_width=6",
                                     color=TXT2, lw=1.8, alpha=a, zorder=3))
        mx = (p0[0] + p1[0]) / 2 + 0.5 * rad * (p1[1] - p0[1])
        my = (p0[1] + p1[1]) / 2 - 0.5 * rad * (p1[0] - p0[0])
        t_ = T(ax, mx, my, label, 18, TXT, "semibold", ha="center", va="center_baseline", alpha=a, z=6)
        wl = text_w(ax, t_)
        rbox(ax, mx - wl / 2 - 16, my - 19, wl + 32, 38, 19, fc=SURF2, alpha=a, z=5)

    # right panel: one big counter that follows the story
    px = 1250
    a = appear(t, 0.9, 0.5)
    T(ax, px, 690, "MEAN POSITION ERROR", 17, MUTED, "semibold", alpha=a)
    if t < grow[1][0]:
        v, stage = lerp(0, vals[0], ease_out(prog(t, *grow[0]))), 0
    elif t < grow[2][0]:
        v, stage = loglerp(vals[0], vals[1], ease_out(prog(t, *grow[1]))), 1
    else:
        v, stage = loglerp(vals[1], vals[2], ease_io(prog(t, *grow[2]))), 2
    T(ax, px + 470, 560, fmt_mm(v), 104, TXT, "semibold", ha="right", alpha=a)
    T(ax, px + 488, 560, "mm", 38, TXT2, alpha=a)
    stage_t = grow[stage][0]
    b = appear(t, stage_t, 0.4) if stage else a
    T(ax, px, 500, f"{names[stage][0]}", 22, TXT2, alpha=b)
    if stage == 2:
        glow(ax, px + 330, 595, 360, C_L1, 0.18 * ease_out(prog(t, *grow[2])), z=0.4)
    rows = [f"{fmt_pct(F['l1_lm_within_1mm_1deg'])} within 1 mm & 1°",
            f"{fmt_pct(F['l1_lm_self_coll'])} self-collision",
            f"{fmt_pct(F['l1_lm_limits'])} joint-limit violations"]
    for i, r in enumerate(rows):
        c = appear(t, 6.5 + 0.3 * i, 0.45)
        sc = ease_back(prog(t, 6.5 + 0.3 * i, 0.45))
        y = 400 - 64 * i
        if c > 0:
            check(ax, px + 18, y + 9, 17 * max(sc, 0.01), alpha=c)
        T(ax, px + 52 - 12 * (1 - c), y, r, 25, TXT, alpha=c)


def _round_pcts(pcts):
    """Whole dots that add up to 100 (largest remainder)."""
    fl = [math.floor(p) for p in pcts]
    for i in sorted(range(len(pcts)), key=lambda i: pcts[i] - fl[i], reverse=True)[: 100 - sum(fl)]:
        fl[i] += 1
    return fl


def draw_waffle(fig, t, F):
    """Scene 6 (50-62 s): 100 dots = 100 % of the episodes; straight path morphs into the DRL agent."""
    ax = canvas(fig)
    chrome(ax, t, "LEVEL 2 · STEER", C_L2)
    sc = F["l2_eval"]["combined"]
    heading(ax, t, "Everything at once: obstacles appear, drift, joints slip",
            f"{F['l2_n']} unseen episodes · identical seeds for every controller · each dot = 1 % of episodes")
    status = [("Reached the target", C_REACHED, "success_pct"), ("Collision", C_COLL, "collision_pct"),
              ("Stuck (timed out)", C_STUCK, "timeout_pct")]
    order = [0, 2, 1]                                                          # grid order: reached, stuck, collision last
    def dots(ctrl):                                                            # noqa: E306
        n = _round_pcts([sc[ctrl][k] for _, _, k in status])
        return [s for s in order for _ in range(n[s])]
    before, after = dots("baseline"), dots("policy")
    changed = [i for i in range(100) if before[i] != after[i]]
    t_morph = 4.3

    gx, gy, step, r = 170, 715, 56, 20
    for i in range(100):
        row, col = divmod(i, 10)
        cx, cy = gx + col * step, gy - row * step
        p_in = prog(t, 0.7 + (row + col) * 0.055, 0.45)
        if p_in <= 0:
            continue
        rad = r * ease_back(p_in)
        c = to_rgb(status[before[i]][1])
        if i in changed:
            k = prog(t, t_morph + changed.index(i) * 0.05, 0.35)
            if k > 0:
                c = mix(status[before[i]][1], status[after[i]][1], ease_io(k))
                rad *= 1 - 0.45 * math.sin(math.pi * k)                        # little squeeze while it flips
        ax.add_patch(Circle((cx, cy), max(rad, 0.1), fc=c, ec="none", zorder=3))
        if after[i] == 1 and t > 7.2:                                          # the remaining collisions pulse
            ph = ((t - 7.2) % 1.4) / 1.4
            ax.add_patch(Circle((cx, cy), r + 26 * ease_out(ph), fc="none", ec=C_COLL, lw=2.5, alpha=0.8 * (1 - ph), zorder=2))

    # right: controller name + three status rows with live numbers
    rx = 860
    k_name = ease_io(prog(t, t_morph - 0.4, 0.6))
    for name, col, a, dy in (("Straight path", C_STRAIGHT, (1 - k_name) * appear(t, 0.8), 30 * k_name),
                             ("DRL agent", C_AGENT, k_name, -30 * (1 - k_name))):
        if a > 0.01:
            rbox(ax, rx, 692 + dy, 12, 58, 6, fc=col, ec="none", alpha=a, z=3)
            T(ax, rx + 34, 700 + dy, name, 50, TXT, "semibold", alpha=a)
    T(ax, rx + 34, 786, "CONTROLLER", 14, MUTED, "semibold", alpha=appear(t, 0.8))
    k_cnt = ease_out(prog(t, 1.0, 1.5))
    k_mrp = ease_io(prog(t, t_morph, 0.05 * len(changed) + 0.35))
    for j, (label, col, key) in enumerate(status):
        y = 560 - 92 * j
        a = appear(t, 1.0 + 0.15 * j)
        ax.add_patch(Circle((rx + 16, y + 12), 13, fc=col, ec="none", alpha=a, zorder=3))
        T(ax, rx + 48, y, label, 28, TXT2, alpha=a)
        v = lerp(sc["baseline"][key] * k_cnt, sc["policy"][key], k_mrp)
        T(ax, 1720, y - 4, f"{v:.0f} %", 52, TXT, "semibold", ha="right", alpha=a)
        ax.plot([rx, 1720], [y - 36, y - 36], color=BORDER, lw=1, alpha=a, zorder=1)

    for i, (label, key) in enumerate((("Collisions", "collision_pct"), ("Success", "success_pct"))):
        a = appear(t, 7.4 + 0.35 * i, 0.5)
        y = 250 - 70 * i
        T(ax, rx - 16 * (1 - a), y, f"{label}  {sc['baseline'][key]:.0f} %  →  {sc['policy'][key]:.0f} %", 38, TXT,
          "semibold", alpha=a)
    a = appear(t, 8.6, 0.5)
    T(ax, rx, 125, f"Classic potential field on the same seeds: {fmt_pct(sc['apf']['success_pct'])} reached · "
      f"{fmt_pct(sc['apf']['collision_pct'])} collisions", 19, MUTED, alpha=a)


def draw_tiny_agent(fig, t, F):
    """Scene 7 (62-72 s): the 2 x 64 network lights up; 11,078 parameters, 17.5 µs, no GPU."""
    ax = canvas(fig)
    chrome(ax, t, "LEVEL 2 · STEER", C_L2)
    heading(ax, t, "Small enough to fly", "The re-routing agent is a 2 × 64 network – it decides in microseconds")
    o, h1, h2, a_ = F["l2_layers"] or (None, 64, 64, None)
    layers = [(o, 10, f"{o} inputs" if o else "inputs"), (h1, 12, str(h1)), (h2, 12, str(h2)),
              (a_, a_ or 6, f"{a_} joint corrections" if a_ else "outputs")]
    xs = [230, 480, 730, 960]
    pos = []
    for (n, nd, _), x in zip(layers, xs):
        ys = np.linspace(700, 250, nd + (1 if n and n > nd else 0))
        if n and n > nd:                                                       # gap for the ellipsis
            ys = np.delete(ys, nd // 2)
        pos.append([(x, y) for y in ys])
    # edges reveal layer by layer, then signals flow through
    for li in range(3):
        k = ease_out(prog(t, 0.7 + 0.45 * li, 0.6))
        if k <= 0:
            continue
        segs = [[p, (p[0] + (q[0] - p[0]) * k, p[1] + (q[1] - p[1]) * k)] for p in pos[li] for q in pos[li + 1]]
        ax.add_collection(LineCollection(segs, colors=[C_AGENT], linewidths=1, alpha=0.16, zorder=1))
    if t > 2.6:
        for kk in range(int((t - 2.6) / 0.16) + 1):
            ts = 2.6 + kk * 0.16
            ph = (t - ts) / 0.75
            if not 0 <= ph < 1:
                continue
            rg = np.random.default_rng(kk)
            path = [pos[li][rg.integers(len(pos[li]))] for li in range(4)]
            seg = min(int(ph * 3), 2)
            f = ph * 3 - seg
            for s in range(seg + 1):
                (xa, ya), (xb, yb) = path[s], path[s + 1]
                e = 1 if s < seg else f
                ax.plot([xa, xa + (xb - xa) * e], [ya, ya + (yb - ya) * e], color=C_AGENT, lw=2,
                        alpha=0.55 * (1 - ph * 0.6), zorder=2)
            (xa, ya), (xb, yb) = path[seg], path[seg + 1]
            hx, hy = xa + (xb - xa) * f, ya + (yb - ya) * f
            glow(ax, hx, hy, 34, C_AGENT, 0.8, z=2.5)
            ax.add_patch(Circle((hx, hy), 4.5, fc="white", ec="none", zorder=6))
    for li, ((n, nd, label), pts) in enumerate(zip(layers, pos)):
        for j, (x, y) in enumerate(pts):
            k = ease_back(prog(t, 0.5 + 0.45 * li + 0.02 * j, 0.4))
            if k > 0:
                ax.add_patch(Circle((x, y), 11 * k, fc=SURF2, ec=C_AGENT, lw=2, zorder=4))
        a = appear(t, 0.6 + 0.45 * li)
        if n and n > nd:
            for d in (-14, 0, 14):
                ax.add_patch(Circle((pts[0][0], (pts[nd // 2 - 1][1] + pts[nd // 2][1]) / 2 + d), 2.6, fc=MUTED, alpha=a, zorder=4))
        T(ax, pts[0][0], 196, label, 19, TXT2, ha="center", alpha=a)

    # right: counters
    px = 1180
    a = appear(t, 0.9)
    T(ax, px, 690, f"{round(F['l2_params'] * ease_out(prog(t, 0.9, 1.8))):,}", 96, TXT, "semibold", alpha=a)
    T(ax, px + 4, 640, "parameters", 26, TXT2, alpha=a)
    a = appear(t, 3.0)
    T(ax, px, 505, f"{F['l2_us'] * ease_out(prog(t, 3.0, 1.4)):.1f} µs", 96, TXT, "semibold", alpha=a)
    T(ax, px + 4, 440, "per decision · NumPy · one CPU core", 26, TXT2, alpha=a)
    a = appear(t, 4.6)
    T(ax, px + 4, 378 - 10 * (1 - a), f"≈ {round(1e6 / F['l2_us'], -3):,.0f} decisions per second", 26, TXT2, "semibold", alpha=a)

    def chip(cx, cy, label, a, crossed=0.0, s=0.72):
        if a <= 0:
            return
        rbox(ax, cx - 34 * s, cy - 34 * s, 68 * s, 68 * s, 10 * s, fc=SURF2, ec=TXT2, lw=2, alpha=a, z=3)
        rbox(ax, cx - 16 * s, cy - 16 * s, 32 * s, 32 * s, 5 * s, fc="none", ec=TXT2, lw=1.6, alpha=a, z=3)
        for d in (-20 * s, 0, 20 * s):
            for (x0_, y0_, x1_, y1_) in ((cx + d, cy + 34 * s, cx + d, cy + 46 * s), (cx + d, cy - 34 * s, cx + d, cy - 46 * s),
                                         (cx + 34 * s, cy + d, cx + 46 * s, cy + d), (cx - 34 * s, cy + d, cx - 46 * s, cy + d)):
                ax.plot([x0_, x1_], [y0_, y1_], color=TXT2, lw=2, alpha=a, zorder=3, solid_capstyle="round")
        if crossed > 0:                                                        # strike-through draws across the chip
            xs_, ys_ = [cx - 54 * s, cx + (-54 + 108 * crossed) * s], [cy + 54 * s, cy + (54 - 108 * crossed) * s]
            ax.plot(xs_, ys_, color=BG, lw=9, zorder=4, solid_capstyle="round")
            ax.plot(xs_, ys_, color=TXT, lw=3.5, zorder=5, solid_capstyle="round")
        T(ax, cx + 66, cy, label, 26, TXT, "semibold", va="center_baseline", alpha=a)

    chip(px + 36, 262, "Runs on one CPU core", appear(t, 5.4))
    glow(ax, px + 36, 262, 90, C_AGENT, 0.3 * appear(t, 5.8), z=0.6)
    chip(px + 36, 158, "No GPU needed on board", appear(t, 6.3), crossed=ease_out(prog(t, 6.7, 0.45)))


def draw_mission_card(fig, t, F):
    """Scene 8 (72-84 s): target locked - 0.002 mm, then the four mission stats."""
    ax = canvas(fig)
    chrome(ax, t, "MISSION · SOLVE + STEER", C_MISSION)
    m = F["m_dyn"]
    heading(ax, t, "Solve, then steer – end to end",
            f"{m['n']} dynamic missions: drifting + sudden obstacles + joint slip · xArm6")
    # reticle closes in on the target, then the number lands
    cx, cy = 560, 600
    k = ease_io(prog(t, 0.6, 1.3))
    for j, base in enumerate((150, 105, 64)):
        rr = lerp(base, 14 + 7 * j, k)
        ax.add_patch(Circle((cx, cy), rr, fc="none", ec=C_MISSION if k >= 1 else TXT2, lw=2.2 - 0.4 * j,
                            alpha=appear(t, 0.4 + 0.1 * j) * (0.9 - 0.2 * j), zorder=3))
    for ang in range(0, 360, 90):
        ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        r0 = lerp(170, 36, k)
        ax.plot([cx + ca * r0, cx + ca * (r0 + 26)], [cy + sa * r0, cy + sa * (r0 + 26)], color=TXT2, lw=2.2,
                alpha=appear(t, 0.4), zorder=3, solid_capstyle="round")
    lock = prog(t, 1.9, 0.7)
    if lock > 0:
        ax.add_patch(Circle((cx, cy), 6, fc=C_MISSION, ec="none", zorder=4))
        glow(ax, cx, cy, 90 + 60 * ease_out(lock), C_MISSION, 0.7 * (1 - 0.5 * lock), z=1)
        if lock < 1:
            ax.add_patch(Circle((cx, cy), 20 + 160 * ease_out(lock), fc="none", ec=C_MISSION, lw=3, alpha=1 - lock, zorder=3))
    a = appear(t, 2.0, 0.55)
    sc = 1.06 - 0.06 * a
    T(ax, 760, 745, "MEDIAN FINAL GRIPPER ERROR", 18, MUTED, "semibold", alpha=appear(t, 1.6))
    T(ax, 752, 545, f"{m['median_final_mm']:.3f}", 150 * sc, TXT, "semibold", alpha=a)
    T(ax, 1330, 545, "mm", 54, TXT2, alpha=a)

    pol = m["controllers"]["policy"]
    tiles = [(m["ik_success_pct"], "IK solved", "Level 1 · Solve", C_L1),
             (m["predicted_conflict_pct"], "straight paths blocked", "path check", MUTED),
             (pol["success_pct"], "reached the target", "Level 2 · Steer", C_L2),
             (pol["self_collision_pct"], "self-collisions", "every controller", C_MISSION)]
    tw, gap = 375, 32
    x0 = (W - 4 * tw - 3 * gap) / 2
    for i, (v, label, tag, col) in enumerate(tiles):
        ts = 2.9 + 0.22 * i
        a = appear(t, ts, 0.5)
        if a <= 0:
            continue
        x, y = x0 + i * (tw + gap), 150 - 30 * (1 - a)
        rbox(ax, x, y, tw, 196, 16, fc=SURF, alpha=a, z=2)
        rbox(ax, x, y + 190, tw, 6, 3, fc=col, ec="none", alpha=a, z=3)
        T(ax, x + 30, y + 150, tag.upper(), 14, MUTED, "semibold", alpha=a)
        T(ax, x + 28, y + 64, f"{v * ease_out(prog(t, ts, 1.2)):.0f} %", 62, TXT, "semibold", alpha=a)
        T(ax, x + 30, y + 26, label, 21, TXT2, alpha=a)
    glow(ax, cx, cy, 300, C_MISSION, 0.10 * appear(t, 2.2), z=0.4)


def draw_curves(fig, t, F):
    """Technical cut: Level 2 training curves (success / collision) drawn left to right, warm-up shaded."""
    ax = canvas(fig)
    chrome(ax, t, "LEVEL 2 · TRAINING", C_L2)
    steps_m = F["l2_steps"] / 1e6
    heading(ax, t, f"Learning to steer: {steps_m:.0f} M steps in {F['l2_minutes']:.0f} min",
            f"PPO · {F['l2_envs']} simulated arms on {F['l2_workers']} CPU workers · GPU updates · smoothed over 5 updates")
    h = F["l2_hist"]
    xs = np.array([r["steps"] for r in h]) / 1e6

    def smooth(key):                                                           # same as the Studio (level2.js)
        v = np.array([r[key] for r in h], float)
        c = np.cumsum(np.insert(v, 0, 0))
        idx = np.arange(len(v))
        lo = np.maximum(0, idx - 4)
        return (c[idx + 1] - c[lo]) / (idx + 1 - lo), v
    warm = xs[min(2, len(xs) - 1)]
    x0, x1 = 210, 1560
    X = lambda s: x0 + s / 20.0 * (x1 - x0) if xs[-1] <= 20.5 else x0 + s / xs[-1] * (x1 - x0)   # noqa: E731
    xmax_axis = 20.0 if xs[-1] <= 20.5 else xs[-1]
    panels = [("Success rate", "success_pct", C_REACHED, 470, 730), ("Collision rate", "collision_pct", C_COLL, 150, 410)]
    k = ease_io(prog(t, 0.9, 8.0))
    xcut = k * xs[-1]
    for pi, (title, key, col, y0, y1) in enumerate(panels):
        a = appear(t, 0.45 + 0.15 * pi, 0.5)
        Y = lambda v: y0 + v / 100 * (y1 - y0)                                 # noqa: E731
        for gv in (0, 50, 100):
            ax.plot([x0, x1], [Y(gv), Y(gv)], color=BORDER, lw=1, alpha=a, zorder=1)
            T(ax, x0 - 14, Y(gv), f"{gv} %", 15, MUTED, ha="right", va="center_baseline", alpha=a)
        ax.add_patch(Rectangle((x0, y0), max(X(warm) - x0, 3), y1 - y0, fc=MUTED, ec="none", alpha=0.25 * a, zorder=1))
        if pi == 1:
            T(ax, X(warm) + 10, y1 - 26, "← shaded: warm-up – the first 3 updates count only short episodes", 15, MUTED, alpha=a)
        ax.add_patch(Circle((x0 + 8, y1 + 30), 8, fc=col, ec="none", alpha=a, zorder=3))
        T(ax, x0 + 26, y1 + 22, title, 22, TXT, "semibold", alpha=a)
        ys, raw = smooth(key)
        if k > 0:
            n = int(np.searchsorted(xs, xcut, side="right"))
            n = max(n, 1)
            px = [X(s) for s in xs[:n]]
            hv = np.interp(xcut, xs, ys)
            ax.plot(px, [Y(v) for v in raw[:n]], color=col, lw=1, alpha=0.13, zorder=2)
            glow_line(ax, px + [X(xcut)], [Y(v) for v in ys[:n]] + [Y(hv)], col, lw=2.6)
            glow(ax, X(xcut), Y(hv), 40, col, 0.7, z=4.5)
            ax.add_patch(Circle((X(xcut), Y(hv)), 6.5, fc="white", ec=col, lw=2.5, zorder=6))
    a = appear(t, 0.6, 0.5)
    for s in range(0, int(xmax_axis) + 1, 5):
        T(ax, X(s), 112, f"{s} M", 15, MUTED, ha="center", alpha=a)
    T(ax, 1800, 878, f"{xcut:.1f} M", 46, TXT, "semibold", ha="right", alpha=appear(t, 0.9))
    T(ax, 1800, 840, "environment steps", 18, MUTED, ha="right", alpha=appear(t, 0.9))


def draw_outcomes(fig, t, F):
    """Technical cut: mission outcomes, three controllers x two scenarios, 100 % stacked bars."""
    ax = canvas(fig)
    chrome(ax, t, "MISSION · SOLVE + STEER", C_MISSION)
    heading(ax, t, "Same missions, three controllers",
            f"{F['m_dyn']['n']} missions per scenario · identical seeds · IK solved in {F['m_dyn']['ik_success_pct']:.0f} %")
    ctrls = [("policy", "DRL agent", C_AGENT), ("apf", "Potential field", C_APF), ("baseline", "Straight path", C_STRAIGHT)]
    segs = [("success_pct", "reached", C_REACHED), ("collision_pct", "collision", C_COLL), ("timeout_pct", "stuck", C_STUCK)]
    a = appear(t, 0.5)
    for i, (_, label, col) in enumerate(segs):                                 # legend
        lx = 620 + i * 210
        ax.add_patch(Circle((lx, 760), 9, fc=col, ec="none", alpha=a, zorder=3))
        T(ax, lx + 20, 760, label, 20, TXT2, va="center_baseline", alpha=a)
    groups = [("Dynamic · drifting + sudden obstacles + joint slip", F["m_dyn"]),
              ("Static + sudden obstacle + joint slip", F["m_comb"])]
    bx0, bx1, bh = 600, 1500, 46
    for gi, (gname, m) in enumerate(groups):
        gy = 680 - gi * 290
        ga = appear(t, 0.7 + 1.5 * gi)
        T(ax, 120, gy, gname, 22, TXT, "semibold", alpha=ga)
        for ri, (ck, cname, ccol) in enumerate(ctrls):
            y = gy - 70 - ri * 66
            ts = 0.9 + 1.5 * gi + 0.25 * ri
            ra = appear(t, ts)
            rbox(ax, 120, y - 9, 18, 18, 4, fc=ccol, ec="none", alpha=ra, z=3)
            T(ax, 152, y, cname, 22, TXT2, va="center_baseline", alpha=ra)
            c = m["controllers"][ck]
            k = ease_out(prog(t, ts, 1.0))
            acc = 0.0
            for key, _, scol in segs:
                v = c[key]
                left = bx0 + acc / 100 * (bx1 - bx0)
                wpx = min(v / 100 * (bx1 - bx0), max(0.0, bx0 + k * (bx1 - bx0) - left))
                if wpx > 0.5:
                    ax.add_patch(Rectangle((left, y - bh / 2), wpx, bh, fc=scol, ec=BG, lw=1.5, zorder=3))
                    if key == "success_pct" and wpx > 90:
                        T(ax, left + 16, y, f"{v:.0f} %", 20, "white", "semibold", va="center_baseline", z=5)
                acc += v
            note = f"{c['collision_pct']:.0f} % collision · {c['timeout_pct']:.0f} % stuck"
            T(ax, bx1 + 20, y, note, 19, TXT2, va="center_baseline", alpha=appear(t, ts + 0.8))


CLIPS = {   # name: (seconds, draw, hero time for --stills, storyboard use)
    "title":        (6.0,  draw_title,        4.5, "Video A scene 1"),
    "ik_drop":      (10.0, draw_ik_drop,      8.5, "Video A scene 3"),
    "waffle":       (12.0, draw_waffle,       9.5, "Video A scene 6"),
    "tiny_agent":   (10.0, draw_tiny_agent,   8.0, "Video A scene 7"),
    "mission_card": (8.0,  draw_mission_card, 6.0, "Video A scene 8"),
    "end":          (8.0,  draw_end,          3.5, "Video A scene 9"),
    "curves":       (12.0, draw_curves,      10.5, "Video B training curves"),
    "outcomes":     (9.0,  draw_outcomes,     7.5, "Video B / deck slide 12"),
}


# ============================================================================================= render
def _fig():
    _setup_fonts()
    return plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)


def render_clip(name, F, out_dir):
    dur, draw, _, _ = CLIPS[name]
    fig = _fig()
    path = out_dir / f"anim_{name}.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
           str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for i in range(round(dur * FPS)):
        draw(fig, i / FPS, F)
        fig.canvas.draw()
        proc.stdin.write(fig.canvas.buffer_rgba())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError(f"ffmpeg failed for {name}")
    plt.close(fig)
    return f"{path.relative_to(ROOT)}  ({dur:.0f} s, {time.time() - t0:.0f} s to render)"


def render_still(name, F, out_dir, at=None):
    dur, draw, hero, _ = CLIPS[name]
    fig = _fig()
    draw(fig, hero if at is None else at, F)
    path = out_dir / f"anim_{name}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return str(path.relative_to(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="*", help=f"subset of: {', '.join(CLIPS)}")
    ap.add_argument("--stills", action="store_true", help="one PNG per clip at its hero time instead of video")
    ap.add_argument("--at", type=float, help="with --stills: time in seconds instead of the hero time")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("-j", "--jobs", type=int, default=3, help="parallel clips (each ffmpeg encoder needs ~0.7 GB RAM)")
    args = ap.parse_args()
    if args.list:
        for n, (d, _, _, use) in CLIPS.items():
            print(f"{n:13s} {d:5.1f} s   {use}")
        return
    names = args.clips or list(CLIPS)
    bad = [n for n in names if n not in CLIPS]
    if bad:
        ap.error(f"unknown clip(s) {bad}; choose from {list(CLIPS)}")
    F = load_facts()
    _setup_fonts()                                       # once, before the workers fork
    out_dir = OUT / "stills" if args.stills else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    job = (lambda n: render_still(n, F, out_dir, args.at)) if args.stills else (lambda n: render_clip(n, F, out_dir))
    if len(names) == 1 or args.jobs == 1:
        results = [job(n) for n in names]
    else:
        with mp.get_context("fork").Pool(min(args.jobs, len(names))) as pool:
            results = pool.map(_run, [(args.stills, n, F, out_dir, args.at) for n in names])
    for r in results:
        print(r)


def _run(a):
    stills, n, F, out_dir, at = a
    return render_still(n, F, out_dir, at) if stills else render_clip(n, F, out_dir)


if __name__ == "__main__":
    main()
