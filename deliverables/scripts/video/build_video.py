#!/usr/bin/env python
"""Assemble the five-minute video from the narration timing and the renderer scenes.

Steps
  1. cues: every scene's events are placed at the start of the sentence that describes them
  2. render each segment (renderer "video" variant, no on-screen captions) -> $NIK_WORK/vseg/
  3. concatenate (scenes fade through the background colour, so plain cuts join them)
  4. audio: narration sentences at their times + an ambient music bed synthesised here (no third-party
     audio), ducked under the voice
  5. subtitles: one cue per sentence (long ones split), written as .srt and burned in (ASS style)
  6. encode H.264 / AAC, 1920x1080, 30 fps -> deliverables/video/Neural_IK_STP_Team06_5min.mp4
  7. write scene_timeline.csv and voiceover_script.md

    python deliverables/scripts/video/build_video.py            # renders what is missing, then assembles
    python deliverables/scripts/video/build_video.py --only-audio
"""
import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import REPO, SCRIPTS, VIDEO, WORK  # noqa: E402

SR = 48000
FPS = 30
OUT = VIDEO / "Neural_IK_STP_Team06_5min.mp4"
CLIP_OF = {"robot_intro": "robot_intro.mp4", "forward_kinematics": "forward_kinematics.mp4",
           "inverse_kinematics": "inverse_kinematics.mp4", "dataset_generation": "dataset_generation.mp4",
           "neural_architecture": "neural_architecture.mp4", "collision_awareness": "collision_awareness.mp4",
           "training_validation": "training_validation.mp4", "inference_demo": "inference_demo.mp4",
           "method_comparison": "method_comparison.mp4", "level2_ppo": "level2_ppo_rerouting.mp4"}


def cues(seg):
    S = [s["start"] for s in seg["sentences"]]
    E = [s["end"] for s in seg["sentences"]]
    D = seg["duration"]
    sc = seg["scene"]
    if sc == "opening":
        return {"move0": 0.8, "move1": 9.0, "title": S[0] + 0.8, "sub": S[1] - 0.3}
    if sc == "objectives":
        return {"l1": S[1] - 0.3, "l2": S[2] - 0.3, "robot": 0.0}
    if sc == "forward_kinematics":
        j0 = S[0] + 0.6
        step = max(1.2, (S[1] + 0.6 * (E[1] - S[1]) - j0) / 6)
        pose = j0 + 6 * step + 0.2
        return {"j0": j0, "step": step, "pose": pose, "eq": min(D - 2.5, pose + 1.4)}
    if sc == "inverse_kinematics":
        b0 = S[1] - 0.2
        return {"target": 0.5, "b0": b0, "bStep": max(0.8, (S[2] - 0.4 - b0) / 4), "seed": S[2] - 0.2,
                "move": S[2] + 1.0, "eq": min(D - 3.0, S[2] + 3.8)}
    if sc == "dataset_generation":
        slow0 = S[0] + 0.3
        return {"slow0": slow0, "slowN": 6, "slowDt": (S[2] - 0.2 - slow0) / 6, "fast1": S[2], "fast2": S[3] - 0.6,
                "cloud": S[3] - 0.4, "stats": S[3] + 1.4}
    if sc == "neural_architecture":
        return {"inputs": S[0] + 0.2, "h1": S[1], "h2": S[1] + 1.3, "h3": S[1] + 2.6, "out": S[1] + 3.8,
                "decode": S[2] - 0.2, "robot": S[2] + 1.0, "circle": S[3] - 0.3, "note": D}
    if sc == "collision_awareness":
        m0 = S[0] + 0.6
        return {"panels": 0.6, "move0": m0, "move1": max(m0 + 6.0, E[0] - 0.4), "loss": S[1] - 0.2, "note": S[1] + 3.0}
    if sc == "training_validation":
        return {"charts": 0.6, "ep0": S[0] + 0.4, "ep1": S[1] + 2.5, "note": S[1]}
    if sc == "inference_demo":
        return {"c0": 0.9, "per": (D - 1.8) / 3}
    if sc == "method_comparison":
        ex0, agg = S[0] + 0.4, S[2] - 0.5
        return {"ex0": ex0, "per": (agg - ex0) / 3, "agg": agg, "note": S[3] - 0.3}
    if sc == "workflow":
        s0 = S[0] + 0.3
        return {"s0": s0, "step": (E[0] - 1.2 - s0) / 5, "l2": E[0] - 0.9, "reg": S[1] - 0.2}
    if sc == "level2_ppo":
        ep0, res = S[2] - 0.6, S[3] - 0.3
        steps = json.load(open(WORK / "sim" / "level2.json"))["episodes"]["policy"]["steps"]
        sps = max(9.0, (steps + 4) / max(1.0, res - ep0 - 1.2))     # the agent arrives before the results appear
        return {"loop": 0.6, "ep0": ep0, "res": res, "sps": sps}
    if sc == "closing":
        return {"chips": 0.8, "thanks": S[1] - 0.4}
    raise ValueError(sc)


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{cmd[0]} failed:\n{r.stderr[-3000:]}")
    return r.stdout


def segment_key(scene, tj):
    """Changes whenever anything that affects the rendered frames changes: cues, the scene, the shared
    renderer code, or the simulation data the scene reads."""
    rend = SCRIPTS / "render"
    js = (rend / "js" / "scenes" / f"{scene}.js").read_text()
    shared = "".join(p.read_text() for p in sorted([*rend.glob("js/*.js"), *rend.glob("css/*.css"), rend / "stage.html"]))
    sims = sorted(set(re.findall(r"data\('([a-z_0-9]+)'", js)) | {"robot"})
    data = "".join((WORK / "sim" / f"{n}.json").read_text() for n in sims if (WORK / "sim" / f"{n}.json").exists())
    return hashlib.sha1((json.dumps(tj, sort_keys=True) + js + shared + data).encode()).hexdigest()[:10]


def render_segments(timing, force=False):
    (WORK / "timing").mkdir(parents=True, exist_ok=True)
    (WORK / "vseg").mkdir(parents=True, exist_ok=True)
    files = []
    for i, seg in enumerate(timing["segments"]):
        tj = {"duration": seg["duration"], "cues": cues(seg)}
        (WORK / "timing" / f"{seg['id']}.json").write_text(json.dumps(tj, indent=1))
        key = segment_key(seg["scene"], tj)
        out = WORK / "vseg" / f"{i:02d}_{seg['id']}_{key}.mp4"
        if force or not out.exists():
            print(f"render {seg['id']} ({seg['duration']:.1f} s)", flush=True)
            subprocess.run([sys.executable, str(SCRIPTS / "render" / "render_scene.py"), seg["scene"], "--variant", "video",
                            "--timing", f"/work/timing/{seg['id']}.json", "--out", str(out), "--crf", "14"], check=True)
        files.append(out)
    return files


# ------------------------------------------------------------------------------------------------ audio
def music_bed(dur, seed=7):
    """Calm ambient pad: Am9 - Fmaj7 - Cmaj7 - G6, 8 s per chord, soft bells, synthetic reverb."""
    from scipy.signal import fftconvolve, butter, sosfilt
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros((2, n))
    midi = lambda m: 440.0 * 2 ** ((m - 69) / 12)
    chords = [[45, 57, 60, 64, 67, 71], [41, 53, 57, 60, 64, 69], [36, 48, 52, 55, 59, 64], [43, 55, 59, 62, 64, 67]]
    L = 8.0
    for k in range(int(np.ceil(dur / L)) + 1):
        ch = chords[k % 4]
        a0 = k * L - 1.5
        i0, i1 = max(0, int(a0 * SR)), min(n, int((a0 + L + 3.0) * SR))
        if i0 >= i1:
            continue
        tt = t[i0:i1] - a0
        env = np.clip(tt / 2.5, 0, 1) * np.clip((L + 3.0 - tt) / 3.0, 0, 1)
        env = env * env * (3 - 2 * env)
        for j, m in enumerate(ch):
            f = midi(m)
            amp = 0.10 if j == 0 else 0.055
            for c, det in enumerate((-0.12, 0.12)):
                ph = rng.uniform(0, 2 * np.pi)
                w = np.sin(2 * np.pi * (f + det) * tt + ph) + 0.18 * np.sin(2 * np.pi * 2 * (f + det) * tt + ph)
                out[c, i0:i1] += amp * env * w * (1 + 0.15 * np.sin(2 * np.pi * 0.11 * tt + j))
        # soft bells: chord tones, two octaves up, every 2 s
        for b in range(4):
            tb = a0 + 1.5 + b * 2.0 + rng.uniform(-0.2, 0.2)
            m = ch[rng.integers(2, len(ch))] + 24
            j0 = int(tb * SR)
            if 0 <= j0 < n:
                j1 = min(n, j0 + int(3.0 * SR))
                tl = (np.arange(j1 - j0)) / SR
                bell = 0.035 * np.exp(-tl * 2.2) * np.sin(2 * np.pi * midi(m) * tl) * np.clip(tl / 0.01, 0, 1)
                pan = rng.uniform(0.3, 0.7)
                out[0, j0:j1] += bell * (1 - pan)
                out[1, j0:j1] += bell * pan
    sos = butter(2, 2400, "low", fs=SR, output="sos")
    out = sosfilt(sos, out, axis=1)
    ir_n = int(2.8 * SR)
    for c in range(2):
        ir = rng.standard_normal(ir_n) * np.exp(-np.arange(ir_n) / SR * 2.4)
        ir[0] = 0
        wet = fftconvolve(out[c], ir)[:n] * 0.012
        out[c] = 0.75 * out[c] + wet
    fade = np.clip(t / 3.0, 0, 1) * np.clip((dur - t) / 4.0, 0, 1)
    out *= fade
    return out / (np.max(np.abs(out)) + 1e-9)


def build_audio(timing, starts, total):
    import soundfile as sf
    from scipy.signal import resample_poly
    n = int(total * SR) + SR
    voice = np.zeros(n)
    speech = np.zeros(n, bool)
    for seg, t0 in zip(timing["segments"], starts):
        for s in seg["sentences"]:
            x, sr = sf.read(str(WORK / "voice" / s["wav"]), dtype="float64")
            if x.ndim > 1:
                x = x.mean(1)
            x = resample_poly(x, SR // 1000 * 1, sr // 1000) if sr != SR else x
            rms = np.sqrt(np.mean(x[np.abs(x) > 0.01] ** 2)) if np.any(np.abs(x) > 0.01) else 1.0
            x = x * (10 ** (-19 / 20) / rms)                        # every sentence at the same loudness
            i = int((t0 + s["start"]) * SR)
            voice[i:i + len(x)] += x[: n - i]
            speech[i:i + len(x)] = True
    # ducking envelope: music -9 dB under speech, smoothed
    duck = np.where(speech, 10 ** (-9 / 20), 1.0)
    k = int(0.35 * SR)                                        # moving average via cumulative sum (O(n))
    cs = np.concatenate([[0.0], np.cumsum(duck)])
    lo = np.clip(np.arange(n) - k // 2, 0, n)
    hi = np.clip(np.arange(n) + k // 2, 0, n)
    duck = (cs[hi] - cs[lo]) / np.maximum(1, hi - lo)
    bed = music_bed(total)                                     # fades out exactly when the picture ends
    music = np.zeros((2, n))
    music[:, :bed.shape[1]] = bed
    music = music * (10 ** (-25 / 20)) * duck
    mix = np.stack([voice + music[0], voice + music[1]])
    peak = np.max(np.abs(mix))
    if peak > 0.89:
        mix *= 0.89 / peak
    path = WORK / "video" / "audio.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), mix.T, SR, subtype="PCM_16")
    return path


# ------------------------------------------------------------------------------------------------ subtitles
def split_sentence(text, max_chars=112):
    """Split a long sentence into cues at punctuation (or the middle space), each <= max_chars."""
    if len(text) <= max_chars:
        return [text]
    best = None
    for i, ch in enumerate(text):
        if ch in ";:,." and 0 < i < len(text) - 1 and text[i + 1] == " ":
            score = abs(i - len(text) / 2)
            if best is None or score < best[0]:
                best = (score, i + 1)
    if best is None:
        mid = text.rfind(" ", 0, len(text) // 2 + 10)
        best = (0, mid)
    a, b = text[: best[1]].strip(), text[best[1]:].strip()
    return split_sentence(a, max_chars) + split_sentence(b, max_chars)


def subtitles(timing, starts):
    cues_ = []
    for seg, t0 in zip(timing["segments"], starts):
        for s in seg["sentences"]:
            parts = split_sentence(s["text"])
            a, b = t0 + s["start"], t0 + s["end"]
            total = sum(len(p) for p in parts)
            for p in parts:
                d = (b - a) * len(p) / total if total else 0
                cues_.append((a, a + d, p))
                a += d
    # hold each cue a little longer, but never past the next one
    return [(a, min(b + 0.15, cues_[i + 1][0] - 0.02) if i + 1 < len(cues_) else b + 0.15, txt)
            for i, (a, b, txt) in enumerate(cues_)]


def ts(t, sep=","):
    h, m = int(t // 3600), int(t % 3600 // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{int(s):02d}{sep}{int(round((s - int(s)) * 1000)):03d}"


def write_srt(cues_, path):
    lines = []
    for i, (a, b, txt) in enumerate(cues_, 1):
        wrapped = "\n".join(textwrap.wrap(txt, 56))
        lines += [str(i), f"{ts(a)} --> {ts(b)}", wrapped, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_ass(cues_, path):
    head = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,Inter Medium,38,&H00FFFFFF,&H00FFFFFF,&H32180F08,&H32180F08,0,0,0,0,100,100,0,0,3,14,0,2,260,260,44,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = []
    for a, b, txt in cues_:
        wrapped = "\\N".join(textwrap.wrap(txt, 64))
        ev.append(f"Dialogue: 0,{ts(a, '.')[1:-1]},{ts(b, '.')[1:-1]},Sub,,0,0,0,,{wrapped}")
    path.write_text(head + "\n".join(ev) + "\n", encoding="utf-8")


def write_docs(timing, starts, total):
    rows = []
    for seg, t0 in zip(timing["segments"], starts):
        rows.append({"segment": seg["id"], "scene_module": seg["scene"], "start_s": round(t0, 2),
                     "end_s": round(t0 + seg["duration"], 2), "duration_s": round(seg["duration"], 2),
                     "simulation_clip": CLIP_OF.get(seg["scene"], "(video-only scene)"),
                     "narration": " ".join(s["text"] for s in seg["sentences"])})
    with open(VIDEO / "scene_timeline.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = [f"# Voice-over script — Neural IK, STP Team 06 (≈ {total / 60:.0f} min {total % 60:.0f} s)", "",
          f"Voice: Kokoro-82M `{timing['voice']}` (text-to-speech, Apache-2.0), speed {timing['speed']}. "
          "Timestamps are from the final video. Every number below is checked against `deliverables/results/` "
          "by `scripts/video/narration.py` before audio is generated.", ""]
    for seg, t0 in zip(timing["segments"], starts):
        md.append(f"## {ts(t0, '.')[3:8]} – {ts(t0 + seg['duration'], '.')[3:8]} · {seg['id']} (`{seg['scene']}`)")
        md.append("")
        for s in seg["sentences"]:
            md.append(f"- **{ts(t0 + s['start'], '.')[3:8]}** {s['text']}")
        md.append("")
    (VIDEO / "voiceover_script.md").write_text("\n".join(md), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only-audio", action="store_true")
    a = ap.parse_args()
    timing = json.load(open(WORK / "voice" / "timing.json"))
    starts = list(np.cumsum([0] + [s["duration"] for s in timing["segments"]])[:-1])
    total = sum(s["duration"] for s in timing["segments"])
    VIDEO.mkdir(parents=True, exist_ok=True)
    audio = build_audio(timing, starts, total)
    cues_ = subtitles(timing, starts)
    write_srt(cues_, VIDEO / "subtitles.srt")
    ass = WORK / "video" / "subtitles.ass"
    write_ass(cues_, ass)
    write_docs(timing, starts, total)
    if a.only_audio:
        return
    files = render_segments(timing, a.force)
    lst = WORK / "video" / "concat.txt"
    lst.write_text("".join(f"file '{f}'\n" for f in files))
    silent = WORK / "video" / "video_silent.mp4"
    sh(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(silent)])
    sh(["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), "-i", str(audio),
        "-vf", f"subtitles={ass}:fontsdir=/usr/share/fonts", "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-preset", "slow", "-crf", "21", "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
        "-metadata", "title=Neural Kinematics for Intelligent Space Manipulators - STP Team 06", str(OUT)])
    dur = float(sh(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(OUT)]))
    print(f"wrote {OUT} ({dur:.1f} s, {OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
