#!/usr/bin/env python
"""Assemble the 90-second marketing video (brief section 4) -> media/out/neural_ik_90s.mp4.

Inputs (make them first):
    media/assets/anim_*.mp4          python media/scripts/anim_charts.py
    media/assets/frames/<shot>/      .venv\\Scripts\\python.exe media\\scripts\\capture_studio.py   (Windows python)

Steps: Studio frames -> clips with caption cards and zooms; voice-over per scene with edge-tts (Microsoft online
neural voice); a calm synthesized music bed ducked under the voice; captions burned in; 0.4 s cross-fades;
H.264 CRF 18 + AAC 192k. Every number in the voice-over and the cards is read from the run files.

    python media/scripts/build_video.py                # build everything that is missing, then the video
    python media/scripts/build_video.py --redo-voice   # regenerate the voice-over
    python media/scripts/build_video.py --check        # also export one frame per scene to media/assets/check/
"""
import argparse
import asyncio
import json
import math
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anim_charts as A  # noqa: E402  (colours, fonts, drawing helpers, run-file facts)

ROOT = A.ROOT
ASSETS = ROOT / "media" / "assets"
BUILD = ASSETS / "build"
VOICE = ASSETS / "voice"
OUT = ROOT / "media" / "out" / "neural_ik_90s.mp4"
W, H, FPS = A.W, A.H, A.FPS
XF = 0.4                                     # cross-fade seconds
SR = 48000
TTS_VOICE, TTS_RATE = "en-IN-NeerjaNeural", "-4%"


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{cmd[0]} failed:\n{r.stderr[-2000:]}")
    return r.stdout


def duration(path):
    return float(sh(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]).strip())


# ================================================================================================ script
def words_ratio(pct):
    """'one in two' style wording for the collision beat (asserts it matches the data)."""
    table = {50: "one in two", 2: "two in a hundred"}
    assert round(pct) in table, f"no wording for {pct} %"
    return table[round(pct)]


def storyboard(F):
    comb = F["l2_eval"]["combined"]
    m = F["m_dyn"]
    pol = m["controllers"]["policy"]
    params_k = round(F["l2_params"], -3)
    assert params_k == 11000, "voice says 'eleven thousand parameters'"
    us = F["l2_us"]
    assert round(us) in (17, 18), "voice says 'seventeen microseconds'"
    final = m["median_final_mm"]
    assert f"{final:.3f}" == "0.002", "voice says 'two thousandths of a millimetre'"
    conflict = m["predicted_conflict_pct"]
    return [
        {"id": "s1", "parts": [("anim", "title")], "vo_at": 0.7,
         "vo": "A robot arm in space can't afford a single collision."},
        {"id": "s2", "parts": [("frames", "mission_intro", None)], "vo_at": 1.4,
         "card": ("MISSION · L1 + L2", A.C_MISSION, "Where must every joint go?",
                  "Red frame = target pose for the gripper · xArm6 · in simulation"),
         "vo": "Tell it where the gripper must go…"},
        {"id": "s3", "parts": [("still", "infer", 3.6), ("anim", "ik_drop")], "vo_at": 0.3,
         "card": ("LEVEL 1 · SOLVE", A.C_L1, "A network guesses, a solver makes it exact",
                  "Orange ghost = network alone · solid arm = network + LM solver"),
         "vo": "…a neural network guesses the joint angles, a solver makes them exact – "
               "in milliseconds, and never colliding with itself."},
        {"id": "s4", "parts": [("frames", "mission_plan", "card")], "vo_at": 0.8,
         "card": ("PATH CHECK", A.MUTED, f"{conflict:.0f} % of straight paths would crash",
                  f"Estimated straight path checked against the obstacles · {m['n']} dynamic missions", "right"),
         "vo": "But the straight path to that answer often runs straight into an obstacle."},
        {"id": "s5", "parts": [("frames", "mission_run", None)], "vo_at": 0.8,
         "card": ("LEVEL 2 · STEER", A.C_L2, "Same mission, two controllers",
                  "Solid arm = DRL agent · translucent green = straight path (collides)"),
         "vo": "So a tiny reinforcement-learning agent steers the arm – reacting to obstacles that appear or move, "
               "and to joints that slip."},
        {"id": "s6", "parts": [("anim", "waffle")], "vo_at": 4.2,
         "vo": f"Collisions drop from {words_ratio(comb['baseline']['collision_pct'])}… "
               f"to {words_ratio(comb['policy']['collision_pct'])}."},
        {"id": "s7", "parts": [("anim", "tiny_agent")], "vo_at": 0.9,
         "vo": "And it's light: eleven thousand parameters, seventeen microseconds per decision – no GPU on board."},
        {"id": "s8", "parts": [("frames", "mission_replay", None), ("anim", "mission_card")], "vo_at": 0.6,
         "card": ("MISSION · SOLVE + STEER", A.C_MISSION, "Solve, then steer", "A second mission, start to target"),
         "vo": f"Solve, then steer: {pol['success_pct']:.0f} percent of dynamic missions reach the target, "
               "to two thousandths of a millimetre."},
        {"id": "s9", "parts": [("anim", "end")], "vo_at": 0.5,
         "vo": "Neural kinematics and reinforcement learning – for collision-free space manipulators."},
    ]


# ================================================================================================ voice
async def _tts(text, path):
    import edge_tts
    await edge_tts.Communicate(text, TTS_VOICE, rate=TTS_RATE).save(str(path))


def make_voice(scenes, redo):
    VOICE.mkdir(parents=True, exist_ok=True)
    for s in scenes:
        mp3 = VOICE / f"vo_{s['id']}.mp3"
        txt = VOICE / f"vo_{s['id']}.txt"
        if redo or not mp3.exists() or not txt.exists() or txt.read_text() != s["vo"]:
            asyncio.run(_tts(s["vo"], mp3))
            txt.write_text(s["vo"])
        s["vo_file"], s["vo_dur"] = mp3, duration(mp3)


def read_audio(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768


# ================================================================================================ music
def _reverb(x, secs=2.6, mix=0.32, seed=3):
    n = int(secs * SR)
    rng = np.random.default_rng(seed)
    ir = rng.standard_normal(n) * np.exp(-np.arange(n) / (SR * secs / 6.0))
    ir /= np.sqrt((ir ** 2).sum())
    size = 1 << int(math.ceil(math.log2(len(x) + n)))
    wet = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(ir, size), size)[: len(x)]
    return (1 - mix) * x + mix * wet


def make_music(total):
    """Calm electronic bed, 96 BPM: soft pad + plucked arpeggio + sub pulse (A minor, 8-beat chords)."""
    bpm = 96
    beat = 60 / bpm
    n = int(total * SR)
    t = np.arange(n) / SR
    out_l, out_r = np.zeros(n), np.zeros(n)
    hz = lambda midi: 440 * 2 ** ((midi - 69) / 12)                       # noqa: E731
    chords = [[57, 60, 64, 67, 71], [53, 57, 60, 64, 67], [48, 52, 55, 59, 64], [55, 59, 62, 66, 69]]   # Am9 Fmaj9 Cmaj7 G6
    bar = 8 * beat
    for ci in range(int(total / bar) + 1):
        notes = chords[ci % len(chords)]
        t0 = ci * bar
        i0, i1 = int(t0 * SR), min(n, int((t0 + bar + 2.0) * SR))
        if i0 >= n:
            break
        tt = t[i0:i1] - t0
        env = np.minimum(1, tt / 1.6) * np.where(tt < bar, 1, np.exp(-(tt - bar) / 0.7))
        for k, note in enumerate(notes):                                  # pad: detuned sines
            f = hz(note)
            v = sum(np.sin(2 * np.pi * f * d * tt + k) for d in (0.997, 1.0, 1.003)) / 3
            v += 0.18 * np.sin(2 * np.pi * 2 * f * tt)
            pan = 0.35 + 0.3 * (k / (len(notes) - 1))
            out_l[i0:i1] += 0.05 * env * v * (1 - pan)
            out_r[i0:i1] += 0.05 * env * v * pan
        root = hz(notes[0] - 24)
        for b in range(8):                                                # sub pulse on every beat
            j0 = int((t0 + b * beat) * SR)
            j1 = min(n, j0 + int(0.5 * SR))
            if j0 >= n:
                break
            u = t[j0:j1] - (t0 + b * beat)
            s = 0.09 * np.sin(2 * np.pi * root * u) * np.exp(-u / 0.22) * np.minimum(1, u / 0.01)
            out_l[j0:j1] += s
            out_r[j0:j1] += s
        arp = [notes[i] + 12 for i in (0, 2, 4, 3, 1, 2, 4, 2)]
        for e in range(16):                                               # 8th-note pluck arpeggio
            j0 = int((t0 + e * beat / 2) * SR)
            j1 = min(n, j0 + int(0.6 * SR))
            if j0 >= n:
                break
            u = t[j0:j1] - (t0 + e * beat / 2)
            f = hz(arp[e % 8])
            s = 0.035 * (np.sin(2 * np.pi * f * u) + 0.3 * np.sin(4 * np.pi * f * u)) * np.exp(-u / 0.16)
            s *= np.minimum(1, u / 0.004)
            pan = 0.3 if e % 2 else 0.7
            out_l[j0:j1] += s * (1 - pan)
            out_r[j0:j1] += s * pan
    out = np.stack([_reverb(out_l, seed=3), _reverb(out_r, seed=4)])
    fade_in, fade_out = np.minimum(1, t / 2.5), np.minimum(1, (total - t) / 3.5).clip(0, 1)
    out *= fade_in * fade_out
    return out / (np.abs(out).max() + 1e-9)


def mix_audio(scenes, total, path):
    voice = np.zeros(int(total * SR) + SR)
    for s in scenes:
        a = read_audio(s["vo_file"])
        i = int(s["vo_start"] * SR)
        voice[i:i + len(a)] += a[: len(voice) - i]
    voice = voice[: int(total * SR)]
    music = make_music(total)
    # ducking: -18 dB bed, a further -12 dB while someone speaks (50 ms attack, 450 ms release)
    lvl = np.abs(voice)
    k = int(0.03 * SR)
    lvl = np.convolve(lvl, np.ones(k) / k, mode="same")
    talking = (lvl > 0.01).astype(float)
    g = np.empty_like(talking)
    acc = 0.0
    a_att, a_rel = 1 - math.exp(-1 / (0.05 * SR)), 1 - math.exp(-1 / (0.45 * SR))
    for i, v in enumerate(talking):                                       # one-pole smoother (fast enough at 48 kHz)
        acc += (v - acc) * (a_att if v > acc else a_rel)
        g[i] = acc
    bed = 10 ** (-18 / 20) * (1 - g * (1 - 10 ** (-12 / 20)))
    vo = voice / (np.abs(voice).max() + 1e-9) * 10 ** (-1.5 / 20)
    mixed = np.stack([vo + music[0] * bed, vo + music[1] * bed])
    mixed /= max(1.0, np.abs(mixed).max() / 0.98)
    pcm = (mixed.T * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


# ================================================================================================ video parts
def card_png(card, path):
    """Transparent caption card (chapter pill + headline + one line), bottom-left, in the Studio style."""
    import matplotlib.pyplot as plt
    A._setup_fonts()
    fig = plt.figure(figsize=(W / A.DPI, H / A.DPI), dpi=A.DPI)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.patch.set_alpha(0)
    chapter, col, title, sub = card[:4]
    t1 = A.T(ax, 0, 178, title, 34, A.TXT, "semibold", z=6)
    t2 = A.T(ax, 0, 136, sub, 19, A.TXT2, z=6)
    w = max(A.text_w(ax, t1), A.text_w(ax, t2)) + 64
    x = W - 60 - w if card[4:] == ("right",) else 60             # bottom-right keeps the zoomed-in panel free
    t1.set_x(x + 32)
    t2.set_x(x + 34)
    A.rbox(ax, x, 112, w, 116, 16, fc=A.SURF, ec=A.BORDER, alpha=0.94, z=4)
    A.rbox(ax, x, 112 + 116 - 5, w, 5, 2.5, fc=col, ec="none", z=5)
    A.pill(ax, x, 270, chapter, col)
    fig.savefig(path, dpi=A.DPI, transparent=True)
    plt.close(fig)


def card_filter(idx, dur):
    return (f"[{idx}:v]format=rgba,fade=in:st=0.35:d=0.45:alpha=1,fade=out:st={max(0.5, dur - 0.9):.2f}:d=0.45:alpha=1[ov];"
            f"[v][ov]overlay=0:0:shortest=1,format=yuv420p[out]")


def zoom_card_expr(n_frames):
    """Push in on the pipeline card '3 · Trajectory estimate' (left panel) after 3.2 s."""
    s0, s1 = int(3.2 * FPS), int(5.4 * FPS)
    p = f"min(1,max(0,(on-{s0})/{s1 - s0}))"
    z = f"1+1.25*(1-cos(PI*{p}))/2"
    cx, cy = 2 * 180, 2 * 690                               # card centre in the 2x upscaled frame
    x = f"max(0,min(iw-iw/zoom,{cx}*({p})+(iw/2)*(1-{p})-iw/zoom/2))"
    y = f"max(0,min(ih-ih/zoom,{cy}*({p})+(ih/2)*(1-{p})-ih/zoom/2))"
    return f"scale=3840:2160,zoompan=z='{z}':x='{x}':y='{y}':d=1:s={W}x{H}:fps={FPS}"


def build_part(scene, k, part, F):
    BUILD.mkdir(parents=True, exist_ok=True)
    out = BUILD / f"{scene['id']}_{k}.mp4"
    kind = part[0]
    enc = ["-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-r", str(FPS), "-an"]
    has_card = "card" in scene and kind in ("frames", "still") and k == 0
    png = BUILD / f"{scene['id']}_card.png"
    if has_card:
        card_png(scene["card"], png)
    if kind == "anim":
        src = ASSETS / f"anim_{part[1]}.mp4"
        sh(["ffmpeg", "-y", "-v", "error", "-i", str(src), *enc, str(out)])
    elif kind == "frames":
        src = ASSETS / "frames" / part[1]
        n = len(list(src.glob("f_*.jpg")))
        if n < 10:
            raise FileNotFoundError(f"{src} has {n} frames - run capture_studio.py first")
        dur = n / FPS
        pre = zoom_card_expr(n) if part[2] == "card" else "null"
        fc = f"[0:v]{pre},scale={W}:{H},setsar=1[v];" + (card_filter(1, dur) if has_card else "[v]format=yuv420p[out]")
        ins = ["-framerate", str(FPS), "-i", str(src / "f_%05d.jpg")] + (["-loop", "1", "-t", f"{dur:.3f}", "-i", str(png)] if has_card else [])
        sh(["ffmpeg", "-y", "-v", "error", *ins, "-filter_complex", fc, "-map", "[out]", *enc, str(out)])
    elif kind == "still":
        src = ASSETS / "frames" / part[1] / "f_00000.jpg"
        dur = part[2]
        nf = int(dur * FPS)
        # slow push-in towards the 3D view (centre right of the left panel)
        zp = (f"scale=3840:2160,zoompan=z='1+0.22*on/{nf}':x='(iw*0.62)-iw/zoom/2':y='(ih*0.42)-ih/zoom/2':"
              f"d={nf}:s={W}x{H}:fps={FPS}")
        fc = f"[0:v]{zp},setsar=1[v];" + card_filter(1, dur)
        sh(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-framerate", str(FPS), "-t", f"{dur:.3f}", "-i", str(src),
            "-loop", "1", "-t", f"{dur:.3f}", "-i", str(png), "-filter_complex", fc, "-map", "[out]", "-frames:v", str(nf),
            *enc, str(out)])
    return out


def pad_clip(path, extra):
    out = path.with_name(path.stem + "_pad.mp4")
    sh(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-vf", f"tpad=stop_mode=clone:stop_duration={extra:.3f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-an", str(out)])
    return out


# ================================================================================================ captions
def srt_time(s):
    ms = int(round(s * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def caption_chunks(text, limit=58):
    """Split a voice line into caption cues of at most two lines of `limit` characters."""
    words, cues, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > limit and cur:
            cues.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    cues.append(cur)
    return cues


def write_srt(scenes, path):
    lines, idx = [], 1
    for s in scenes:
        chunks = caption_chunks(s["vo"])
        total = sum(len(c) for c in chunks)
        t = s["vo_start"]
        for c in chunks:
            d = s["vo_dur"] * len(c) / total
            lines += [str(idx), f"{srt_time(t)} --> {srt_time(t + d + 0.15)}", c, ""]
            idx += 1
            t += d
    path.write_text("\n".join(lines), encoding="utf-8")


# ================================================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--redo-voice", action="store_true")
    ap.add_argument("--check", action="store_true", help="export one frame per scene to media/assets/check/")
    args = ap.parse_args()

    F = A.load_facts()
    scenes = storyboard(F)
    make_voice(scenes, args.redo_voice)

    # parts -> clips, timeline with cross-fades; a scene is stretched (last frame held) if its voice needs it
    clips, t = [], 0.0
    for s in scenes:
        s["start"] = t
        files = [build_part(s, k, p, F) for k, p in enumerate(s["parts"])]
        durs = [duration(f) for f in files]
        scene_len = sum(durs) - XF * (len(files) - 1)
        need = s["vo_at"] + s["vo_dur"] + 0.5 + XF
        if need > scene_len:
            files[-1] = pad_clip(files[-1], need - scene_len)
            durs[-1] = duration(files[-1])
            scene_len = sum(durs) - XF * (len(files) - 1)
        s["vo_start"] = t + s["vo_at"]
        s["len"] = scene_len
        clips += list(zip(files, durs))
        t += scene_len - XF
        print(f"{s['id']}: {s['start']:5.1f}-{s['start'] + scene_len:5.1f} s  voice {s['vo_dur']:.1f} s  | {s['vo']}")
    total = sum(d for _, d in clips) - XF * (len(clips) - 1)
    print(f"total {total:.1f} s, {len(clips)} clips")

    wav = BUILD / "mix.wav"
    mix_audio(scenes, total, wav)
    srt = BUILD / "captions.srt"
    write_srt(scenes, srt)

    # cross-fade chain + captions + audio
    ins, fc, prev, off = [], [], "0:v", 0.0
    for i, (f, d) in enumerate(clips):
        ins += ["-i", str(f)]
        if i == 0:
            off = d - XF
            continue
        lab = f"x{i}"
        fc.append(f"[{prev}][{i}:v]xfade=transition=fade:duration={XF}:offset={off:.3f}[{lab}]")
        prev, off = lab, off + d - XF
    fontdir = Path.home() / ".cache" / "neural_ik_fonts"
    style = ("FontName=Segoe UI,FontSize=12,PrimaryColour=&H00F7F2EE,BackColour=&H40161A0E,OutlineColour=&H40161A0E,"
             "BorderStyle=4,Outline=0,Shadow=0,MarginV=6,Alignment=2")
    fc.append(f"[{prev}]subtitles='{srt}':fontsdir='{fontdir}':force_style='{style}',format=yuv420p[v]")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sh(["ffmpeg", "-y", "-v", "error", *ins, "-i", str(wav), "-filter_complex", ";".join(fc), "-map", "[v]",
        "-map", f"{len(clips)}:a", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", str(OUT)])
    print(f"wrote {OUT.relative_to(ROOT)}  ({duration(OUT):.1f} s)")

    meta = [{k: (str(v) if isinstance(v, Path) else v) for k, v in s.items() if k not in ("parts",)} for s in scenes]
    (BUILD / "timeline.json").write_text(json.dumps(meta, indent=1, default=str))
    if args.check:
        chk = ASSETS / "check"
        chk.mkdir(exist_ok=True)
        for s in scenes:
            at = s["start"] + min(s["len"] - 0.6, max(1.5, s["len"] * 0.75))
            sh(["ffmpeg", "-y", "-v", "error", "-ss", f"{at:.2f}", "-i", str(OUT), "-frames:v", "1", str(chk / f"{s['id']}.png")])
        print(f"check frames in {chk.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
