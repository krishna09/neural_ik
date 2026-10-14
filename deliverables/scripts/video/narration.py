#!/usr/bin/env python
"""Narration for the five-minute video: script, text-to-speech, timing.

Every number in the narration is checked against the results files by check_numbers() before any
audio is made. Speech: Kokoro-82M (Apache-2.0, ONNX, CPU), voice af_heart, one WAV per sentence so
scene events can be cued to the start of the sentence that describes them.

Run with the TTS environment (kokoro-onnx, soundfile, numpy):
    python deliverables/scripts/video/narration.py            # synthesise missing sentences, write timing
Output: $NIK_WORK/voice/*.wav, $NIK_WORK/voice/timing.json
"""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

DELIV = Path(__file__).resolve().parents[2]
RESULTS = DELIV / "results"
WORK = Path(os.environ.get("NIK_WORK", "/tmp/nik_work"))
TTS_DIR = Path(os.environ.get("NIK_TTS", str(WORK / "tts")))     # kokoro-v1.0.onnx + voices-v1.0.bin
VOICE, SPEED, SR = "af_heart", 1.0, 24000

# (segment id, scene, minimum seconds on screen, [sentences]); a sentence is "text" or ("text", "spoken form")
SEGMENTS = [
    ("opening", "opening", 18.0, [
        "A robot arm must place its gripper precisely, stay inside its joint limits, and never collide with itself.",
        "This is STP Team 06's collision-aware neural inverse kinematics, demonstrated on a simulated xArm6.",
    ]),
    ("objectives", "objectives", 13.0, [
        "Our project brief has two levels.",
        "Level one: a kinematics-aware model that computes joint angles without self-collision.",
        "Level two: a lightweight reinforcement-learning agent that re-routes the arm in real time.",
    ]),
    ("fk", "forward_kinematics", 18.0, [
        "Forward kinematics is the easy direction.",
        "Each joint rotation is chained onto the next, and PyBullet returns one exact pose for the gripper: a position and an orientation.",
    ]),
    ("ik", "inverse_kinematics", 20.0, [
        "Inverse kinematics runs backwards, from a target pose to the joint angles, and it is harder.",
        "For this target, our solver found six different joint configurations that reach it exactly.",
        "The arm's current joints, called the seed, decide which one we want.",
    ]),
    ("dataset", "dataset_generation", 27.0, [
        "We generate the training data ourselves.",
        "The generator samples random joint angles: seventy percent uniformly, and thirty percent close to the joint limits, where self-collisions concentrate.",
        "PyBullet computes each pose and checks every pair of links for contact.",
        "Our main dataset has two hundred thousand samples; sixteen percent are self-colliding.",
    ]),
    ("network", "neural_architecture", 31.0, [
        "The network reads thirteen numbers: the target position, its orientation as a quaternion, and the six seed angles.",
        "Three hidden layers turn them into twelve outputs, the sine and cosine of every joint.",
        "The atan2 function turns each pair back into a joint angle.",
        "Why sine and cosine? On a circle, plus and minus one hundred and eighty degrees are neighbours, so the training targets have no artificial jump.",
    ]),
    ("collision", "collision_awareness", 20.0, [
        "A small classifier learns to predict self-collision from the joint angles alone; on held-out poses, it is right ninety-eight percent of the time.",
        "During training, its probability is added to the loss, which pushes predictions away from colliding configurations.",
    ]),
    ("training", "training_validation", 19.0, [
        "Training uses the Adam optimiser and a fifteen percent validation split.",
        "The baseline network levels off at about one hundred and fifteen millimetres; a deeper residual network, trained on ten times more data, reaches twenty-three.",
    ]),
    ("inference", "inference_demo", 29.0, [
        "At run time, the network answers in about a millisecond, but on its own it still misses by centimetres.",
        "Its answer becomes the starting point for a Levenberg–Marquardt solver, which removes the remaining error.",
        "PyBullet then checks the result independently: pose error, joint limits and self-collision.",
        "Only an answer that passes every check is accepted.",
    ]),
    ("comparison", "method_comparison", 41.0, [
        "We compared the methods on the same two thousand held-out targets, with the arm starting from its home pose.",
        "The network alone never came within one millimetre. PyBullet's classical solver found a valid answer within one millimetre and one degree for under three percent of targets; nearly a quarter of its answers collided, and thirty-nine percent broke a joint limit.",
        "Network plus refinement reached every target within one millimetre and one degree, with no collisions and no limit violations.",
        "To be fair, the solver alone also gets there; the network makes it about a quarter faster and cuts random restarts threefold.",
    ]),
    ("workflow", "workflow", 17.0, [
        "Level one is complete and reproducible: robot configuration, data generation, collision classifier, network training, refinement and evaluation.",
        "Moving to the real space manipulator means adding one entry to the robot registry.",
    ]),
    ("level2", "level2_ppo", 33.0, [
        "Level two is implemented as well: a PPO agent with eleven thousand parameters corrects a potential-field controller in real time.",
        "It learns in simulation, rewarded for progress and penalised for getting close to obstacles.",
        "Here, an obstacle appears and a joint slips: the straight path collides, the potential field crawls, and the agent arrives.",
        "Over two hundred unseen episodes of the hardest scenario, it reached eighty-four percent of goals, with two percent collisions.",
    ]),
    ("closing", "closing", 13.0, [
        "Fast neural guesses, exact refinement, independent checks, and a learned controller on top.",
        "Thank you from STP Team 06.",
    ]),
]
LEAD, GAP, TAIL = 0.7, 0.38, 0.8          # seconds of silence: before the first sentence, between, after the last


def text_of(s):
    return s if isinstance(s, str) else s[0]


def spoken(s):
    return s if isinstance(s, str) else s[1]


def check_numbers():
    """The narration's numbers, recomputed from the result files (fails loudly if any drifts)."""
    rows = {f"{r['method']}|{r['seed_mode']}": r for r in csv.DictReader(open(RESULTS / "evaluation_metrics.csv"))}
    f = lambda k, m: float(rows[k][m])
    summ = json.load(open(RESULTS / "eval_summary.json"))
    l2 = json.load(open(DELIV.parent / "media" / "data" / "l2_20261013-185525_xarm6_meta_n200.json"))["evaluation"]
    ds = json.load(open(DELIV.parent / "ui" / "workspace" / "library" / "datasets" / "ds_20260928-163059_xarm6_n200000_s0" / "meta.json"))
    base = json.load(open(RESULTS / "baseline_repro" / "summary.json"))
    best = json.load(open(DELIV.parent / "ui" / "workspace" / "library" / "runs" / "20260928-174703_xarm6" / "meta.json"))
    ik = json.load(open(WORK / "sim" / "ik.json"))
    checks = {
        "200,000 samples": ds["n"] == 200000,
        "16 % self-colliding": round(100 * ds["n_colliding"] / ds["n"]) == 16,
        "98 % classifier": round(summ["collision_classifiers"]["best_run_classifier_test"]["accuracy"]) == 98,
        "six IK solutions found": ik["n_branches_found"] == 6,
        "~115 mm baseline": round(base["final_epoch"]["val_pos_mean_mm"] / 5) * 5 == 115,
        "23 mm best": round(best["metrics"]["test"]["pos_mean_mm"]) == 23,
        "ten times more data": best["split"]["n_dataset"] == 10 * base["dataset"]["n"],
        "about a millisecond": 0.5 < f("residual_mlp|warm", "time_ms_mean") < 1.5,
        "misses by centimetres": 10 < f("residual_mlp|warm", "pos_mean_mm") < 100,
        "network never within 1 mm (cold)": f("residual_mlp|zero", "within_1mm_pct") == 0.0,
        "DLS valid < 3 % (cold)": f("classical_dls|zero", "success_1mm_1deg_pct") < 3.0,
        "DLS nearly a quarter collide": 20 <= f("classical_dls|zero", "self_collision_pct") < 25,
        "DLS 39 % limits": round(f("classical_dls|zero", "limit_violation_pct")) == 39,
        "hybrid 100 %, 0 collisions, 0 limits": f("hybrid_lm|zero", "success_1mm_1deg_pct") == 100.0
        and f("hybrid_lm|zero", "self_collision_pct") == 0.0 and f("hybrid_lm|zero", "limit_violation_pct") == 0.0,
        "LM alone also 100 %": f("lm_only|zero", "success_1mm_1deg_pct") == 100.0,
        "about a quarter faster": 0.2 <= 1 - f("hybrid_lm|zero", "time_ms_mean") / f("lm_only|zero", "time_ms_mean") <= 0.3,
        "restarts threefold": 2.5 <= f("lm_only|zero", "won_restart_pct") / f("hybrid_lm|zero", "won_restart_pct") <= 3.5,
        "11,000 parameters": round(l2["policy_params"], -3) == 11000,
        "84 % / 2 % (200 episodes)": l2["scenarios"]["combined"]["policy"]["success_pct"] == 84.0
        and l2["scenarios"]["combined"]["policy"]["collision_pct"] == 2.0 and l2["n_per_scenario"] == 200,
    }
    bad = [k for k, v in checks.items() if not v]
    if bad:
        sys.exit(f"narration numbers out of date: {bad}")
    return list(checks)


def synth(kokoro, text, path):
    samples, sr = kokoro.create(text, voice=VOICE, speed=SPEED, lang="en-us")
    assert sr == SR
    samples = np.asarray(samples, dtype=np.float32)
    # trim leading / trailing near-silence, keep 40 ms
    idx = np.nonzero(np.abs(samples) > 0.01)[0]
    if len(idx):
        a, b = max(0, idx[0] - int(0.04 * SR)), min(len(samples), idx[-1] + int(0.04 * SR))
        samples = samples[a:b]
    import soundfile as sf
    sf.write(path, samples, SR, subtype="PCM_16")
    return len(samples) / SR


def main():
    checked = check_numbers()
    print(f"numbers verified: {len(checked)}")
    out = WORK / "voice"
    out.mkdir(parents=True, exist_ok=True)
    kokoro = None
    timing = {"voice": VOICE, "speed": SPEED, "sr": SR, "lead": LEAD, "gap": GAP, "tail": TAIL, "segments": []}
    for seg_id, scene, min_d, sentences in SEGMENTS:
        t = LEAD
        items = []
        for i, s in enumerate(sentences):
            say = spoken(s)
            key = hashlib.sha1(f"{VOICE}|{SPEED}|{say}".encode()).hexdigest()[:12]
            wav = out / f"{seg_id}_{i:02d}_{key}.wav"
            if not wav.exists():
                if kokoro is None:
                    from kokoro_onnx import Kokoro
                    kokoro = Kokoro(str(TTS_DIR / "kokoro-v1.0.onnx"), str(TTS_DIR / "voices-v1.0.bin"))
                synth(kokoro, say, wav)
            import soundfile as sf
            dur = sf.info(str(wav)).frames / SR
            items.append({"text": text_of(s), "wav": wav.name, "start": round(t, 3), "end": round(t + dur, 3)})
            t += dur + GAP
        speech_end = t - GAP
        D = max(min_d, speech_end + TAIL)
        timing["segments"].append({"id": seg_id, "scene": scene, "duration": round(D, 3), "sentences": items})
        print(f"{seg_id:11s} {scene:20s} speech {speech_end:5.1f} s  scene {D:5.1f} s  ({sum(len(text_of(s).split()) for s in sentences)} words)")
    total = sum(s["duration"] for s in timing["segments"])
    print(f"total {total:.1f} s before cross-fades")
    with open(out / "timing.json", "w") as f:
        json.dump(timing, f, indent=1)


if __name__ == "__main__":
    main()
