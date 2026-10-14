#!/usr/bin/env python
"""Render the standalone simulation clips (clip variant, on-screen captions) to $NIK_WORK/clips/<name>.mp4,
then encode the deliverable copies into deliverables/simulations/ (smaller files, same 1920x1080 / 30 fps).

    python deliverables/scripts/render/render_clips.py                 # every clip
    python deliverables/scripts/render/render_clips.py robot_intro      # some
"""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import SIMS, WORK  # noqa: E402

CLIPS = {                       # scene module -> deliverable file name
    "robot_intro": "robot_intro.mp4",
    "forward_kinematics": "forward_kinematics.mp4",
    "inverse_kinematics": "inverse_kinematics.mp4",
    "dataset_generation": "dataset_generation.mp4",
    "neural_architecture": "neural_architecture.mp4",
    "collision_awareness": "collision_awareness.mp4",
    "training_validation": "training_validation.mp4",
    "inference_demo": "inference_demo.mp4",
    "method_comparison": "method_comparison.mp4",
    "level2_ppo": "level2_ppo_rerouting.mp4",
}
HERE = Path(__file__).resolve().parent


def main(names):
    (WORK / "clips").mkdir(parents=True, exist_ok=True)
    SIMS.mkdir(parents=True, exist_ok=True)
    for n in names:
        t0 = time.time()
        master = WORK / "clips" / f"{n}.mp4"
        subprocess.run([sys.executable, str(HERE / "render_scene.py"), n, "--out", str(master), "--crf", "14"], check=True)
        out = SIMS / CLIPS[n]
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(master), "-c:v", "libx264", "-preset", "slow",
                        "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out)], check=True)
        print(f"== {n}: {time.time() - t0:.0f} s -> {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(CLIPS))
