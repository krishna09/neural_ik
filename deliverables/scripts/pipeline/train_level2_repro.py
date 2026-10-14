#!/usr/bin/env python
"""Train a reproduction of the Level 2 re-routing agent (level2/ppo.py, unchanged) on the CPU.

The team's best agent (l2_20261013-185525_xarm6: 20 M steps, 134 min with GPU updates and 10 CPU
workers) is not in git - only its evaluation file media/data/l2_..._meta_n200.json is. This script
trains a new agent with the same code and settings (80 environments, observation v2, residual on the
potential field, 20 M steps) on a CPU, so the video shows a real trained policy and the evaluation
can be re-run here (pipeline/evaluate_level2_repro.py).

    python deliverables/scripts/pipeline/train_level2_repro.py --steps 20000000 --workers 3
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import REPO, WORK  # noqa: E402

sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--envs", type=int, default=80)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(WORK / "level2_repro"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "train_log.jsonl"), "a")

    def show(h):
        log.write(json.dumps(h) + "\n")
        log.flush()
        if h["update"] == 1 or h["update"] % 10 == 0 or h["update"] == h["updates"]:
            print(f"update {h['update']:4d}/{h['updates']}  steps {h['steps']:>9,}  {h['fps']:>5,}/s  "
                  f"success {h['success_pct']:5.1f}%  collision {h['collision_pct']:5.1f}%  "
                  f"timeout {h['timeout_pct']:5.1f}%", flush=True)

    from level2.ppo import train          # imported here: spawned workers must stay torch-free
    t0 = time.time()
    train("xarm6", a.steps, a.envs, a.seed, a.out, on_progress=show, workers=a.workers, device="cpu")
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {a.out}/policy.pt", flush=True)


if __name__ == "__main__":
    main()
