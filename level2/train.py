"""
Train the Level 2 re-routing policy from the command line.

    python -m level2.train --robot xarm6 --steps 3000000
    python -m level2.evaluate --policy level2/runs/<id>/policy.pt
"""

import argparse
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="xarm6", choices=["xarm6", "kuka_iiwa"])
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--envs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda when available)")
    args = ap.parse_args()
    out = args.out or os.path.join(HERE, "runs", time.strftime("%Y%m%d-%H%M%S") + f"_{args.robot}")

    def show(h):
        if h["update"] == 1:
            print(f"PPO updates on {h['device']} · {h['envs']} PyBullet envs on {h['workers']} CPU worker processes", flush=True)
        print(f"update {h['update']:4d}/{h['updates']}  steps {h['steps']:>9,}  {h['fps']:>6,}/s  "
              f"success {h['success_pct']:5.1f}%  collision {h['collision_pct']:5.1f}% (self {h['self_collision_pct']:4.1f}%)  timeout {h['timeout_pct']:5.1f}%", flush=True)

    from .ppo import train   # imported here: spawned simulation workers re-import this module and must stay torch-free
    train(args.robot, args.steps, args.envs, args.seed, out, on_progress=show, device=args.device)
    print("saved", os.path.join(out, "policy.pt"))


if __name__ == "__main__":
    main()
