#!/usr/bin/env python
"""Evaluate the reproduced Level 2 agent exactly the way the team's agent was evaluated, and put both
results in one table.

level2/evaluate.py (unchanged): 200 unseen episodes per scenario, seeds 50,000-50,199, the same seeds for
the DRL policy, the straight-line baseline and the potential field. The checkpoint is policy.pt, which
level2/ppo.py keeps as the best smoothed training score - the same rule that produced the team's agent.

    python deliverables/scripts/pipeline/evaluate_level2_repro.py [--policy $NIK_WORK/level2_repro/policy.pt]
Output: results/level2_repro_eval.json, results/level2_evaluation.csv
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import L2_META, RESULTS, WORK, read_json, write_json  # noqa: E402

CONTROLLERS = (("baseline", "straight path"), ("apf", "potential field"), ("policy", "DRL agent (PPO)"))


def training_summary(log_path):
    """Steps and time of the reproduction run. Wall time includes the periods the run was paused (it shared
    the CPU with the video renderer), so active time is also estimated: update intervals longer than five
    times the median interval are counted as one median interval."""
    log = [json.loads(line) for line in open(log_path)]
    run = log[[i for i, h in enumerate(log) if h["update"] == 1][-1]:]          # last run in the file
    t = np.array([0.0] + [h["secs"] for h in run])
    dt = np.diff(t)
    med = float(np.median(dt))
    last = run[-1]
    return {"steps": last["steps"], "updates": last["update"], "updates_planned": last["updates"],
            "envs": last["envs"], "workers": last["workers"], "device": last["device"],
            "wall_secs": round(float(t[-1]), 1), "active_secs_estimate": round(float(np.where(dt > 5 * med, med, dt).sum()), 1),
            "final_training_stats": {k: last[k] for k in ("success_pct", "collision_pct", "timeout_pct", "std")}}


def rows_for(source, ev):
    out = []
    for scenario, ctrls in ev["scenarios"].items():
        for key, label in CONTROLLERS:
            s = ctrls[key]
            out.append({"source": source, "scenario": scenario, "controller": label, "n_episodes": s["n"],
                        "success_pct": round(s["success_pct"], 2), "collision_pct": round(s["collision_pct"], 2),
                        "timeout_pct": round(s["timeout_pct"], 2),
                        "steps_to_goal_mean": "" if s["steps_to_goal"] is None else round(s["steps_to_goal"], 1),
                        "min_clearance_mm_mean": round(s["min_clearance_mm"], 1), "act_us_mean": round(s["act_us"], 1),
                        "policy_params": ev["policy_params"] if key == "policy" else "",
                        "policy_decision_us": round(ev["decision_us"], 1) if key == "policy" else ""})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=str(WORK / "level2_repro" / "policy.pt"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=50_000)
    a = ap.parse_args()
    from level2.evaluate import evaluate     # project code, unchanged

    t0 = time.time()
    out = evaluate("xarm6", a.policy, n=a.n, seed=a.seed,
                   on_progress=lambda p: print(f"  {p['scenario']:9s} agent {p['policy_success']:5.1f}%  "
                                               f"potential field {p['apf_success']:5.1f}%  straight {p['baseline_success']:5.1f}%",
                                               flush=True))
    out["eval_secs"] = round(time.time() - t0, 1)
    out["checkpoint"] = Path(a.policy).name
    out["training"] = training_summary(Path(a.policy).with_name("train_log.jsonl"))
    write_json(RESULTS / "level2_repro_eval.json", out)

    team = read_json(L2_META)
    rows = rows_for("team agent l2_20261013-185525 (recorded)", team["evaluation"])
    rows += rows_for("reproduction (re-trained for this presentation)", out)
    with open(RESULTS / "level2_evaluation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    c = out["scenarios"]["combined"]
    print(f"combined: agent {c['policy']['success_pct']:.1f}% reached / {c['policy']['collision_pct']:.1f}% collided; "
          f"potential field {c['apf']['success_pct']:.1f}% / {c['apf']['collision_pct']:.1f}%; "
          f"straight {c['baseline']['success_pct']:.1f}% / {c['baseline']['collision_pct']:.1f}%")
    print("wrote", RESULTS / "level2_repro_eval.json", "and", RESULTS / "level2_evaluation.csv")


if __name__ == "__main__":
    main()
