"""
Level 2 evaluation: the trained DRL policy vs. the straight-line baseline on the
same scenarios (same seeds), plus recorded episodes for the Studio's 3D playback.

    python -m level2.evaluate --policy level2/runs/<id>/policy.pt --n 200
"""

import argparse
import json
import time

import numpy as np

from .env import ReachAvoidEnv, apf_action, straight_line_action
from .ppo import NumpyPolicy

SCENARIOS = {
    "mixed":    None,                                                             # random mix, as in training
    "static":   {"static": True, "sudden": False, "slip": False},                 # obstacles on the path from the start
    "sudden":   {"static": False, "sudden": True, "slip": False},                 # an obstacle appears mid-motion
    "moving":   {"static": True, "sudden": True, "slip": False, "moving": True},  # drifting obstacles
    "slip":     {"static": False, "sudden": False, "slip": True},                 # hardware: sudden joint slip
    "combined": {"static": True, "sudden": True, "slip": True},                   # everything at once
}


def run_episode(env, controller, seed, scenario=None, record=False, start=None, goal=None):
    """controller: NumpyPolicy, "apf" (potential-field baseline) or None (straight-line baseline)."""
    obs = env.reset(seed=seed, scenario=scenario, start=start, goal=goal)
    frames, done, info, ret, min_clear, t_act = [], False, {}, 0.0, 1.0, 0.0
    if record:
        frames.append({"t": 0, "q": [round(float(v), 5) for v in env.q], "links": env.link_frames(),
                       "obstacles": env.obstacles(), "clearance": None})
    while not done:
        t0 = time.perf_counter()
        if controller is None:
            a, absolute = straight_line_action(env), True
        elif controller == "apf":
            a, absolute = apf_action(env), True
        else:
            a, absolute = controller.act(obs), False
        t_act += time.perf_counter() - t0
        obs, r, done, info = env.step(a, absolute=absolute)
        ret += r
        min_clear = min(min_clear, info["clearance"])
        if record:
            frames.append({"t": env.t, "q": [round(float(v), 5) for v in env.q], "links": env.link_frames(),
                           "obstacles": env.obstacles(), "clearance": round(float(info["clearance"]), 4),
                           "risk": round(float(info.get("collision_risk", 0)), 4)})
    out = {"outcome": info["outcome"], "collision": info.get("collision"), "steps": env.t, "return": ret,
           "min_clearance_m": min_clear, "final_ee_err_m": info["ee_err"], "events": list(env.events),
           "scenario": dict(env.scenario), "act_us": 1e6 * t_act / max(1, env.t)}
    if record:
        out["frames"] = frames
    return out


def summarize(eps):
    n = len(eps)
    pct = lambda k: 100.0 * sum(e["outcome"] == k for e in eps) / max(1, n)
    succ = [e for e in eps if e["outcome"] == "success"]
    return {
        "n": n, "success_pct": pct("success"), "collision_pct": pct("collision"), "timeout_pct": pct("timeout"),
        "steps_to_goal": float(np.mean([e["steps"] for e in succ])) if succ else None,
        "min_clearance_mm": 1000 * float(np.mean([e["min_clearance_m"] for e in eps])),
        "act_us": float(np.mean([e["act_us"] for e in eps])),
    }


def evaluate(robot, policy_path, n=100, seed=50_000, scenarios=None, env=None, on_progress=None):
    """Policy vs. baseline on identical seeds for every scenario."""
    pol = NumpyPolicy(policy_path)
    own = env is None
    env = env or ReachAvoidEnv(robot, seed=seed, **dict({"base": "straight"}, **(pol.meta.get("env_cfg") or {})))
    res = {}
    names = scenarios or list(SCENARIOS)
    try:
        for si, name in enumerate(names):
            sc = SCENARIOS[name]
            rows = {}
            for label, ctrl in (("policy", pol), ("baseline", None), ("apf", "apf")):
                eps = [run_episode(env, ctrl, seed + i, sc) for i in range(n)]
                rows[label] = summarize(eps)
            res[name] = rows
            if on_progress:
                on_progress({"scenario": name, "i": si + 1, "n": len(names), **{f"{k}_success": v["success_pct"] for k, v in rows.items()}})
    finally:
        if own:
            env.close()
    return {"robot": robot, "n_per_scenario": n, "seed": seed, "scenarios": res,
            "policy_params": pol.n_params, "decision_us": decision_latency_us(pol)}


def decision_latency_us(pol, reps=2000):
    obs = np.zeros(len(pol.mean), np.float32)
    pol.act(obs)
    t0 = time.perf_counter()
    for _ in range(reps):
        pol.act(obs)
    return 1e6 * (time.perf_counter() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=50_000)
    args = ap.parse_args()
    pol = NumpyPolicy(args.policy)
    out = evaluate(pol.meta["robot"], args.policy, n=args.n, seed=args.seed)
    print(f"robot {out['robot']}  ·  policy {out['policy_params']:,} parameters  ·  {out['decision_us']:.1f} µs per decision (NumPy, 1 CPU core)")
    print(f"{'scenario':10s} {'controller':10s} {'success':>8s} {'collision':>10s} {'timeout':>8s} {'steps':>6s} {'clearance':>10s}")
    for name, rows in out["scenarios"].items():
        for label, s in rows.items():
            st = f"{s['steps_to_goal']:.0f}" if s["steps_to_goal"] else "-"
            print(f"{name:10s} {label:10s} {s['success_pct']:7.1f}% {s['collision_pct']:9.1f}% {s['timeout_pct']:7.1f}% {st:>6s} {s['min_clearance_mm']:8.0f} mm")
    with open(args.policy.replace(".pt", "_eval.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
