"""
Level 2 job runner (started by ui/server.py): train or evaluate a DRL re-routing policy.

    params = {"mode": "train" | "eval", "policy_dir": ..., "robot": ..., ...}

Streams "@@EV {json}" lines like the Level 1 runners. The policy directory gets
policy.pt (best so far, updated during training), history.json and meta.json.
"""

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, HERE)

from libio import read_json, write_json  # noqa: E402


def emit(ev):
    sys.stdout.write("@@EV " + json.dumps(ev) + "\n")
    sys.stdout.flush()


def run_eval(params, meta_path, n):
    from level2.evaluate import evaluate
    pol = os.path.join(params["policy_dir"], "policy.pt")
    emit({"type": "phase", "phase": "evaluating", "n": n})
    res = evaluate(params["robot"], pol, n=n, on_progress=lambda e: emit({"type": "l2_eval_progress", **e}))
    m = read_json(meta_path) or {}
    m["evaluation"] = res
    m["evaluated"] = time.time()
    write_json(meta_path, m)
    emit({"type": "l2_eval_done", "policy_id": m.get("id"), "decision_us": res["decision_us"]})
    return res


def main():
    params = json.loads(sys.argv[1])
    meta_path = os.path.join(params["policy_dir"], "meta.json")
    if params["mode"] == "eval":
        run_eval(params, meta_path, int(params.get("n", 100)))
        return

    from level2.ppo import train
    m = read_json(meta_path) or {}
    t0 = time.time()

    def progress(h):
        emit({"type": "l2_update", "policy_id": m.get("id"), **h})

    def live(ep, upd, n):
        emit({"type": "l2_live", "policy_id": m.get("id"), "robot": params["robot"], "update": upd, "updates": n, "episode": ep})

    hist = train(params["robot"], int(params["steps"]), int(params.get("envs", 80)), int(params.get("seed", 0)),
                 params["policy_dir"], on_progress=progress, device=params.get("device"), on_live=live)
    m = read_json(meta_path) or m
    m.update(status="evaluating", train_secs=round(time.time() - t0, 1), steps_done=hist[-1]["steps"] if hist else 0,
             device=hist[-1].get("device") if hist else None, last=hist[-1] if hist else None)
    write_json(meta_path, m)
    run_eval(params, meta_path, int(params.get("eval_n", 60)))
    m = read_json(meta_path)
    m["status"] = "done"
    m["finished"] = time.time()
    write_json(meta_path, m)


if __name__ == "__main__":
    main()
