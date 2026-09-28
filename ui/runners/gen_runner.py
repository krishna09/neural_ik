"""
Runs the project's generate_dataset.generate() UNCHANGED and streams what it
is doing to the UI server.

generate() looks up set_joint_config / get_ee_pose / is_self_colliding as
module globals at call time, so wrapping those three names on the imported
module lets us observe every sample (joint angles -> FK pose -> collision
label) without editing generate_dataset.py. The wrappers call the originals
and return their results untouched, so the saved .npz is byte-for-byte what
`python generate_dataset.py --robot X --n N --seed S` would write.

Invoked by ui/server.py with cwd = the new library entry's folder
(ui/workspace/library/datasets/<id>/), so generate()'s relative output path
"data/<robot>_ik_dataset.npz" lands inside that entry and never overwrites
an earlier dataset.

Movement split (collision_pct):
  * None ("natural") -> generate(robot, n, seed) once; whatever share of
    self-colliding samples it produces (~15-16% for xarm6 / kuka) is kept,
    and the file is exactly generate()'s output.
  * a percentage -> generate() is called (still unchanged) with a larger n to
    build a pool that contains enough of both kinds, then exactly
    round(n * pct / 100) colliding and the rest collision-free samples are
    drawn from that pool (seeded, so reproducible) and saved in the same
    .npz format. The pool size is re-estimated and retried if too small.
"""

import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, HERE)

import generate_dataset as gd  # noqa: E402
from libio import read_json, rel, sha256_file, write_json  # noqa: E402


def emit(ev):
    sys.stdout.write("@@EV " + json.dumps(ev) + "\n")
    sys.stdout.flush()


MAX_POOL = 2_000_000


def main():
    params = json.loads(sys.argv[1])
    robot = params["robot"]
    n = int(params["n"])
    seed = int(params.get("seed", 0))
    ds_id = params["ds_id"]
    workspace = params["workspace"]
    collision_pct = params.get("collision_pct")          # None = natural split
    delay = float(params.get("delay_ms", 0)) / 1000.0
    # With a per-sample delay the user wants to watch every sample; otherwise
    # generation takes ~2 s for 20k samples, so stream ~25 frames/s of it.
    interval = 0.0 if delay > 0 else 0.04
    oversample_frac = 0.3                       # generate()'s default

    os.makedirs("data", exist_ok=True)  # generate() assumes data/ exists

    st = {"i": 0, "q": None, "pos": None, "orn": None, "ncoll": 0, "last": 0.0,
          "pool": n, "n_uniform": int(n * (1 - oversample_frac))}
    orig_set, orig_pose, orig_coll = gd.set_joint_config, gd.get_ee_pose, gd.is_self_colliding

    def set_joint_config(body_id, joint_indices, q):
        st["q"] = q
        return orig_set(body_id, joint_indices, q)

    def get_ee_pose(body_id, ee_link_index):
        pos, orn = orig_pose(body_id, ee_link_index)
        st["pos"], st["orn"] = pos, orn
        return pos, orn

    def is_self_colliding(body_id, margin=0.0):
        collided = orig_coll(body_id, margin)
        i = st["i"]
        st["i"] += 1
        st["ncoll"] += int(collided)
        now = time.time()
        if now - st["last"] >= interval or st["i"] == st["pool"]:
            st["last"] = now
            emit({
                "type": "gen_sample", "robot": robot, "ds_id": ds_id, "i": st["i"], "n": st["pool"],
                "target_n": n, "phase": "uniform" if i < st["n_uniform"] else "near-limits",
                "q": [float(v) for v in st["q"]],
                "pos": [float(v) for v in st["pos"]], "orn": [float(v) for v in st["orn"]],
                "collided": bool(collided), "n_colliding": st["ncoll"],
            })
        if delay:
            time.sleep(delay)
        return collided

    gd.set_joint_config = set_joint_config
    gd.get_ee_pose = get_ee_pose
    gd.is_self_colliding = is_self_colliding

    def run_generate(pool, attempt):
        st.update(i=0, ncoll=0, last=0.0, pool=pool, n_uniform=int(pool * (1 - oversample_frac)))
        emit({"type": "gen_start", "robot": robot, "ds_id": ds_id, "n": n, "pool": pool, "seed": seed,
              "collision_pct": collision_pct, "attempt": attempt, "phase": "uniform", "i": 0})
        return os.path.abspath(gd.generate(robot, pool, seed))   # the project's generate(), unchanged

    t0 = time.time()
    if collision_pct is None:
        out_path = run_generate(n, 0)
        n_colliding, split = st["ncoll"], {"mode": "natural"}
    else:
        need_col = int(round(n * float(collision_pct) / 100.0))
        need_free = n - need_col
        rate = 0.15                           # first guess of generate()'s collision rate
        for attempt in range(8):
            want = max(need_col / max(rate, 1e-4) if need_col else 0,
                       need_free / max(1 - rate, 1e-4) if need_free else 0)
            pool = int(min(MAX_POOL, max(n, math.ceil(want * 1.15))))
            out_path = run_generate(pool, attempt)
            with np.load(out_path) as z:
                d = {k: z[k] for k in z.files}
            lab = d["collision_labels"]
            n_c = int(lab.sum())
            if n_c >= need_col and len(lab) - n_c >= need_free:
                break
            if (need_free and n_c == len(lab)) or (need_col and n_c == 0):
                # One kind never occurs (e.g. panda: hand/fingers always touch, so every
                # sample is labelled colliding) - a bigger pool would not help.
                kind = "collision-free" if n_c == len(lab) else "colliding"
                sys.exit(f"generate() produced no {kind} samples in a pool of {len(lab)} for {robot}, "
                         f"so a {collision_pct}% collision split is impossible - use the natural split")
            if pool >= MAX_POOL:
                sys.exit(f"could not reach {collision_pct}% colliding: a pool of {pool} samples only had "
                         f"{n_c} colliding / {len(lab) - n_c} collision-free")
            rate = min(0.9999, max(1e-4, n_c / len(lab)))
        emit({"type": "gen_rebalance", "robot": robot, "ds_id": ds_id, "pool": pool, "pool_colliding": n_c,
              "n_free": need_free, "n_colliding": need_col})
        rng = np.random.default_rng(seed)
        idx = np.sort(np.concatenate([
            rng.choice(np.nonzero(lab == 1)[0], need_col, replace=False),
            rng.choice(np.nonzero(lab == 0)[0], need_free, replace=False),
        ]))
        np.savez(out_path, joints=d["joints"][idx], positions=d["positions"][idx],
                 orientations=d["orientations"][idx], collision_labels=lab[idx],
                 joint_limits=d["joint_limits"], dof=d["dof"])
        n_colliding = need_col
        split = {"mode": "custom", "collision_pct_requested": float(collision_pct),
                 "pool_n": pool, "pool_colliding": n_c,
                 "note": f"{need_free} collision-free + {need_col} colliding samples drawn (seeded) from "
                         f"a pool of {pool} produced by generate(robot, {pool}, seed={seed})"}
    secs = time.time() - t0

    meta = read_json("meta.json", {}) or {}
    meta.update({
        "id": ds_id, "robot": robot, "n": n, "seed": seed, "seed_source": "generate button",
        "dof": int(len(st["q"])), "n_colliding": n_colliding, "source": "ui", "split": split,
        "file": rel(out_path, workspace), "sha256": sha256_file(out_path),
        "secs": round(secs, 2), "status": "ready",
    })
    write_json("meta.json", meta)
    emit({"type": "gen_done", "robot": robot, "ds_id": ds_id, "n": n, "n_colliding": n_colliding,
          "split": split, "path": out_path, "secs": secs})


if __name__ == "__main__":
    main()
