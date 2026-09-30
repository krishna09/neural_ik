"""
Runs a dataset generator and streams what it is doing to the UI server.

Generators (chosen in the UI, see generator_registry.py):
  * "base" - the project's generate_dataset.generate(), UNCHANGED.
    generate() looks up set_joint_config / get_ee_pose / is_self_colliding as
    module globals at call time, so wrapping those three names on the imported
    module lets us observe every sample (joint angles -> FK pose -> collision
    label) without editing generate_dataset.py. The wrappers call the originals
    and return their results untouched, so the saved .npz is byte-for-byte what
    `python generate_dataset.py --robot X --n N --seed S` would write.
  * "datagen/<file>.py" - an alternative generator, also used as it is:
      - with a progress callback it reports every sample itself;
      - without one ("adapted") the same wrapping trick follows its own
        set_joint_config / get_ee_pose / collision-check functions;
      - seed / out_path are passed only if generate() takes them, and the
        options set in the UI are passed as keyword arguments;
      - generators that read cfg["self_collision_ignore_pairs"] can be handed
        the link pairs that touch by construction (clean collision matrix) -
        needed for the Panda, whose hand and closed fingers always touch.

Invoked by ui/server.py with cwd = the new library entry's folder
(ui/workspace/library/datasets/<id>/), so the relative output path
"data/<robot>_ik_dataset.npz" lands inside that entry and never overwrites
an earlier dataset.

Movement split (collision_pct):
  * None ("natural") -> generate(robot, n, seed) once; whatever share of
    self-colliding samples the generator produces is kept, and the file is
    exactly the generator's output.
  * a percentage -> the generator is called (still unchanged) with a larger n
    to build a pool that contains enough of both kinds, then exactly
    round(n * pct / 100) colliding and the rest collision-free samples are
    drawn from that pool (seeded, so reproducible) and saved in the same
    .npz format. The pool size is re-estimated and retried if too small.
    Generators that keep colliding samples out of the dataset in
    rejected_joints / rejected_positions / rejected_orientations arrays add
    those rows (rejected_reason 1 = real contact) to the pool as colliding.
"""

import json
import math
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, HERE)

import generate_dataset as gd  # noqa: E402
import generator_registry as GR  # noqa: E402
from libio import read_json, rel, sha256_file, write_json  # noqa: E402


def emit(ev):
    sys.stdout.write("@@EV " + json.dumps(ev) + "\n")
    sys.stdout.flush()


MAX_POOL = 2_000_000
REQUIRED_KEYS = ("joints", "positions", "orientations", "collision_labels", "joint_limits", "dof")
REJECTED_KEYS = ("rejected_joints", "rejected_positions", "rejected_orientations")
COLLISION_CHECKS = ("is_self_colliding", "check_self_collision", "is_colliding", "in_collision", "check_collision")
GIVE_UP_AFTER = 3000        # adapted generators: stop if this many samples in a row are all rejected


def load_pool(path, gen_file):
    """The generated file as one labelled pool (+ its rejected_* rows as colliding)."""
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    missing = [k for k in REQUIRED_KEYS if k not in d]
    if missing:
        sys.exit(f"{gen_file} wrote {os.path.basename(path)} without the key(s) {', '.join(missing)} "
                 f"that training needs")
    pool = {k: d[k] for k in ("joints", "positions", "orientations")}
    pool["collision_labels"] = d["collision_labels"]
    if all(k in d for k in REJECTED_KEYS) and len(d["rejected_joints"]):
        reason = d.get("rejected_reason")
        keep = np.ones(len(d["rejected_joints"]), bool) if reason is None else (reason == 1)
        for k, rk in zip(("joints", "positions", "orientations"), REJECTED_KEYS):
            pool[k] = np.concatenate([pool[k], d[rk][keep].astype(pool[k].dtype)])
        pool["collision_labels"] = np.concatenate([
            pool["collision_labels"], np.ones(int(keep.sum()), dtype=pool["collision_labels"].dtype)])
    pool["joint_limits"], pool["dof"] = d["joint_limits"], d["dof"]
    return pool


def rejected_info(path):
    with np.load(path) as z:
        if not all(k in z.files for k in REJECTED_KEYS):
            return None
        reason = z["rejected_reason"] if "rejected_reason" in z.files else None
        return {"n": int(len(z["rejected_joints"])),
                "colliding": int((reason == 1).sum()) if reason is not None else None,
                "too_close": int((reason == 2).sum()) if reason is not None else None}


def main():
    params = json.loads(sys.argv[1])
    robot = params["robot"]
    n = int(params["n"])
    seed = int(params.get("seed", 0))
    ds_id = params["ds_id"]
    workspace = params["workspace"]
    gen_id = params.get("generator") or GR.BASE_ID
    collision_pct = params.get("collision_pct")          # None = natural split
    gen_opts = dict(params.get("gen_opts") or {})
    ignore_pairs = bool(params.get("ignore_structural_pairs"))
    delay = float(params.get("delay_ms", 0)) / 1000.0
    # With a per-sample delay the user wants to watch every sample; otherwise
    # generation takes ~2 s for 20k samples, so stream ~25 frames/s of it.
    interval = 0.0 if delay > 0 else 0.04
    oversample_frac = 0.3                       # generate_dataset.generate()'s default
    out_rel = f"data/{robot}_ik_dataset.npz"
    src = GR.source_path(gen_id)
    gen_file = rel(src, PROJECT_DIR)

    os.makedirs("data", exist_ok=True)          # generate_dataset.generate() assumes data/ exists

    st = {"i": 0, "tried": 0, "q": None, "pos": None, "orn": None, "ncoll": 0, "last": 0.0,
          "pool": n, "n_uniform": int(n * (1 - oversample_frac)), "need_col": 0}
    adapted = False
    matrix = {}

    def record(q, pos, orn, collided, phase, rejected=None):
        """One generated sample -> counters, throttled live event, optional slow motion.
        rejected (adapted generators): whether the generator keeps this sample out
        of the dataset; then "i" counts the samples it keeps."""
        st["q"] = q
        st["tried"] += 1
        st["ncoll"] += int(collided)
        if rejected is None or not rejected:
            st["i"] += 1
        if adapted and st["i"] == 0 and st["tried"] >= GIVE_UP_AFTER:
            hint = (" - tick 'Skip link pairs that touch by construction' in the generator options"
                    if not ignore_pairs and GR_ENTRY.get("supports_ignore_pairs") else "")
            sys.exit(f"{gen_file} rejected all of the first {st['tried']:,} samples for {robot} "
                     f"({st['ncoll']:,} self-colliding){hint}. Stopped instead of waiting for its own "
                     f"attempt limit.")
        now = time.time()
        if now - st["last"] >= interval or st["i"] == st["pool"]:
            st["last"] = now
            ev = {
                "type": "gen_sample", "robot": robot, "ds_id": ds_id, "i": min(st["i"], st["pool"]),
                "n": st["pool"], "target_n": n, "phase": phase,
                "q": [float(v) for v in q],
                "pos": [float(v) for v in pos], "orn": [float(v) for v in orn],
                "collided": bool(collided), "n_colliding": st["ncoll"],
            }
            if rejected is not None:
                ev.update(tried=st["tried"], rejected=bool(rejected))
            emit(ev)
        if delay:
            time.sleep(delay)

    GR_ENTRY = {}
    if gen_id == GR.BASE_ID:
        gen_module = gd
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
            phase = "uniform" if st["i"] < st["n_uniform"] else "near-limits"
            record(st["q"], st["pos"], st["orn"], collided, phase)
            return collided

        gd.set_joint_config = set_joint_config
        gd.get_ee_pose = get_ee_pose
        gd.is_self_colliding = is_self_colliding

        def run(pool):
            return gd.generate(robot, pool, seed)             # the project's generator, unchanged
    else:
        gen_module = GR.load(gen_id)
        info = GR.describe(gen_module)
        GR_ENTRY = next((g for g in GR.list_generators() if g["id"] == gen_id), {})
        accepts = info["accepts"]
        option_names = {o["name"] for o in info["options"]}
        adapted = not accepts["progress"]

        if ignore_pairs and hasattr(gen_module, "load_robot"):
            # Hand the generator the link pairs that touch by construction through its own
            # cfg["self_collision_ignore_pairs"] setting (the file itself stays unchanged).
            import collision_filter
            orig_load = gen_module.load_robot

            def load_robot(*a, **kw):
                body, cfg, limits = orig_load(*a, **kw)
                report = collision_filter.apply(body, cfg, limits) or {}
                pairs = [tuple(pr) for pr in report.get("disabled_pairs", [])]
                cfg = dict(cfg)
                cfg["self_collision_ignore_pairs"] = list(cfg.get("self_collision_ignore_pairs", [])) + pairs
                matrix.clear()
                matrix.update({k: v for k, v in report.items() if k != "disabled_pairs"})
                print(f"[gen] ignoring {len(pairs)} link pairs that touch by construction "
                      f"(clean collision matrix)", flush=True)
                return body, cfg, limits

            gen_module.load_robot = load_robot

        if adapted:
            # Follow the file's own FK / collision-check functions (looked up as module
            # globals at call time, like generate_dataset.py's), results untouched.
            if callable(getattr(gen_module, "set_joint_config", None)):
                orig_set = gen_module.set_joint_config

                def set_joint_config(*a, **kw):
                    st["q"] = np.asarray(a[2] if len(a) > 2 else kw.get("q"), dtype=float)
                    return orig_set(*a, **kw)

                gen_module.set_joint_config = set_joint_config
            if callable(getattr(gen_module, "get_ee_pose", None)):
                orig_pose = gen_module.get_ee_pose

                def get_ee_pose(*a, **kw):
                    res = orig_pose(*a, **kw)
                    st["pos"], st["orn"] = res[0], res[1]
                    return res

                gen_module.get_ee_pose = get_ee_pose
            check_name = next((c for c in COLLISION_CHECKS if callable(getattr(gen_module, c, None))), None)
            if check_name:
                orig_check = getattr(gen_module, check_name)

                def collision_check(*a, **kw):
                    res = orig_check(*a, **kw)
                    if isinstance(res, tuple):     # e.g. (colliding, near_collision, min_distance)
                        collided = bool(res[0])
                        flags = [bool(x) for x in res[:2] if isinstance(x, (bool, np.bool_))]
                        rejected = any(flags) if gen_opts.get("reject_near", True) else collided
                    else:
                        collided = rejected = bool(res)
                    if st["q"] is not None and st["pos"] is not None:
                        record(st["q"], st["pos"], st["orn"], collided, "sampling", rejected)
                    return res

                setattr(gen_module, check_name, collision_check)
        else:
            def progress(i, total, q, pos, orn, collided, phase):
                record(q, pos, orn, collided, phase)

        def run(pool):
            kw = dict(gen_opts)
            if accepts["seed"]:
                kw["seed"] = seed
            if accepts["out_path"]:
                kw["out_path"] = out_rel
            if accepts["progress"]:
                kw["progress"] = progress
            if "collision_pool_size" in option_names and st["need_col"]:
                # its rejected-sample pool must be able to hold the requested collision movements
                kw["collision_pool_size"] = max(int(kw.get("collision_pool_size", 0)), int(st["need_col"] * 1.5) + 1000)
            ret = gen_module.generate(robot, pool, **kw)
            # the file lands in data/ of this library entry (cwd), or where generate() says
            for cand in ([ret] if isinstance(ret, (str, os.PathLike)) else []) + [out_rel]:
                if cand and os.path.isfile(cand):
                    if os.path.normcase(os.path.abspath(cand)) != os.path.normcase(os.path.abspath(out_rel)):
                        shutil.copy2(cand, out_rel)
                    return out_rel
            sys.exit(f"{gen_file}: generate() finished but no dataset file was found "
                     f"(looked for {ret!r} and {out_rel})")

    def run_generate(pool, attempt):
        st.update(i=0, tried=0, ncoll=0, last=0.0, pool=pool, n_uniform=int(pool * (1 - oversample_frac)))
        emit({"type": "gen_start", "robot": robot, "ds_id": ds_id, "n": n, "pool": pool, "seed": seed,
              "generator": gen_id, "collision_pct": collision_pct, "attempt": attempt,
              "phase": "sampling", "i": 0})
        return os.path.abspath(run(pool))

    t0 = time.time()
    if collision_pct is None:
        out_path = run_generate(n, 0)
        with np.load(out_path) as z:
            missing = [k for k in REQUIRED_KEYS if k not in z.files]
        if missing:
            sys.exit(f"{gen_file} wrote {os.path.basename(out_path)} without the key(s) {', '.join(missing)} "
                     f"that training needs")
        split = {"mode": "natural"}
    else:
        need_col = int(round(n * float(collision_pct) / 100.0))
        need_free = n - need_col
        st["need_col"] = need_col
        rate = 0.15                           # first guess of the generator's collision rate
        prev_c = -1
        for attempt in range(8):
            want = max(need_col / max(rate, 1e-4) if need_col else 0,
                       need_free / max(1 - rate, 1e-4) if need_free else 0)
            pool = int(min(MAX_POOL, max(n, math.ceil(want * 1.15))))
            out_path = run_generate(pool, attempt)
            d = load_pool(out_path, gen_file)
            lab = d["collision_labels"]
            n_c = int(lab.sum())
            if n_c >= need_col and len(lab) - n_c >= need_free:
                break
            if (need_free and n_c == len(lab)) or (need_col and n_c == 0):
                # One kind never occurs (e.g. the base generator on the Panda labels
                # every sample colliding) - a bigger pool would not help.
                kind = "collision-free" if n_c == len(lab) else "colliding"
                sys.exit(f"the generator produced no {kind} samples in a pool of {len(lab)} for {robot}, "
                         f"so a {collision_pct}% collision split is impossible - use the natural split"
                         + (" or the clean generator" if gen_id == GR.BASE_ID else ""))
            if pool >= MAX_POOL or n_c == prev_c:
                sys.exit(f"could not reach {collision_pct}% colliding: a pool of {len(lab)} samples only had "
                         f"{n_c} colliding / {len(lab) - n_c} collision-free")
            prev_c = n_c
            rate = min(0.9999, max(1e-4, n_c / len(lab)))
        emit({"type": "gen_rebalance", "robot": robot, "ds_id": ds_id, "pool": int(len(lab)), "pool_colliding": n_c,
              "n_free": need_free, "n_colliding": need_col})
        rng = np.random.default_rng(seed)
        idx = np.sort(np.concatenate([
            rng.choice(np.nonzero(lab == 1)[0], need_col, replace=False),
            rng.choice(np.nonzero(lab == 0)[0], need_free, replace=False),
        ]))
        extra = rejected_info(out_path)
        np.savez(out_path, joints=d["joints"][idx], positions=d["positions"][idx],
                 orientations=d["orientations"][idx], collision_labels=lab[idx],
                 joint_limits=d["joint_limits"], dof=d["dof"])
        split = {"mode": "custom", "collision_pct_requested": float(collision_pct),
                 "pool_n": int(len(lab)), "pool_colliding": n_c,
                 "note": f"{need_free} collision-free + {need_col} colliding samples drawn (seeded) from "
                         f"a pool of {len(lab)} produced by the generator with seed={seed}"
                         + (f" (its {extra['colliding'] if extra['colliding'] is not None else extra['n']} rejected "
                            f"colliding samples included as label 1; its extra arrays are not kept because "
                            f"rows were re-drawn)" if extra else "")}
    secs = time.time() - t0

    with np.load(out_path) as z:
        labels = z["collision_labels"]
        n_rows, n_colliding, dof = int(len(labels)), int(np.asarray(labels).sum()), int(z["dof"])
    rejected = rejected_info(out_path)
    generator = {"id": gen_id, "file": gen_file, "sha256": sha256_file(src)}
    if gen_opts:
        generator["options"] = gen_opts
    if ignore_pairs:
        generator["ignore_structural_pairs"] = True
    report = getattr(gen_module, "LAST_REPORT", None) or matrix or None
    meta = read_json("meta.json", {}) or {}
    meta.update({
        "id": ds_id, "robot": robot, "n": n_rows, "seed": seed, "seed_source": "generate button",
        "dof": dof, "n_colliding": n_colliding, "source": "ui", "split": split,
        "generator": generator, "generator_report": dict(report) if report else None,
        "rejected_pool": rejected,
        "file": rel(out_path, workspace), "sha256": sha256_file(out_path),
        "secs": round(secs, 2), "status": "ready",
    })
    write_json("meta.json", meta)
    emit({"type": "gen_done", "robot": robot, "ds_id": ds_id, "n": n_rows, "n_colliding": n_colliding,
          "split": split, "generator": generator, "generator_report": meta["generator_report"],
          "rejected": rejected, "path": out_path, "secs": secs})


if __name__ == "__main__":
    main()
