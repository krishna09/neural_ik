#!/usr/bin/env python
"""Assets for the deck: video posters (frames of the rendered clips), the title hero image, and one
data file with every number the slides show (read from the result files, never typed by hand).

    python deliverables/scripts/deck/prepare_deck_assets.py
Output: $NIK_WORK/deck/{posters/*.png, hero.png, circle.png, data.json}
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BEST_RUN_DIR, DATASET_200K, L2_META, REPO, RESULTS, SCRIPTS, SIMS, WORK, read_json  # noqa: E402

OUT = WORK / "deck"
POSTERS = {                       # clip -> time (s) of a representative frame
    "robot_intro": 12.3, "forward_kinematics": 19.0, "inverse_kinematics": 10.5, "dataset_generation": 23.5,
    "neural_architecture": 15.5, "collision_awareness": 20.0, "training_validation": 21.0, "inference_demo": 9.0,
    "method_comparison": 12.5, "level2_ppo_rerouting": 19.0,
}


def posters():
    d = OUT / "posters"
    d.mkdir(parents=True, exist_ok=True)
    for name, t in POSTERS.items():
        src = SIMS / f"{name}.mp4"
        if not src.exists():
            print(f"  (missing clip {src.name})")
            continue
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t), "-i", str(src), "-frames:v", "1",
                        "-vf", "scale=1280:720", str(d / f"{name}.png")], check=True)


def hero_and_circle():
    """Hero: the opening scene with its titles moved out of the timeline (robot + target only)."""
    from PIL import Image, ImageDraw, ImageFilter
    timing = WORK / "timing" / "hero.json"
    timing.parent.mkdir(parents=True, exist_ok=True)
    timing.write_text(json.dumps({"duration": 16, "cues": {"title": 999, "sub": 999, "trail": 0}}))
    st = OUT / "stills"
    subprocess.run([sys.executable, str(SCRIPTS / "render" / "render_scene.py"), "opening", "--variant", "video",
                    "--timing", "/work/timing/hero.json", "--stills", "9.5", "--stills-dir", str(st)], check=True,
                   capture_output=True)
    im = Image.open(st / "opening_009.50.png").convert("RGB")
    # fade the left side into the slide background so the title sits on a clean field
    bg = Image.new("RGB", im.size, (10, 21, 39))
    mask = Image.new("L", im.size, 0)
    g = ImageDraw.Draw(mask)
    w = im.size[0]
    for x in range(w):
        a = min(1.0, max(0.0, (x - 0.30 * w) / (0.28 * w)))
        g.line([(x, 0), (x, im.size[1])], fill=int(255 * a * a * (3 - 2 * a)))
    Image.composite(im, bg, mask.filter(ImageFilter.GaussianBlur(4))).save(OUT / "hero.png")
    # sine / cosine circle panel from the network scene
    subprocess.run([sys.executable, str(SCRIPTS / "render" / "render_scene.py"), "neural_architecture",
                    "--stills", "20", "--stills-dir", str(st)], check=True, capture_output=True)
    Image.open(st / "neural_architecture_020.00.png").crop((330, 300, 1090, 860)).save(OUT / "circle.png")


def data():
    rows = {f"{r['method']}|{r['seed_mode']}": r for r in csv.DictReader(open(RESULTS / "evaluation_metrics.csv"))}
    f = lambda k, m: float(rows[k][m])
    meth = ["residual_mlp", "classical_dls", "hybrid_lm", "lm_only", "baseline_mlp", "baseline_hybrid", "hybrid_dls"]
    ev = {m: {mode: {k: (float(v) if k not in ("method", "seed_mode", "notes") and v != "" else v)
                     for k, v in rows[f"{m}|{mode}"].items()} for mode in ("warm", "zero", "random")} for m in meth}
    summ = read_json(RESULTS / "eval_summary.json")
    base = read_json(RESULTS / "baseline_repro" / "summary.json")
    meta = read_json(BEST_RUN_DIR / "meta.json")
    ds = read_json(DATASET_200K / "meta.json")
    hist = [h for h in csv.DictReader(open(RESULTS / "training_history.csv")) if h["phase"] == "ik"]
    best_ep = [e for e in meta["epochs"] if e["phase"] == "ik"]
    l2 = read_json(L2_META)
    repro = RESULTS / "level2_repro_eval.json"
    inf = read_json(WORK / "sim" / "inference.json")
    ik = read_json(WORK / "sim" / "ik.json")
    lim = read_json(WORK / "sim" / "robot.json")["limits"]
    import math
    out = {
        "eval": ev,
        "full_test": summ["full_test_split"],
        "classifier": summ["collision_classifiers"]["best_run_classifier_test"],
        "params": summ["params"],
        "baseline_repro": {"evaluate_py": base["evaluate_py_unchanged_model"], "dataset": base["dataset"],
                           "final": base["final_epoch"], "secs": base["seconds"], "classifier": base["collision_classifier_val"]},
        "best_run": {"id": meta["id"], "gpu": meta["gpu"], "train_secs": meta["train_secs"], "epochs": meta["params"]["epochs"],
                     "split": meta["split"], "test": meta["metrics"]["test"]},
        "dataset": {"n": ds["n"], "n_colliding": ds["n_colliding"], "secs": ds["secs"]},
        "curves": {"baseline_val_mm": [round(float(h["val_pos_mean_mm"]), 2) for h in hist],
                   "baseline_train_mse": [round(float(h["train_fit_mse"]), 5) for h in hist],
                   "baseline_val_mse": [round(float(h["val_mse"]), 5) for h in hist],
                   "best_val_mm": [round(e["acc"]["val"]["pos_mean_mm"], 2) for e in best_ep]},
        "level2": {"team": l2["evaluation"], "train_secs": l2["train_secs"], "steps": l2["steps_done"],
                   "repro": read_json(repro) if repro.exists() else None},
        "inference": [{"net_mm": c["network"]["pos_err_mm"], "net_deg": c["network"]["orn_err_deg"],
                       "lm_mm": c["hybrid"]["pos_err_mm"], "lm_deg": c["hybrid"]["orn_err_deg"]} for c in inf["cases"]],
        "timing": inf["timing"],
        "ik_branches": ik["n_branches_found"],
        "limits_deg": [[round(math.degrees(a)), round(math.degrees(b))] for a, b in lim],
    }
    def clean(v):                                   # NaN (no timing for the full-split row) is not JSON
        if isinstance(v, float) and v != v:
            return None
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, list):
            return [clean(x) for x in v]
        return v
    (OUT / "data.json").write_text(json.dumps(clean(out), indent=1, allow_nan=False))
    print("data.json written")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    what = sys.argv[1:] or ["posters", "hero", "data"]
    if "posters" in what:
        posters()
    if "hero" in what:
        hero_and_circle()
    if "data" in what:
        data()
