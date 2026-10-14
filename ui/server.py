"""
Neural IK Studio - a local web UI / digital twin for the Level 1 pipeline.

    .venv\\Scripts\\python.exe ui\\server.py          (then open http://127.0.0.1:8765)

What it does
  * Serves a three.js page that shows the robot in 3D. Link poses always come
    from PyBullet forward kinematics (same robot_config.load_robot the project
    uses), so the 3D view is exactly what the dataset / training code "sees".
  * "Generate", "Train" and "Evaluate" run the project's own code, unmodified,
    in a subprocess (see ui/runners/) and stream events back here.
  * Keeps a library in ui/workspace/library/: every generated dataset and
    every trained model gets its own folder + meta.json and is never
    overwritten, so they can always be browsed, re-evaluated and used for
    inference.
  * Streams live CPU / RAM / GPU stats (psutil + nvidia-smi) to the page.

Nothing in the parent project folder is written. "Import from project" only
copies the project's data/ and models/ files into the library.
"""

import argparse
import glob
import hashlib
import json
import mimetypes
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(UI_DIR)
WORKSPACE = os.path.join(UI_DIR, "workspace")
LIB = os.path.join(WORKSPACE, "library")
DS_DIR = os.path.join(LIB, "datasets")
RUNS_DIR = os.path.join(LIB, "runs")
L2_DIR = os.path.join(WORKSPACE, "level2", "policies")   # Level 2 DRL policies (separate from the Level 1 library)
MISSIONS_DIR = os.path.join(WORKSPACE, "missions")          # saved Mission runs (Level 1 + Level 2)
STATIC_DIR = os.path.join(UI_DIR, "static")
RUNNERS_DIR = os.path.join(UI_DIR, "runners")
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, RUNNERS_DIR)

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import pybullet as p  # noqa: E402

import collision_filter  # noqa: E402
import generator_registry as GR  # noqa: E402
import ik_metrics as M  # noqa: E402
import model_registry as MR  # noqa: E402
from libio import read_json, rel, sha256_file, write_json  # noqa: E402
from robot_config import ROBOTS, connect, load_robot  # noqa: E402

NO_WINDOW = 0x08000000 if os.name == "nt" else 0  # don't flash consoles on Windows

for d in (DS_DIR, RUNS_DIR):
    os.makedirs(d, exist_ok=True)


def r5(seq):
    return [round(float(v), 5) for v in seq]


def stamp(t=None):
    return time.strftime("%Y%m%d-%H%M%S", time.localtime(t or time.time()))


# --------------------------------------------------------------------------
# Event hub (Server-Sent Events fan-out)
# --------------------------------------------------------------------------

class Hub:
    def __init__(self):
        self.clients = set()
        self.lock = threading.Lock()

    def subscribe(self):
        q = queue.Queue(maxsize=2000)
        with self.lock:
            self.clients.add(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            self.clients.discard(q)

    def publish(self, ev):
        data = json.dumps(ev)
        with self.lock:
            clients = list(self.clients)
        for q in clients:
            try:
                q.put_nowait(data)
            except queue.Full:
                pass


HUB = Hub()
STATE = {
    "job": None,                 # public dict describing current / last job
    "stats": deque(maxlen=300),  # last 5 min of 1 Hz stats
    "log": deque(maxlen=500),
    "epochs": [],                # epoch_end events of the current / last training job
}
STATE_LOCK = threading.Lock()


# --------------------------------------------------------------------------
# PyBullet forward kinematics service (one DIRECT client, guarded by a lock)
# --------------------------------------------------------------------------

PB_LOCK = threading.RLock()
_BODIES = {}
MESH_FILES = {}          # key -> absolute path (only files PyBullet itself reported)


def pb_init():
    with PB_LOCK:
        connect(gui=False)


def get_robot(name):
    if name not in ROBOTS:
        raise ValueError(f"unknown robot {name}")
    with PB_LOCK:
        if name not in _BODIES:
            body, cfg, limits = load_robot(name)
            collision_filter.apply(body, cfg, limits)     # ignore link pairs that touch by construction
            _BODIES[name] = (body, cfg, limits)
        return _BODIES[name]


def _register_mesh(path):
    """Stable, content-addressed mesh URL key. The browser caches meshes, so a
    key must always mean the same file - numbering meshes in load order made a
    cached URL point at another robot's mesh after a server restart."""
    path = os.path.abspath(path)
    key = hashlib.sha1(f"{path}|{os.path.getmtime(path)}".encode()).hexdigest()[:20]
    MESH_FILES[key] = path
    return key


def robot_model(name):
    with PB_LOCK:
        body, cfg, limits = get_robot(name)
        visuals = []
        for v in p.getVisualShapeData(body):
            entry = {
                "link": v[1], "type": v[2], "dims": r5(v[3]),
                "pos": r5(v[5]), "orn": r5(v[6]), "rgba": r5(v[7]),
            }
            fname = v[4].decode(errors="ignore")
            if v[2] == p.GEOM_MESH and fname:
                mid = _register_mesh(fname)
                ext = os.path.splitext(fname)[1].lower().lstrip(".")
                entry["mesh"] = f"/mesh/{mid}.{ext}"
                entry["ext"] = ext
            visuals.append(entry)
        joints = [
            {"name": n, "index": i, "lo": lo, "hi": hi}
            for n, i, (lo, hi) in zip(cfg["joint_names"], cfg["joint_indices"], limits)
        ]
    return {
        "robot": name, "dof": cfg["dof"], "description": cfg.get("description", ""),
        "ee_link": cfg["ee_link_index"], "joints": joints, "visuals": visuals,
    }


def _link_frames(body):
    """World pose of every link's URDF frame (which is what visual origins are
    expressed in - verified: getVisualShapeData local frames are zero for links
    whose URDF visual origin is zero, while their inertial offsets are not)."""
    frames = {}
    bpos, born = p.getBasePositionAndOrientation(body)          # base COM frame
    dyn = p.getDynamicsInfo(body, -1)
    ipos, iorn = p.invertTransform(dyn[3], dyn[4])
    lpos, lorn = p.multiplyTransforms(bpos, born, ipos, iorn)   # base URDF frame
    frames["-1"] = [r5(lpos), r5(lorn)]
    for j in range(p.getNumJoints(body)):
        st = p.getLinkState(body, j, computeForwardKinematics=True)
        frames[str(j)] = [r5(st[4]), r5(st[5])]
    return frames


def fk(robot, q, collision=True):
    with PB_LOCK:
        body, cfg, _ = get_robot(robot)
        for idx, a in zip(cfg["joint_indices"], q):
            p.resetJointState(body, idx, float(a))
        links = _link_frames(body)
        ee = links[str(cfg["ee_link_index"])]
        out = {"links": links, "ee_pos": ee[0], "ee_orn": ee[1]}
        if collision:
            p.performCollisionDetection()
            cps = p.getContactPoints(bodyA=body, bodyB=body)
            out["colliding"] = len(cps) > 0
            out["collision_links"] = sorted({c[3] for c in cps} | {c[4] for c in cps})
    return out


def quat_angle_deg(q1, q2):
    return float(M.quat_angle_deg(q1, q2)[0])


# --------------------------------------------------------------------------
# Library: datasets
# --------------------------------------------------------------------------

LIB_LOCK = threading.RLock()


def ds_entries():
    out = []
    for d in sorted(os.listdir(DS_DIR)):
        m = read_json(os.path.join(DS_DIR, d, "meta.json"))
        if m:
            out.append(m)
    out.sort(key=lambda m: m.get("created", 0), reverse=True)
    return out


def ds_meta(ds_id):
    if ds_id.startswith("project:"):
        robot = ds_id.split(":", 1)[1]
        path = os.path.join(PROJECT_DIR, "data", f"{robot}_ik_dataset.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"project has no dataset for {robot}")
        return {"id": ds_id, "robot": robot, "source": "project data/ (live, read-only)",
                "path_abs": path, "file": rel(path, PROJECT_DIR), "status": "ready"}
    m = read_json(os.path.join(DS_DIR, ds_id, "meta.json"))
    if not m:
        raise FileNotFoundError(f"no dataset {ds_id}")
    m["path_abs"] = os.path.join(WORKSPACE, m["file"]) if m.get("file") else None
    return m


_DS_CACHE = {}


def load_ds(ds_id):
    path = ds_meta(ds_id)["path_abs"]
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"dataset {ds_id} has no data file (yet)")
    mtime = os.path.getmtime(path)
    hit = _DS_CACHE.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    _DS_CACHE[path] = (mtime, d)
    return d


def _unique_dir(root, name):
    cand, k = name, 2
    while os.path.exists(os.path.join(root, cand)):
        cand, k = f"{name}_{k}", k + 1
    return cand


def register_dataset_file(src, robot, source, move=False):
    """Add an existing .npz to the library (deduplicated by content hash)."""
    with LIB_LOCK:
        sha = sha256_file(src)
        for e in ds_entries():
            if e.get("sha256") == sha:
                if move:
                    os.remove(src)
                return e["id"], False
        with np.load(src) as z:
            d = {k: z[k] for k in ("joints", "joint_limits", "collision_labels", "dof")}
        seed = M.infer_generation_seed(d)
        n = int(len(d["joints"]))
        created = os.path.getmtime(src)
        ds_id = _unique_dir(DS_DIR, f"ds_{stamp(created)}_{robot}_n{n}" + (f"_s{seed}" if seed is not None else ""))
        dest = os.path.join(DS_DIR, ds_id, "data", f"{robot}_ik_dataset.npz")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        (shutil.move if move else shutil.copy2)(src, dest)
        write_json(os.path.join(DS_DIR, ds_id, "meta.json"), {
            "id": ds_id, "robot": robot, "n": n, "seed": seed,
            "seed_source": "recovered from the first sample" if seed is not None else "unknown",
            "dof": int(d["dof"]), "n_colliding": int(d["collision_labels"].sum()), "source": source,
            "file": rel(dest, WORKSPACE), "sha256": sha, "created": created, "status": "ready",
        })
        return ds_id, True


# --------------------------------------------------------------------------
# Library: runs (trained models)
# --------------------------------------------------------------------------

def run_dir(run_id):
    d = os.path.join(RUNS_DIR, run_id)
    if not os.path.isdir(d) or os.path.dirname(os.path.normpath(d)) != os.path.normpath(RUNS_DIR):
        raise FileNotFoundError(f"no run {run_id}")
    return d


def run_meta(run_id):
    m = read_json(os.path.join(run_dir(run_id), "meta.json"))
    if not m:
        raise FileNotFoundError(f"run {run_id} has no meta.json")
    return m


def has_models(run_id, m):
    return bool(m.get("models")) and os.path.isfile(os.path.join(run_dir(run_id), m["models"]["ik"]))


def inference_log(run_id):
    path = os.path.join(run_dir(run_id), "inference_log.jsonl")
    recs = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return recs


def inference_summary(recs, tol=None):
    if not recs:
        return {"n": 0}
    extra = (tol,) if tol else ()
    net = M.summarize([r["pos_err_mm"] / 1000 for r in recs], [r["orn_err_deg"] for r in recs],
                      colliding=[r["pred_self_collision"] for r in recs],
                      limit_viol=[r["limit_violation"] for r in recs], extra_pos_mm=extra)
    truth = [r for r in recs if r.get("joint_err_deg") is not None]
    if truth:
        net["joint_mae_deg"] = float(np.mean([r["joint_err_deg"] for r in truth]))
    net["latency_ms_mean"] = float(np.mean([r["latency_ms"] for r in recs]))
    out = {"n": len(recs), "network": net}
    refined = [r for r in recs if r.get("refined")]
    if refined:
        out["refined"] = M.summarize([r["refined"]["pos_err_mm"] / 1000 for r in refined],
                                     [r["refined"]["orn_err_deg"] for r in refined],
                                     colliding=[bool(r["refined"].get("self_collision")) for r in refined],
                                     limit_viol=[bool(r["refined"].get("limit_violation")) for r in refined],
                                     extra_pos_mm=extra)
        out["refined"]["methods"] = sorted({r["refined"].get("method", "dls") for r in refined})
    return out


def run_summary(m):
    rid = m["id"]
    try:
        n_inf = len(inference_log(rid))
        models = has_models(rid, m)
    except FileNotFoundError:
        n_inf, models = 0, False
    ik_epochs = [e for e in m.get("epochs", []) if e.get("phase") == "ik"]
    return {
        "id": rid, "kind": m.get("kind"), "robot": m.get("robot"), "dataset_id": m.get("dataset_id"),
        "status": m.get("status"), "created": m.get("created"), "epochs_done": len(ik_epochs),
        "params": m.get("params", {}), "metrics": m.get("metrics", {}),
        "collision_metrics": m.get("collision_metrics", {}),
        "evaluations": [{"id": e["id"], "dataset_id": e["dataset_id"], "summary": e["summary"],
                         "overlap": e.get("overlap")} for e in m.get("evaluations", [])],
        "n_inferences": n_inf, "has_models": models, "notes": m.get("notes"),
        "early_stop_result": m.get("early_stop_result"),
        "model": m.get("model") or {"id": MR.BASE_ID, "file": "train_ik.py", "class": "IKNet"},
    }


def run_entries():
    out = []
    for d in os.listdir(RUNS_DIR):
        m = read_json(os.path.join(RUNS_DIR, d, "meta.json"))
        if m:
            out.append(run_summary(m))
    out.sort(key=lambda r: r.get("created") or 0, reverse=True)
    return out


def run_detail(run_id):
    m = run_meta(run_id)
    rd = run_dir(run_id)
    m["has_models"] = has_models(run_id, m)
    m["manifest_data"] = read_json(os.path.join(rd, m["manifest"])) if m.get("manifest") else None
    recs = inference_log(run_id)
    tol = (m.get("params") or {}).get("pos_tol_mm")
    m["inference"] = {"summary": inference_summary(recs, tol), "recent": recs[-200:][::-1]}
    try:
        m["dataset"] = {k: v for k, v in ds_meta(m["dataset_id"]).items() if k != "path_abs"} if m.get("dataset_id") else None
    except FileNotFoundError:
        m["dataset"] = None
    return m


def library():
    runs = run_entries()
    datasets = ds_entries()
    for d in datasets:
        d["used_by"] = [r["id"] for r in runs if r.get("dataset_id") == d["id"]]
    project = []
    for robot in ROBOTS:
        path = os.path.join(PROJECT_DIR, "data", f"{robot}_ik_dataset.npz")
        if os.path.exists(path):
            project.append({"id": f"project:{robot}", "robot": robot, "file": rel(path, PROJECT_DIR),
                            "source": "project data/ (live, read-only)", "created": os.path.getmtime(path)})
    return {"datasets": datasets, "runs": runs, "project_datasets": project}


def _job_uses(kind, item_id):
    """True if the running job reads or writes this dataset / run."""
    with STATE_LOCK:
        job = STATE["job"]
    if not job or job.get("status") != "running":
        return False
    params = job.get("params") or {}
    if kind == "dataset":
        return item_id in (params.get("dataset_id"), params.get("ds_id"))
    if kind == "policy":
        return item_id == params.get("policy_id")
    return item_id == params.get("run_id")


def delete_item(body):
    """Permanently delete a library dataset or run (the UI asks for confirmation first).

    Deleting a dataset with with_runs=True also deletes the runs trained on it; otherwise
    those runs keep their models but lose their test samples and dataset browsing.
    """
    kind, item_id = body.get("kind"), str(body.get("id") or "")
    if kind not in ("dataset", "run", "policy", "mission"):
        raise ValueError("kind must be 'dataset', 'run', 'policy' or 'mission'")
    if item_id.startswith("project:"):
        raise ValueError("project data/ files are read-only and cannot be deleted from the UI")
    root = {"dataset": DS_DIR, "run": RUNS_DIR, "policy": L2_DIR, "mission": MISSIONS_DIR}[kind]
    if not item_id or os.path.basename(item_id) != item_id or item_id in (".", ".."):
        raise ValueError(f"bad {kind} id")
    with LIB_LOCK:
        target = os.path.join(root, item_id)
        if not os.path.isdir(target):
            raise FileNotFoundError(f"no {kind} {item_id}")
        runs = [r["id"] for r in run_entries() if r.get("dataset_id") == item_id] \
            if kind == "dataset" and body.get("with_runs") else []
        for k, i in [(kind, item_id)] + [("run", r) for r in runs]:
            if _job_uses(k, i):
                raise RuntimeError(f"{k} {i} is used by the running job – stop it first")
        deleted = []
        for r in runs:
            shutil.rmtree(os.path.join(RUNS_DIR, r))
            deleted.append({"kind": "run", "id": r})
        shutil.rmtree(target)
        deleted.append({"kind": kind, "id": item_id})
    # drop cached data / models of what was removed
    for path in [p for p in _DS_CACHE if p.startswith(target + os.sep)]:
        _DS_CACHE.pop(path, None)
    gone = {d["id"] for d in deleted if d["kind"] == "run"}
    with _MODEL_LOCK:
        for key in [k for k in _MODELS if k[0] in gone]:
            _MODELS.pop(key, None)
    _log("[library] deleted " + ", ".join(f"{d['kind']} {d['id']}" for d in deleted))
    return {"deleted": deleted}


def overlap_info(run_m, ds_m):
    """Does evaluating this model on this dataset measure generalisation, or
    does the dataset share rows with what the model was trained on?"""
    train_id = run_m.get("dataset_id")
    if not train_id:
        return {"level": "unknown", "note": "training dataset of this model is unknown"}
    if train_id == ds_m["id"]:
        return {"level": "training-data", "note": "this is the dataset the model was trained on (mostly training rows)"}
    try:
        tr = ds_meta(train_id)
    except FileNotFoundError:
        return {"level": "unknown", "note": "training dataset not in the library"}
    if tr.get("sha256") and tr.get("sha256") == ds_m.get("sha256"):
        return {"level": "training-data", "note": "identical file to the training dataset"}
    if tr.get("seed") is None or ds_m.get("seed") is None:
        return {"level": "unknown", "note": "generator seed unknown - rows may overlap the training data"}
    if tr["seed"] == ds_m["seed"]:
        return {"level": "same-seed", "note": f"same generator seed ({tr['seed']}) as the training data: "
                                              "generate() is deterministic, so rows largely coincide"}
    return {"level": "independent", "note": f"independent: generator seed {ds_m['seed']} vs training seed {tr['seed']}"}


# --------------------------------------------------------------------------
# One-time migration of pre-library workspace files + import from project
# --------------------------------------------------------------------------

def _epochs_from_history(path):
    h = read_json(path, {}) or {}
    return [{"phase": "ik", "epoch": i + 1, "epochs": len(h.get("val_mse", [])), "val_mse": v}
            for i, v in enumerate(h.get("val_mse", []))]


def migrate_legacy():
    """ui/workspace/{data,models,renders} from before the library existed ->
    library entries (moved inside ui/workspace; nothing outside is touched)."""
    old_data = os.path.join(WORKSPACE, "data")
    old_models = os.path.join(WORKSPACE, "models")
    old_renders = os.path.join(WORKSPACE, "renders")
    ds_by_robot = {}
    for f in sorted(glob.glob(os.path.join(old_data, "*_ik_dataset.npz"))):
        robot = os.path.basename(f)[: -len("_ik_dataset.npz")]
        ds_by_robot[robot], _ = register_dataset_file(f, robot, "UI (generated before the library existed)", move=True)
        print(f"migrated dataset {f} -> {ds_by_robot[robot]}")
    for ik in sorted(glob.glob(os.path.join(old_models, "*_ik_net.pt"))):
        robot = os.path.basename(ik)[: -len("_ik_net.pt")]
        cands = sorted(glob.glob(os.path.join(old_renders, f"*_{robot}")), reverse=True)
        render_src = next((c for c in cands if os.path.exists(os.path.join(c, "manifest.json"))), None)
        run_id = _unique_dir(RUNS_DIR, os.path.basename(render_src) if render_src else f"legacy_{stamp(os.path.getmtime(ik))}_{robot}")
        rd = os.path.join(RUNS_DIR, run_id)
        os.makedirs(os.path.join(rd, "models"))
        for suffix in ("_ik_net.pt", "_collision_net.pt", "_history.json"):
            src = os.path.join(old_models, robot + suffix)
            if os.path.exists(src):
                shutil.move(src, os.path.join(rd, "models", robot + suffix))
        manifest = None
        if render_src:
            shutil.move(render_src, os.path.join(rd, "renders"))
            man_path = os.path.join(rd, "renders", "manifest.json")
            man = read_json(man_path, {})
            for c in man.get("cases", []):
                for im in c.get("images", []):
                    im["file"] = rel(os.path.join(rd, "renders", os.path.basename(im["file"])), WORKSPACE)
            write_json(man_path, man)
            manifest = "renders/manifest.json"
        write_json(os.path.join(rd, "meta.json"), {
            "id": run_id, "kind": "legacy-ui", "robot": robot, "dataset_id": ds_by_robot.get(robot),
            "status": "done", "created": os.path.getmtime(os.path.join(rd, "models", robot + "_ik_net.pt")),
            "params": {"epochs": len(_epochs_from_history(os.path.join(rd, "models", robot + "_history.json")))},
            "epochs": _epochs_from_history(os.path.join(rd, "models", robot + "_history.json")),
            "metrics": {}, "collision_metrics": {}, "evaluations": [],
            "models": {"ik": f"models/{robot}_ik_net.pt", "collision": f"models/{robot}_collision_net.pt",
                       "history": f"models/{robot}_history.json"},
            "models_sha": sha256_file(os.path.join(rd, "models", robot + "_ik_net.pt")),
            "manifest": manifest,
            "notes": "Trained by the UI before the library existed, on the whole dataset: no held-out test "
                     "split, and train_ik's train/validation split was not recorded. Use 'Evaluate on dataset' "
                     "with a dataset generated from a different seed for an independent test.",
        })
        print(f"migrated model {ik} -> run {run_id}")
    # Render folders left without models: their model files were overwritten
    # by a later training run (the pre-library UI always wrote models/<robot>_*).
    for rdir in sorted(glob.glob(os.path.join(old_renders, "*"))):
        man = read_json(os.path.join(rdir, "manifest.json"))
        if not man:
            continue
        run_id = _unique_dir(RUNS_DIR, os.path.basename(rdir))
        rd = os.path.join(RUNS_DIR, run_id)
        os.makedirs(rd)
        shutil.move(rdir, os.path.join(rd, "renders"))
        for c in man.get("cases", []):
            for im in c.get("images", []):
                im["file"] = rel(os.path.join(rd, "renders", os.path.basename(im["file"])), WORKSPACE)
        write_json(os.path.join(rd, "renders", "manifest.json"), man)
        write_json(os.path.join(rd, "meta.json"), {
            "id": run_id, "kind": "legacy-ui", "robot": man.get("robot"), "dataset_id": None,
            "status": "images only", "created": man.get("created"),
            "params": {"epochs": man.get("epochs"), "collision_epochs": man.get("collision_epochs")},
            "epochs": [], "metrics": {}, "collision_metrics": {}, "evaluations": [], "models": None,
            "manifest": "renders/manifest.json",
            "notes": "Images only: this run's model files were overwritten by a later training run "
                     "before the library existed, so it cannot be evaluated or used for inference.",
        })
        print(f"migrated render-only run {rdir} -> {run_id}")
    for d in (old_models, old_renders, old_data):
        if os.path.isdir(d) and not os.listdir(d):
            os.rmdir(d)
    attach_datasets()


def attach_datasets():
    """Runs whose training dataset is unknown: find the library dataset whose
    rows match the run's tracked training cases (index + joint angles)."""
    for r in os.listdir(RUNS_DIR):
        mp = os.path.join(RUNS_DIR, r, "meta.json")
        m = read_json(mp)
        if not m or m.get("dataset_id") or not m.get("manifest"):
            continue
        man = read_json(os.path.join(RUNS_DIR, r, m["manifest"]), {}) or {}
        cases = [c for c in man.get("cases", []) if c.get("dataset_index") is not None]
        if not cases:
            continue
        for ds in ds_entries():
            if ds.get("robot") != m.get("robot") or ds.get("status") != "ready":
                continue
            try:
                d = load_ds(ds["id"])
            except FileNotFoundError:
                continue
            if all(c["dataset_index"] < len(d["joints"]) and
                   np.allclose(np.degrees(M.wrap(d["joints"][c["dataset_index"]] - np.array(c["q_true"]))), 0, atol=1e-3)
                   for c in cases):
                m["dataset_id"] = ds["id"]
                m["notes"] = (m.get("notes") or "") + f" Training dataset identified by matching its {len(cases)} tracked cases."
                write_json(mp, m)
                print(f"run {r}: training dataset identified as {ds['id']}")
                break


def import_project():
    """Copy (never move) the project's data/ and models/ into the library."""
    added = {"datasets": [], "runs": []}
    with LIB_LOCK:
        for robot in ROBOTS:
            ds_id = None
            data = os.path.join(PROJECT_DIR, "data", f"{robot}_ik_dataset.npz")
            if os.path.exists(data):
                ds_id, new = register_dataset_file(data, robot, "project data/ (copied)")
                if new:
                    added["datasets"].append(ds_id)
            ik = os.path.join(PROJECT_DIR, "models", f"{robot}_ik_net.pt")
            if not os.path.exists(ik):
                continue
            sha = sha256_file(ik)
            if any((read_json(os.path.join(RUNS_DIR, r, "meta.json"), {}) or {}).get("models_sha") == sha
                   for r in os.listdir(RUNS_DIR)):
                continue
            run_id = _unique_dir(RUNS_DIR, f"proj_{stamp(os.path.getmtime(ik))}_{robot}")
            rd = os.path.join(RUNS_DIR, run_id)
            os.makedirs(os.path.join(rd, "models"))
            for suffix in ("_ik_net.pt", "_collision_net.pt", "_history.json"):
                src = os.path.join(PROJECT_DIR, "models", robot + suffix)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(rd, "models", robot + suffix))
            hist = os.path.join(rd, "models", robot + "_history.json")
            write_json(os.path.join(rd, "meta.json"), {
                "id": run_id, "kind": "project-import", "robot": robot, "dataset_id": ds_id,
                "status": "done", "created": os.path.getmtime(ik),
                "params": {"epochs": len(_epochs_from_history(hist))}, "epochs": _epochs_from_history(hist),
                "metrics": {}, "collision_metrics": {}, "evaluations": [],
                "models": {"ik": f"models/{robot}_ik_net.pt", "collision": f"models/{robot}_collision_net.pt",
                           "history": f"models/{robot}_history.json"},
                "models_sha": sha, "manifest": None,
                "notes": "Copied from the project's models/ folder (trained with train_ik.py on the command line). "
                         "Assumed to have been trained on the project's data/ file imported alongside it. "
                         "No held-out test split and no recorded train/validation split - evaluate on a dataset "
                         "generated with a different seed for an independent test.",
            })
            added["runs"].append(run_id)
        attach_datasets()
    return added


# --------------------------------------------------------------------------
# Inference (single target pose, in-process, CPU)
# --------------------------------------------------------------------------

_MODELS = {}
_MODEL_LOCK = threading.Lock()


def load_models(run_id):
    m = run_meta(run_id)
    rd = run_dir(run_id)
    ik_path = os.path.join(rd, m["models"]["ik"])
    key = (run_id, os.path.getmtime(ik_path))
    with _MODEL_LOCK:
        if key not in _MODELS:
            dof = ROBOTS[m["robot"]]["dof"]
            # the run's own network class (modelset snapshot or train_ik.IKNet), eval mode
            ik, col = MR.load_run_models(rd, m, dof, "cpu")
            _MODELS[key] = (ik, col, dof, m)
        return _MODELS[key]


def infer(body):
    import torch
    run_id = body["run_id"]
    ik, col, dof, m = load_models(run_id)
    robot = m["robot"]
    pos = [float(v) for v in body["target_pos"]]
    if body.get("target_rpy_deg") is not None:
        orn = list(p.getQuaternionFromEuler([np.radians(float(a)) for a in body["target_rpy_deg"]]))
    else:
        orn = [float(v) for v in body["target_orn"]]
        nrm = float(np.linalg.norm(orn)) or 1.0
        orn = [v / nrm for v in orn]
    seed = [float(v) for v in body["seed_q"]]
    q_true = body.get("q_true")
    x = torch.tensor([pos + orn + seed], dtype=torch.float32)

    with torch.no_grad():
        ik(x)                                        # warm-up (first call allocates)
        t0 = time.perf_counter()
        raw = ik(x)
        q_pred = ik.decode_angles(raw)[0].numpy()
        latency = (time.perf_counter() - t0) * 1000
        col_prob = float(torch.sigmoid(col(torch.tensor(q_pred[None], dtype=torch.float32))).reshape(-1)[0].item()) if col else None

    f_pred = fk(robot, q_pred)
    _, _, limits = get_robot(robot)
    rec = {
        "t": time.time(), "run_id": run_id, "source": body.get("source", "custom"),
        "seed_mode": body.get("seed_mode"), "dataset_id": body.get("dataset_id"), "dataset_index": body.get("dataset_index"),
        "target_pos": r5(pos), "target_orn": r5(orn), "seed_q": r5(seed), "q_true": r5(q_true) if q_true else None,
        "q_pred": r5(q_pred), "latency_ms": latency,
        "pos_err_mm": float(np.linalg.norm(np.array(f_pred["ee_pos"]) - np.array(pos)) * 1000),
        "orn_err_deg": quat_angle_deg(f_pred["ee_orn"], orn),
        "joint_err_deg": float(np.degrees(np.abs(M.wrap(q_pred - np.array(q_true)))).mean()) if q_true else None,
        "pred_self_collision": bool(f_pred["colliding"]),
        "collision_net_prob": col_prob,
        "limit_violation": bool(M.limit_violations(q_pred[None], limits)[0]),
    }
    out = {"record": rec, "fk_pred": f_pred}

    method = body.get("refine_method") or ("dls" if int(body.get("refine_iters", 0)) > 0 else "none")
    iters = int(body.get("refine_iters", 0))
    if method == "lm":
        # network + numerical refinement (refine/lm_solver.py, CPU so it never competes with a
        # training job for the GPU): LM from the network output and the seed, random restarts
        # if still outside tolerance, then PyBullet checks the result
        from refine import lm_solver
        st = _refine_settings({"iters": iters or 30, "restarts": int(body.get("refine_restarts", 16)),
                               "use_seed": True, "tol_mm": float(body.get("refine_tol_mm", 1.0)),
                               "tol_deg": float(body.get("refine_tol_deg", 1.0))})
        chain = M.chain_for(robot, "cpu")
        t0 = time.perf_counter()
        with torch.no_grad():
            res = lm_solver.solve(chain, np.array([pos]), np.array([orn]), [q_pred[None], np.array([seed])],
                                  ref=np.array([seed]), iters=st["iters"], restarts=st["restarts"],
                                  tol_mm=st["tol_mm"], tol_deg=st["tol_deg"])
        q_ref = res["q"][0]
        f_ref = fk(robot, q_ref)
        pe = float(np.linalg.norm(np.array(f_ref["ee_pos"]) - np.array(pos)) * 1000)
        oe = quat_angle_deg(f_ref["ee_orn"], orn)
        lim_bad = bool(M.limit_violations(q_ref[None], limits)[0])
        w = int(res["winner"][0])
        if st["restarts"] and (f_ref["colliding"] or lim_bad or pe > st["tol_mm"] or oe > st["tol_deg"]):
            # another IK branch from fresh random starts: first one PyBullet confirms collision-free
            with torch.no_grad():
                cand = lm_solver.random_candidates(chain, np.array([pos]), np.array([orn]),
                                                   n_random=max(32, st["restarts"]), iters=max(50, st["iters"]))
            score = cand["pos_err_mm"][:, 0] / st["tol_mm"] + cand["orn_err_deg"][:, 0] / st["tol_deg"]
            for k in np.argsort(score):
                if cand["pos_err_mm"][k, 0] > st["tol_mm"] or cand["orn_err_deg"][k, 0] > st["tol_deg"]:
                    break
                qc = cand["q"][k, 0]
                fc = fk(robot, qc)
                pc = float(np.linalg.norm(np.array(fc["ee_pos"]) - np.array(pos)) * 1000)
                oc = quat_angle_deg(fc["ee_orn"], orn)
                if not fc["colliding"] and not M.limit_violations(qc[None], limits)[0] and pc <= st["tol_mm"] and oc <= st["tol_deg"]:
                    q_ref, f_ref, pe, oe, lim_bad, w = qc, fc, pc, oc, False, M.RETRY
                    break
        t_ref = (time.perf_counter() - t0) * 1000
        rec["refined"] = {
            "method": "lm", "iters": st["iters"], "restarts": st["restarts"], "q": r5(q_ref), "ms": t_ref,
            "pos_err_mm": pe, "orn_err_deg": oe, "self_collision": bool(f_ref["colliding"]),
            "limit_violation": lim_bad,
            "winner": ("collision-free retry" if w == M.RETRY else ["network", "seed"][w] if w < 2 else f"random restart {w - 1}"),
            "success": bool(pe <= st["tol_mm"] and oe <= st["tol_deg"] and not lim_bad and not f_ref["colliding"]),
            "tol_mm": st["tol_mm"], "tol_deg": st["tol_deg"],
        }
        out["fk_refined"] = f_ref
    elif method == "dls" and iters > 0:
        with PB_LOCK:
            b, cfg, _ = get_robot(robot)
            for idx, a in zip(cfg["joint_indices"], q_pred):
                p.resetJointState(b, idx, float(a))
            movable = [j for j in range(p.getNumJoints(b)) if p.getJointInfo(b, j)[2] != p.JOINT_FIXED]
            cur = [p.getJointState(b, j)[0] for j in movable]
            t0 = time.perf_counter()
            sol = p.calculateInverseKinematics(b, cfg["ee_link_index"], pos, orn, currentPositions=cur,
                                               maxNumIterations=iters, residualThreshold=1e-5)
            t_ref = (time.perf_counter() - t0) * 1000
            q_ref = np.array([sol[movable.index(j)] for j in cfg["joint_indices"]])
        f_ref = fk(robot, q_ref)
        rec["refined"] = {
            "method": "dls", "iters": iters, "q": r5(q_ref), "ms": t_ref,
            "pos_err_mm": float(np.linalg.norm(np.array(f_ref["ee_pos"]) - np.array(pos)) * 1000),
            "orn_err_deg": quat_angle_deg(f_ref["ee_orn"], orn),
            "self_collision": bool(f_ref["colliding"]),
            "limit_violation": bool(M.limit_violations(q_ref[None], limits)[0]),
        }
        out["fk_refined"] = f_ref

    if body.get("log", True):
        with open(os.path.join(run_dir(run_id), "inference_log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    return out


# --------------------------------------------------------------------------
# Datasets: samples for the browser
# --------------------------------------------------------------------------

_SPLIT_CACHE = {}


def split_membership(ds_id, i):
    out = []
    for d in os.listdir(RUNS_DIR):
        m = read_json(os.path.join(RUNS_DIR, d, "meta.json"))
        if not m or m.get("dataset_id") != ds_id:
            continue
        path = os.path.join(RUNS_DIR, d, "split.npz")
        if not os.path.exists(path):
            out.append({"run_id": d, "split": "whole dataset used (split not recorded)"})
            continue
        mt = os.path.getmtime(path)
        if _SPLIT_CACHE.get(path, (None,))[0] != mt:
            with np.load(path) as z:
                _SPLIT_CACHE[path] = (mt, {k: set(z[k].tolist()) for k in z.files})
        sp = _SPLIT_CACHE[path][1]
        if i in sp.get("test", ()):
            name = "TEST (held out)"
        elif i in sp.get("ik_train", ()):
            name = "train"
        elif i in sp.get("ik_val", ()):
            name = "validation"
        elif i in sp.get("col_train", ()):
            name = "collision-net train only (self-colliding)"
        elif i in sp.get("col_val", ()):
            name = "collision-net validation only (self-colliding)"
        else:
            name = "?"
        out.append({"run_id": d, "split": name})
    return out


def dataset_sample(ds_id, i):
    d = load_ds(ds_id)
    n = len(d["joints"])
    i = max(0, min(n - 1, int(i)))
    q = d["joints"][i]
    res = fk(ds_meta(ds_id)["robot"], q)
    res.update({
        "index": i, "n": n, "q": r5(q),
        "pos": r5(d["positions"][i]), "orn": r5(d["orientations"][i]),
        "label": int(d["collision_labels"][i]),
        "membership": split_membership(ds_id, i),
    })
    return res


def dataset_cloud(ds_id, max_points=6000):
    d = load_ds(ds_id)
    n = len(d["positions"])
    idx = np.linspace(0, n - 1, min(n, max_points)).astype(int)
    return {
        "positions": np.round(d["positions"][idx], 4).reshape(-1).tolist(),
        "labels": d["collision_labels"][idx].astype(int).tolist(),
        "n": n,
    }


# --------------------------------------------------------------------------
# Jobs (generate / train / evaluate) - one subprocess at a time
# --------------------------------------------------------------------------

JOB_LOCK = threading.Lock()
_PROC = {"proc": None, "ps": None, "meta_path": None}
RUNNER_FILES = {"generate": "gen_runner.py", "train": "train_runner.py", "evaluate": "eval_runner.py",
                "l2train": "l2_runner.py", "l2eval": "l2_runner.py"}


def _set_job(**kw):
    with STATE_LOCK:
        job = dict(STATE["job"] or {})
        job.update(kw)
        STATE["job"] = job
    HUB.publish({"type": "job", "job": job})


def _log(line):
    with STATE_LOCK:
        STATE["log"].append(line)
    HUB.publish({"type": "log", "line": line})


def _enrich(ev):
    """Attach forward-kinematics link poses so the browser can draw the robot."""
    t = ev.get("type")
    robot = ev.get("robot")
    try:
        if t == "gen_sample":
            ev["fk"] = fk(robot, ev["q"])
        elif t == "train_sample":
            ev["fk_true"] = fk(robot, ev["q_true"])
            if ev.get("q_pred") is not None:
                fp = fk(robot, ev["q_pred"])
                ev["fk_pred"] = fp
                if ev.get("target_pos") is not None:
                    ev["pos_err_mm"] = float(np.linalg.norm(
                        np.array(fp["ee_pos"]) - np.array(ev["target_pos"]))) * 1000
                    ev["orn_err_deg"] = quat_angle_deg(fp["ee_orn"], ev["target_orn"])
    except Exception as e:
        ev["fk_error"] = str(e)
    return ev


def _reader(proc, kind):
    for raw in proc.stdout:
        line = raw.rstrip("\r\n")
        if line.startswith("@@EV "):
            try:
                ev = json.loads(line[5:])
            except json.JSONDecodeError:
                _log(line)
                continue
            ev = _enrich(ev)
            t = ev.get("type")
            if t == "epoch_end":
                with STATE_LOCK:
                    STATE["epochs"].append(ev)
            if t in ("gen_start", "gen_sample", "train_sample", "epoch_end", "phase", "eval_start", "eval_progress"):
                _set_job(progress={k: ev.get(k) for k in ("phase", "epoch", "epochs", "batch", "nb", "i", "n") if k in ev})
            elif t in ("l2_update", "l2_eval_progress"):
                _set_job(progress={k: ev.get(k) for k in ("update", "updates", "steps", "success_pct", "collision_pct", "scenario", "i", "n") if k in ev})
            HUB.publish(ev)
        elif line.strip() and "pybullet build time" not in line:
            _log(line)
    code = proc.wait()
    with STATE_LOCK:
        status = (STATE["job"] or {}).get("status")
    final = "stopped" if status == "stopped" else ("done" if code == 0 else "failed")
    if status != "stopped":
        _set_job(status=final, exit_code=code, ended=time.time())
    mp = _PROC.get("meta_path")
    if mp and final != "done":
        if kind == "generate":
            # a failed / stopped generation leaves no usable data - drop its library entry
            shutil.rmtree(os.path.dirname(mp), ignore_errors=True)
        else:
            m = read_json(mp)
            if m and m.get("status") in ("running", "generating", "evaluating"):
                m["status"] = final
                write_json(mp, m)
    _log(f"[{kind}] finished with exit code {code}")
    if kind == "generate" and final == "done":
        attach_datasets()
    _PROC.update(proc=None, ps=None, meta_path=None)
    HUB.publish({"type": "library_changed"})


def start_job(kind, params, cwd, meta_path=None):
    with JOB_LOCK:
        if _PROC["proc"] is not None and _PROC["proc"].poll() is None:
            raise RuntimeError("a job is already running")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(
            [sys.executable, "-u", os.path.join(RUNNERS_DIR, RUNNER_FILES[kind]), json.dumps(params)],
            cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
            creationflags=NO_WINDOW,
        )
        _PROC.update(proc=proc, meta_path=meta_path)
        try:
            _PROC["ps"] = psutil.Process(proc.pid)
            _PROC["ps"].cpu_percent(None)
        except psutil.Error:
            _PROC["ps"] = None
        with STATE_LOCK:
            STATE["job"] = None
            if kind == "train":
                STATE["epochs"] = []
        _set_job(kind=kind, params=params, status="running", pid=proc.pid, started=time.time(), progress={})
        _log(f"[{kind}] started: {json.dumps({k: v for k, v in params.items() if k not in ('workspace', 'dataset_file', 'run_dir')})}")
        threading.Thread(target=_reader, args=(proc, kind), daemon=True).start()


def stop_job():
    proc = _PROC["proc"]
    if proc is None or proc.poll() is not None:
        return False
    _set_job(status="stopped", ended=time.time())
    try:
        for child in psutil.Process(proc.pid).children(recursive=True):
            child.kill()
    except psutil.Error:
        pass
    proc.kill()
    return True


def job_generate(body):
    robot, n, seed = body["robot"], int(body.get("n", 20000)), int(body.get("seed", 0))
    if robot not in ROBOTS:
        raise ValueError(f"unknown robot {robot}")
    if n < 1:
        raise ValueError("n must be at least 1")
    cp = body.get("collision_pct")
    cp = None if cp in (None, "", "natural") else float(cp)   # None = natural split from generate()
    if cp is not None and not 0 <= cp <= 100:
        raise ValueError("collision percentage must be between 0 and 100")
    gen_id = body.get("generator") or GR.BASE_ID
    gen = next((g for g in GR.list_generators() if g["id"] == gen_id), None)
    if gen is None:
        raise ValueError(f"unknown generator {gen_id!r}")
    if not gen["ok"]:
        raise ValueError(f"generator {gen_id} cannot be used: {gen['error']}")
    # generator options = generate()'s own keyword parameters (see generator_registry.describe)
    given = body.get("gen_opts") or {}
    known = {o["name"]: o for o in gen.get("options", [])}
    unknown = sorted(set(given) - set(known))
    if unknown:
        raise ValueError(f"{gen['file']} has no option(s) {', '.join(unknown)}")
    cast = {"bool": bool, "int": int, "float": float, "str": str}
    gen_opts = {}
    for name, v in given.items():
        try:
            gen_opts[name] = cast[known[name]["type"]](v)
        except (TypeError, ValueError):
            raise ValueError(f"generator option {name}: expected a {known[name]['type']} value") from None
    ignore_pairs = bool(body.get("ignore_structural_pairs")) and bool(gen.get("supports_ignore_pairs"))
    tag = f"_{gen['tag']}" if gen.get("tag") else ""
    with LIB_LOCK:
        ds_id = _unique_dir(DS_DIR, f"ds_{stamp()}_{robot}_n{n}_s{seed}" + (f"_c{cp:g}" if cp is not None else "") + tag)
        d = os.path.join(DS_DIR, ds_id)
        os.makedirs(d)
        mp = os.path.join(d, "meta.json")
        write_json(mp, {"id": ds_id, "robot": robot, "n": n, "seed": seed, "source": "ui",
                        "split": {"mode": "natural"} if cp is None else {"mode": "custom", "collision_pct_requested": cp},
                        "generator": {"id": gen_id, "file": gen["file"]},
                        "status": "generating", "created": time.time()})
    start_job("generate", {"robot": robot, "n": n, "seed": seed, "delay_ms": float(body.get("delay_ms", 0)),
                           "collision_pct": cp, "generator": gen_id, "gen_opts": gen_opts,
                           "ignore_structural_pairs": ignore_pairs, "ds_id": ds_id, "workspace": WORKSPACE},
              cwd=d, meta_path=mp)
    return ds_id


def job_train(body):
    robot = body["robot"]
    ds = ds_meta(body["dataset_id"])
    if ds.get("robot") != robot or ds.get("status") != "ready" or body["dataset_id"].startswith("project:"):
        raise ValueError("pick a ready library dataset of this robot (use 'Import from project' for project data)")
    if ds.get("n") is not None and ds.get("n_colliding") is not None:
        n_free = ds["n"] - ds["n_colliding"]
        if n_free < 200:
            raise ValueError(f"this dataset has only {n_free} collision-free samples - the IK network trains "
                             "only on collision-free movements, so generate one with a lower collision %")
        if ds["n_colliding"] == 0:
            _log("[train] note: dataset has no colliding samples - the collision classifier sees only one class")
    with LIB_LOCK:
        run_id = _unique_dir(RUNS_DIR, f"{stamp()}_{robot}")
        d = os.path.join(RUNS_DIR, run_id)
        os.makedirs(d)
        mp = os.path.join(d, "meta.json")
        write_json(mp, {"id": run_id, "kind": "ui-train", "robot": robot, "dataset_id": ds["id"],
                        "status": "running", "created": time.time()})
    model_id = body.get("model_id") or MR.BASE_ID
    entry = next((e for e in MR.list_models(ROBOTS[robot]["dof"], robot) if e["id"] == model_id), None)
    if entry is None:
        raise ValueError(f"unknown model {model_id!r} - put model files in {MR._rel(MR.MODELSET_DIR)}/")
    if not entry["ok"]:
        raise ValueError(f"model {model_id} cannot be trained: {entry['error']}")
    pos_tol = float(body.get("pos_tol_mm", 50))
    if not 0 < pos_tol <= 5000:
        raise ValueError("position tolerance must be between 0 and 5000 mm")
    es = body.get("early_stop")
    if es:
        es = {"metric": es.get("metric", "pct_within"), "target_pct": float(es.get("target_pct", 90)),
              "split": es.get("split", "train"), "patience": int(es.get("patience", 2))}
        if es["metric"] not in ("pct_within", "mean_error") or es["split"] not in ("train", "val"):
            raise ValueError("early stop: metric must be pct_within / mean_error, split train / val")
        if not 0 < es["target_pct"] <= 100 or es["patience"] < 1:
            raise ValueError("early stop: target % must be in (0, 100] and epochs in a row >= 1")
    params = {
        "robot": robot, "run_id": run_id, "dataset_id": ds["id"], "dataset_file": ds["path_abs"],
        "workspace": WORKSPACE, "model_id": model_id, "pos_tol_mm": pos_tol, "early_stop": es or None,
        "epochs": int(body.get("epochs", 60)),
        "collision_epochs": int(body.get("collision_epochs", 15)), "cases": int(body.get("cases", 8)),
        "render_every": int(body.get("render_every", 5)), "sample_hz": float(body.get("sample_hz", 6)),
        "test_frac": float(body.get("test_frac", 0.15)), "split_seed": int(body.get("split_seed", 2024)),
    }
    start_job("train", params, cwd=d, meta_path=mp)
    return run_id


def job_evaluate(body):
    run_id = body["run_id"]
    m = run_meta(run_id)
    if not has_models(run_id, m):
        raise ValueError("this run has no saved model")
    ds = ds_meta(body["dataset_id"])
    if ds.get("robot") != m["robot"]:
        raise ValueError(f"dataset is for {ds.get('robot')}, model is for {m['robot']}")
    seed_mode = body.get("seed_mode") or "noisy"
    if seed_mode not in M.SEED_MODES:
        raise ValueError(f"seed mode must be one of {', '.join(M.SEED_MODES)}")
    params = {"run_id": run_id, "run_dir": run_dir(run_id), "robot": m["robot"], "dataset_id": ds["id"],
              "dataset_file": ds["path_abs"], "workspace": WORKSPACE, "seed_mode": seed_mode,
              "refine": _refine_settings(body.get("refine")),
              "max_samples": int(body.get("max_samples", 0)), "overlap": overlap_info(m, ds)}
    start_job("evaluate", params, cwd=run_dir(run_id))


# --------------------------------------------------------------------------
# Level 2: DRL re-routing policies (level2/ package). Kept apart from Level 1.
# --------------------------------------------------------------------------

def l2_meta(pid):
    d = os.path.join(L2_DIR, pid)
    if os.path.basename(pid) != pid or not os.path.isdir(d):
        raise FileNotFoundError(f"no policy {pid}")
    m = read_json(os.path.join(d, "meta.json")) or {"id": pid}
    m["has_policy"] = os.path.isfile(os.path.join(d, "policy.pt"))
    return m


def l2_policies():
    os.makedirs(L2_DIR, exist_ok=True)
    out = []
    for d in os.listdir(L2_DIR):
        try:
            m = l2_meta(d)
        except FileNotFoundError:
            continue
        m.pop("history", None)
        out.append(m)
    out.sort(key=lambda m: m.get("created") or 0, reverse=True)
    return out


def l2_detail(pid):
    m = l2_meta(pid)
    m["history"] = read_json(os.path.join(L2_DIR, pid, "history.json")) or []
    return m


def job_l2_train(body):
    robot = body.get("robot", "xarm6")
    if robot not in ("xarm6", "kuka_iiwa"):
        raise ValueError("Level 2 supports xarm6 and kuka_iiwa")
    steps = int(body.get("steps", 3_000_000))
    if not 50_000 <= steps <= 50_000_000:
        raise ValueError("steps must be between 50,000 and 50,000,000")
    os.makedirs(L2_DIR, exist_ok=True)
    pid = _unique_dir(L2_DIR, f"l2_{stamp()}_{robot}")
    d = os.path.join(L2_DIR, pid)
    os.makedirs(d)
    params = {"mode": "train", "policy_id": pid, "policy_dir": d, "robot": robot, "steps": steps,
              "envs": int(body.get("envs", 80)), "seed": int(body.get("seed", 0)), "eval_n": int(body.get("eval_n", 60)),
              "device": body.get("device") or None}
    mp = os.path.join(d, "meta.json")
    write_json(mp, {"id": pid, "robot": robot, "status": "running", "created": time.time(),
                    "params": {k: v for k, v in params.items() if k not in ("policy_dir", "mode")}})
    start_job("l2train", params, cwd=d, meta_path=mp)
    return pid


def job_l2_eval(body):
    m = l2_meta(body["id"])
    if not m["has_policy"]:
        raise ValueError("this policy has no policy.pt yet")
    params = {"mode": "eval", "policy_id": m["id"], "policy_dir": os.path.join(L2_DIR, m["id"]),
              "robot": m["robot"], "n": max(10, min(500, int(body.get("n", 100))))}
    start_job("l2eval", params, cwd=params["policy_dir"])


_L2_ENVS, _L2_LOCK = {}, threading.Lock()


def _l2_env(robot, pol):
    """Cached Level 2 env in the settings the policy was trained with (older policies: straight-line base).
    Call with _L2_LOCK held."""
    from level2.env import ReachAvoidEnv
    env_cfg = dict({"base": "straight"}, **(pol.meta.get("env_cfg") or {})) if pol else {}
    key = (robot, json.dumps(env_cfg, sort_keys=True))
    if key not in _L2_ENVS:
        _L2_ENVS[key] = ReachAvoidEnv(robot, seed=0, **env_cfg)
    return _L2_ENVS[key]


def l2_rollout(body):
    """Record one episode for the 3D playback: the policy and the straight-line baseline on the same scenario."""
    from level2.env import ReachAvoidEnv
    from level2.evaluate import SCENARIOS, run_episode
    from level2.ppo import NumpyPolicy
    pid = body.get("id")
    pol = NumpyPolicy(os.path.join(L2_DIR, pid, "policy.pt")) if pid else None
    robot = pol.meta["robot"] if pol else body.get("robot", "xarm6")
    name = body.get("scenario", "combined")
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name}")
    seed = int(body.get("seed", 0))
    with _L2_LOCK:
        env = _l2_env(robot, pol)
        out = {"robot": robot, "scenario": name, "seed": seed}
        for label, ctrl in (("policy", pol), ("baseline", None), ("apf", "apf")):
            if label == "policy" and pol is None:
                continue
            ep = run_episode(env, ctrl, seed, SCENARIOS[name], record=True)
            out[label] = ep
        # start / goal poses for drawing (same seed -> same start and goal for both controllers)
        env.reset(seed=seed, scenario=SCENARIOS[name])
        out["start_links"] = env.link_frames()
        env._set_q(env.q_goal)
        out["goal_links"] = env.link_frames()
        out["goal_ee"] = [round(float(v), 4) for v in env.ee_goal]
        env._set_q(env.q)
    return out


# --------------------------------------------------------------------------
# Mission = Level 1 + Level 2: target pose -> IK (Level 1 run) -> trajectory estimate -> DRL execution
# --------------------------------------------------------------------------

MISSION_SCENARIOS = {
    "dynamic":  {"static": True, "sudden": True, "slip": True, "moving": True},   # everything, obstacles drift
    "combined": {"static": True, "sudden": True, "slip": True},
    "sudden":   {"static": False, "sudden": True, "slip": False},
    "static":   {"static": True, "sudden": False, "slip": False},
    "slip":     {"static": False, "sudden": False, "slip": True},
    "free":     {"static": False, "sudden": False, "slip": False},                 # no obstacles: pure IK + motion
}


def _same_pose_near(q, ref, lim):
    """Shift each joint by multiples of 360 deg (same pose) to the copy inside the limits closest to ref."""
    out = np.array(q, float)
    for j in range(len(out)):
        cands = [out[j] + 2 * np.pi * k for k in range(-2, 3) if lim[j, 0] - 1e-9 <= out[j] + 2 * np.pi * k <= lim[j, 1] + 1e-9]
        if cands:
            out[j] = min(cands, key=lambda v: abs(v - ref[j]))
    return out


def _ee_pose(env):
    st = p.getLinkState(env.body, env.ee, computeForwardKinematics=True, physicsClientId=env.cid)
    return np.array(st[4]), np.array(st[5])


def _final_approach(env, q_goal, ep=None, dq=0.01, max_steps=40):
    """After the agent arrives (within 2 cm / 0.06 rad), servo exactly onto the IK goal in small,
    collision-checked steps - like a controller's final settling. Recorded frames are appended to ep."""
    steps, blocked = 0, False
    while steps < max_steps and np.max(np.abs(q_goal - env.q)) > 1e-5:
        q = env.q + np.clip(q_goal - env.q, -dq, dq)
        env._set_q(q)
        env._update_clearances()
        clear = min((v[0] for v in env._clr.values()), default=0.5)
        if clear <= 0.0 or env._self_collision():
            env._set_q(env.q)                     # would touch something: stay where the agent stopped
            blocked = True
            break
        env.q = q
        steps += 1
        if ep is not None and "frames" in ep:
            ep["frames"].append({"t": env.t + steps, "q": [round(float(v), 5) for v in env.q], "links": env.link_frames(),
                                 "obstacles": env.obstacles(), "clearance": round(float(clear), 4), "phase": "final approach"})
    return {"steps": steps, "blocked": blocked}


def mission(body, record=True, save=False):
    from level2.evaluate import run_episode
    from level2.ppo import NumpyPolicy
    l1, pid = body["l1_run"], body["l2_policy"]
    pol = NumpyPolicy(os.path.join(L2_DIR, pid, "policy.pt"))
    robot = pol.meta["robot"]
    m1 = run_meta(l1)
    if m1["robot"] != robot:
        raise ValueError(f"the Level 1 run is for {m1['robot']} but the Level 2 agent is for {robot}")
    name = body.get("scenario", "dynamic")
    if name not in MISSION_SCENARIOS:
        raise ValueError(f"unknown scenario {name}")
    sc, seed = MISSION_SCENARIOS[name], int(body.get("seed", 0))
    tgt = body.get("target") or {"mode": "random"}

    # 1) start configuration (the arm's current joints) and the target pose
    with _L2_LOCK:
        env = _l2_env(robot, pol)
        env.reset(seed=seed, scenario={"static": False, "sudden": False, "slip": False})
        q_start = env.q.copy()
        if tgt.get("mode") == "custom":
            pos = [float(v) for v in tgt["pos"]]
            orn = list(p.getQuaternionFromEuler([np.radians(float(a)) for a in tgt["rpy_deg"]]))
        else:                                        # a reachable target: the pose of a random collision-free configuration
            env._set_q(env.q_goal)
            pe, oe = _ee_pose(env)
            pos, orn = [float(v) for v in pe], [float(v) for v in oe]
            env._set_q(env.q)
        lim = env.lim.copy()
        start_links = env.link_frames()

    # 2) Level 1: neural IK + LM refinement, seeded with the current joints
    t0 = time.perf_counter()
    res = infer({"run_id": l1, "target_pos": pos, "target_orn": orn, "seed_q": [float(v) for v in q_start],
                 "refine_method": "lm", "refine_iters": 30, "refine_restarts": 16, "log": False, "source": "mission"})
    rec, ref = res["record"], res["record"].get("refined") or {}
    q_ik = _same_pose_near(ref.get("q") or rec["q_pred"], q_start, lim)
    ik = {"run": l1, "net_pos_err_mm": rec["pos_err_mm"], "net_orn_err_deg": rec["orn_err_deg"], "net_ms": rec["latency_ms"],
          "pos_err_mm": ref.get("pos_err_mm", rec["pos_err_mm"]), "orn_err_deg": ref.get("orn_err_deg", rec["orn_err_deg"]),
          "ms": (time.perf_counter() - t0) * 1000, "self_collision": ref.get("self_collision", rec["pred_self_collision"]),
          "limit_violation": ref.get("limit_violation", rec["limit_violation"]), "winner": ref.get("winner"),
          "q": [round(float(v), 5) for v in q_ik]}
    ik["success"] = bool(ik["pos_err_mm"] <= 1.0 and ik["orn_err_deg"] <= 1.0 and not ik["self_collision"] and not ik["limit_violation"])
    out = {"robot": robot, "scenario": name, "seed": seed, "target": {"mode": tgt.get("mode", "random"), "pos": pos, "orn": orn},
           "start_links": start_links, "ik": ik}
    if not ik["success"]:
        out["verdict"] = "IK failed – no collision-free joint solution within 1 mm / 1°; the arm does not move."
        return _save_mission(out, body) if save else out

    with _L2_LOCK:
        env = _l2_env(robot, pol)
        # 3) trajectory estimate: the straight joint-space path to the IK goal, checked against the obstacles
        #    known at the start and for self-collision (what a planner would predict before moving)
        env.reset(seed=seed, scenario=sc, start=q_start, goal=q_ik)
        planned = []
        for t in np.linspace(0, 1, 40):
            env._set_q(q_start + t * (q_ik - q_start))
            d = min((env._clearance(k)[0] for k in range(len(env.obs_bodies)) if env.active[k]), default=0.5)
            planned.append({"p": [round(float(v), 4) for v in env._ee()], "self": bool(env._self_collision()), "clear": round(float(d), 4)})
        env._set_q(q_ik)
        goal_links = env.link_frames()
        env._set_q(env.q)
        pts = np.array([w["p"] for w in planned])
        out["planned"] = planned
        out["estimate"] = {
            "waypoints": len(planned), "joint_path_rad": float(np.linalg.norm(q_ik - q_start)),
            "ee_path_m": float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))),
            "obstacle_conflicts": sum(w["clear"] <= 0 for w in planned), "near_misses": sum(0 < w["clear"] < 0.05 for w in planned),
            "self_collisions": sum(w["self"] for w in planned), "min_clearance_m": min(w["clear"] for w in planned),
            "obstacles_at_start": int(env.active.sum()),
        }
        out["goal_links"], out["goal_ee"] = goal_links, [round(float(v), 4) for v in env.ee_goal]

        # 4) execution: DRL agent, plus the two classical controllers on the identical mission
        out["episodes"] = {}
        for label, ctrl in (("policy", pol), ("baseline", None), ("apf", "apf")):
            ep = run_episode(env, ctrl, seed, sc, record=record, start=q_start, goal=q_ik)
            if ep["outcome"] == "success":
                ep["final_approach"] = _final_approach(env, q_ik, ep if record else None)
            pe, oe = _ee_pose(env)                       # 5) final gripper pose vs. the original target
            ep["final_pos_err_mm"] = float(np.linalg.norm(pe - np.array(pos)) * 1000)
            ep["final_orn_err_deg"] = quat_angle_deg(list(oe), orn)
            out["episodes"][label] = ep
    a = out["episodes"]["policy"]
    out["verdict"] = ("Target reached" if a["outcome"] == "success" else
                      f"Not reached ({a['outcome']}{' – ' + a['collision'] if a.get('collision') else ''})")
    return _save_mission(out, body) if save else out


def _ep_summary(e):
    return {k: e.get(k) for k in ("outcome", "collision", "steps", "min_clearance_m", "final_pos_err_mm", "final_orn_err_deg")} | \
        {"events": [x["type"] for x in e.get("events", [])]}


def _save_mission(out, body, kind="single"):
    """Every mission / statistics run gets its own folder in ui/workspace/missions/ (never overwritten)."""
    os.makedirs(MISSIONS_DIR, exist_ok=True)
    robot = out.get("robot") or "robot"
    tag = f"_s{out['seed']}" if kind == "single" else f"_n{out['n']}"
    mid = _unique_dir(MISSIONS_DIR, f"{'ms' if kind == 'single' else 'mb'}_{stamp()}_{robot}_{out.get('scenario', '')}{tag}")
    d = os.path.join(MISSIONS_DIR, mid)
    os.makedirs(d)
    out["id"] = mid
    meta = {"id": mid, "kind": kind, "created": time.time(), "robot": robot, "scenario": out.get("scenario"),
            "l1_run": body.get("l1_run"), "l2_policy": body.get("l2_policy"),
            "target": (body.get("target") or {}).get("mode", "random")}
    if kind == "single":
        meta.update(seed=out["seed"], verdict=out.get("verdict"), ik_pos_err_mm=out["ik"]["pos_err_mm"], ik_success=out["ik"]["success"],
                    conflicts=(out.get("estimate") or {}).get("obstacle_conflicts"),
                    episodes={k: _ep_summary(e) for k, e in (out.get("episodes") or {}).items()})
    else:
        meta.update(n=out["n"], seed=out["seed"], ik_success_pct=out["ik_success_pct"],
                    controllers={k: {kk: c[kk] for kk in ("success_pct", "collision_pct", "timeout_pct")} for k, c in out["controllers"].items()})
    write_json(os.path.join(d, "meta.json"), meta)
    with open(os.path.join(d, "mission.json"), "w", encoding="utf-8") as f:
        json.dump(out, f)
    return out


def missions_list():
    os.makedirs(MISSIONS_DIR, exist_ok=True)
    out = [m for m in (read_json(os.path.join(MISSIONS_DIR, d, "meta.json")) for d in os.listdir(MISSIONS_DIR)) if m]
    out.sort(key=lambda m: m.get("created") or 0, reverse=True)
    return out


def mission_get(mid):
    d = os.path.join(MISSIONS_DIR, mid)
    if os.path.basename(mid) != mid or not os.path.isfile(os.path.join(d, "mission.json")):
        raise FileNotFoundError(f"no saved mission {mid}")
    with open(os.path.join(d, "mission.json"), encoding="utf-8") as f:
        return json.load(f)


def mission_batch(body):
    n = max(1, min(100, int(body.get("n", 20))))
    seed0 = int(body.get("seed", 1000))
    rows, t0 = [], time.time()
    for i in range(n):
        rows.append(mission(dict(body, seed=seed0 + i), record=False))
    per = [{"seed": r["seed"], "ik_success": r["ik"]["success"], "ik_pos_err_mm": r["ik"]["pos_err_mm"], "ik_ms": r["ik"]["ms"],
            "conflicts": (r.get("estimate") or {}).get("obstacle_conflicts"),
            "episodes": {k: _ep_summary(e) for k, e in (r.get("episodes") or {}).items()}} for r in rows]
    ok = [r for r in rows if r["ik"]["success"]]
    pct = lambda xs, f: 100.0 * sum(1 for x in xs if f(x)) / max(1, len(xs))
    ctrl = {}
    for c in ("policy", "baseline", "apf"):
        eps = [r["episodes"][c] for r in ok]
        succ = [e for e in eps if e["outcome"] == "success"]
        ctrl[c] = {"success_pct": pct(eps, lambda e: e["outcome"] == "success"),
                   "collision_pct": pct(eps, lambda e: e["outcome"] == "collision"),
                   "self_collision_pct": pct(eps, lambda e: e.get("collision") == "self"),
                   "timeout_pct": pct(eps, lambda e: e["outcome"] == "timeout"),
                   "final_pos_err_mm": float(np.median([e["final_pos_err_mm"] for e in succ])) if succ else None,
                   "final_orn_err_deg": float(np.median([e["final_orn_err_deg"] for e in succ])) if succ else None,
                   "steps": float(np.mean([e["steps"] for e in succ])) if succ else None}
    out = {"n": n, "seed": seed0, "robot": rows[0]["robot"] if rows else None, "scenario": body.get("scenario", "dynamic"), "secs": time.time() - t0,
            "l1_run": body.get("l1_run"), "l2_policy": body.get("l2_policy"), "rows": per,
            "ik_success_pct": pct(rows, lambda r: r["ik"]["success"]),
            "ik_pos_err_mm": float(np.median([r["ik"]["pos_err_mm"] for r in rows])),
            "ik_ms": float(np.mean([r["ik"]["ms"] for r in rows])),
            "predicted_conflict_pct": pct(ok, lambda r: r["estimate"]["obstacle_conflicts"] > 0),
            "controllers": ctrl}
    return _save_mission(out, body, kind="batch")


def _refine_settings(r):
    """Validated refine/lm_solver settings from a request (None / false = network only)."""
    if not r:
        return None
    r = dict(M.REFINE_DEFAULTS, **{k: v for k, v in r.items() if k in M.REFINE_DEFAULTS})
    out = {"iters": int(r["iters"]), "restarts": int(r["restarts"]), "use_seed": bool(r["use_seed"]),
           "tol_mm": float(r["tol_mm"]), "tol_deg": float(r["tol_deg"])}
    if not 1 <= out["iters"] <= 500 or not 0 <= out["restarts"] <= 256:
        raise ValueError("refinement: iterations must be 1-500 and random restarts 0-256")
    if out["tol_mm"] <= 0 or out["tol_deg"] <= 0:
        raise ValueError("refinement: tolerances must be positive")
    return out


# --------------------------------------------------------------------------
# System stats (1 Hz)
# --------------------------------------------------------------------------

GPU_FIELDS = ["name", "util", "mem_util", "mem_used", "mem_total", "temp", "power", "power_limit", "clock_sm"]


def gpu_stats():
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,"
             "temperature.gpu,power.draw,power.limit,clocks.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4, creationflags=NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        g = {}
        for k, v in zip(GPU_FIELDS, parts):
            if k == "name":
                g[k] = v
            else:
                try:
                    g[k] = float(v)
                except ValueError:
                    g[k] = None
        gpus.append(g)
    return gpus


def stats_loop():
    psutil.cpu_percent(percpu=True)
    while True:
        time.sleep(1.0)
        vm = psutil.virtual_memory()
        per = psutil.cpu_percent(percpu=True)
        s = {
            "type": "stats", "t": time.time(),
            "cpu": round(sum(per) / max(1, len(per)), 1), "per_cpu": per,
            "cpu_count": psutil.cpu_count(logical=True),
            "ram_used_gb": round((vm.total - vm.available) / 1e9, 2),
            "ram_total_gb": round(vm.total / 1e9, 2), "ram_pct": vm.percent,
            "gpus": gpu_stats(),
        }
        ps = _PROC["ps"]
        if ps is not None:
            # On Windows a venv python.exe is a launcher that runs the real
            # interpreter as a child, so sum over the whole process tree.
            try:
                tree = [ps] + ps.children(recursive=True)
            except psutil.Error:
                tree = []
            cpu = rss = threads = 0
            cache = _PROC.setdefault("cache", {})
            for proc in tree:
                try:
                    cached = cache.setdefault(proc.pid, proc)
                    cpu += cached.cpu_percent(None)
                    rss += cached.memory_info().rss
                    threads += cached.num_threads()
                except psutil.Error:
                    pass
            if tree:
                s["job_cpu"] = round(cpu, 1)
                s["job_rss_gb"] = round(rss / 1e9, 2)
                s["job_threads"] = threads
        else:
            _PROC["cache"] = {}
        with STATE_LOCK:
            STATE["stats"].append(s)
        HUB.publish(s)


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, cache=False):
        if not os.path.isfile(path):
            return self._send(404, {"error": "not found"})
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        if path.endswith(".js"):
            ctype = "text/javascript"
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        return json.loads(self.rfile.read(n).decode() or "{}")

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        qs = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
        path = url.path
        try:
            if path in ("/", "/index.html"):
                return self._file(os.path.join(STATIC_DIR, "index.html"))
            if path.startswith("/static/"):
                target = os.path.normpath(os.path.join(STATIC_DIR, path[len("/static/"):]))
                if not target.startswith(STATIC_DIR):
                    return self._send(403, {"error": "forbidden"})
                return self._file(target)
            if path.startswith("/mesh/"):
                key = path[len("/mesh/"):].split(".")[0]
                if key not in MESH_FILES:
                    return self._send(404, {"error": "unknown mesh"})
                return self._file(MESH_FILES[key], cache=True)
            if path.startswith("/files/"):
                relp = urllib.parse.unquote(path[len("/files/"):])
                target = os.path.normpath(os.path.join(WORKSPACE, relp))
                if not target.startswith(LIB + os.sep):
                    return self._send(403, {"error": "forbidden"})
                return self._file(target)
            if path == "/events":
                return self._events()
            if path == "/api/robots":
                return self._send(200, [{"name": k, "dof": v["dof"], "description": v["description"]} for k, v in ROBOTS.items()])
            if path == "/api/robot_model":
                return self._send(200, robot_model(qs["robot"]))
            if path == "/api/library":
                return self._send(200, library())
            if path == "/api/generators":
                return self._send(200, {"folder": MR._rel(GR.DATAGEN_DIR), "generators": GR.list_generators()})
            if path == "/api/models":
                robot = qs.get("robot", "xarm6")
                if robot not in ROBOTS:
                    raise ValueError(f"unknown robot {robot}")
                return self._send(200, {"folder": MR._rel(MR.MODELSET_DIR), "models": MR.list_models(ROBOTS[robot]["dof"], robot)})
            if path == "/api/run":
                return self._send(200, run_detail(qs["id"]))
            if path == "/api/dataset/sample":
                return self._send(200, dataset_sample(qs["ds"], qs.get("i", 0)))
            if path == "/api/dataset/cloud":
                return self._send(200, dataset_cloud(qs["ds"]))
            if path == "/api/test_sample":
                return self._send(200, self._test_sample(qs["run"], qs.get("k")))
            if path == "/api/missions":
                return self._send(200, missions_list())
            if path == "/api/missions/get":
                return self._send(200, mission_get(qs["id"]))
            if path == "/api/l2/policies":
                return self._send(200, l2_policies())
            if path == "/api/l2/policy":
                return self._send(200, l2_detail(qs["id"]))
            if path == "/api/state":
                return self._send(200, self._snapshot())
            return self._send(404, {"error": "not found"})
        except (FileNotFoundError, KeyError, ValueError, IndexError) as e:
            return self._send(400, {"error": str(e)})

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        try:
            body = self._body()
            if path == "/api/fk":
                return self._send(200, fk(body["robot"], body["q"]))
            if path == "/api/generate":
                return self._send(200, {"ok": True, "ds_id": job_generate(body)})
            if path == "/api/train":
                return self._send(200, {"ok": True, "run_id": job_train(body)})
            if path == "/api/evaluate":
                job_evaluate(body)
                return self._send(200, {"ok": True})
            if path == "/api/infer":
                return self._send(200, infer(body))
            if path == "/api/stop":
                return self._send(200, {"stopped": stop_job()})
            if path == "/api/l2/train":
                return self._send(200, {"ok": True, "id": job_l2_train(body)})
            if path == "/api/l2/evaluate":
                job_l2_eval(body)
                return self._send(200, {"ok": True})
            if path == "/api/mission":
                return self._send(200, mission(body, save=True))
            if path == "/api/mission/batch":
                return self._send(200, mission_batch(body))
            if path == "/api/l2/rollout":
                return self._send(200, l2_rollout(body))
            if path == "/api/delete":
                out = delete_item(body)
                HUB.publish({"type": "library_changed"})
                return self._send(200, out)
            if path == "/api/import_project":
                added = import_project()
                HUB.publish({"type": "library_changed"})
                return self._send(200, added)
            return self._send(404, {"error": "not found"})
        except RuntimeError as e:
            return self._send(409, {"error": str(e)})
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as e:
            return self._send(400, {"error": str(e)})

    def _test_sample(self, run_id, k=None):
        """A held-out test row of a run (for honest inference demos)."""
        m = run_meta(run_id)
        sp = os.path.join(run_dir(run_id), "split.npz")
        if not os.path.exists(sp):
            raise ValueError("this run has no recorded held-out test split")
        with np.load(sp) as z:
            test = z["test"]
        d = load_ds(m["dataset_id"])
        free = test[d["collision_labels"][test] == 0]
        i = int(free[int(k) % len(free)] if k is not None else np.random.default_rng().choice(free))
        return {"dataset_id": m["dataset_id"], "index": i}

    def _snapshot(self):
        with STATE_LOCK:
            return {"job": STATE["job"], "stats": list(STATE["stats"]), "log": list(STATE["log"])[-200:],
                    "epochs": list(STATE["epochs"])}

    def _events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q = HUB.subscribe()
        try:
            hello = {"type": "hello", **self._snapshot()}
            self.wfile.write(f"data: {json.dumps(hello)}\n\n".encode())
            self.wfile.flush()
            while True:
                try:
                    data = q.get(timeout=15)
                    self.wfile.write(f"data: {data}\n\n".encode())
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            HUB.unsubscribe(q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    # Windows SO_REUSEADDR would let a second server bind the same port and
    # silently split requests between two processes - fail loudly instead.
    ThreadingHTTPServer.allow_reuse_address = os.name != "nt"

    class Server(ThreadingHTTPServer):
        def handle_error(self, request, client_address):
            if isinstance(sys.exc_info()[1], (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)):
                return   # a browser tab closed / reloaded mid-request - not an error
            super().handle_error(request, client_address)

    try:
        srv = Server((args.host, args.port), Handler)
    except OSError as e:
        sys.exit(f"Port {args.port} is already in use (is Neural IK Studio already running?): {e}")
    migrate_legacy()
    os.makedirs(MR.MODELSET_DIR, exist_ok=True)
    os.makedirs(GR.DATAGEN_DIR, exist_ok=True)
    pb_init()
    threading.Thread(target=stats_loop, daemon=True).start()
    srv.daemon_threads = True
    url = f"http://{args.host}:{args.port}/"
    print(f"Neural IK Studio running at {url}   (Ctrl+C to quit)")
    print(f"library (all datasets + models): {LIB}")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_job()


if __name__ == "__main__":
    main()
