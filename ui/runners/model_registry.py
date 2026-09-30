"""
Which IK network a training run uses.

  "base"                 -> train_ik.IKNet  (the project's own model, trained as-is)
  "modelset/<file>.py"   -> a model file in <project>/modelset/

Every model is trained by the unchanged train_ik.main() loop (same loss,
optimiser, batch size, splits); only the network class is swapped in. So a
model file must follow the same contract as train_ik.IKNet:

  class IKNet(nn.Module):                  # any nn.Module name works; "IKNet" is preferred
      def __init__(self, dof): ...         # dof = number of arm joints
      def forward(self, x): ...            # x (B, 7+dof) = [xyz, quaternion xyzw, seed joints]
                                           # -> (B, 2*dof) = [sin q1..q_dof, cos q1..q_dof]
      def decode_angles(self, raw_out): .. # -> (B, dof) joint angles in radians

Each run keeps a copy of the model source it was trained with
(<run>/model_source.py) and later rebuilds the network from that copy, so
editing or deleting a model file never breaks evaluation / inference of
older runs.
"""

import hashlib
import importlib.util
import inspect
import os
import shutil
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(HERE))
MODELSET_DIR = os.path.join(PROJECT_DIR, "modelset")
BASE_ID = "base"
BASE_FILE = os.path.join(PROJECT_DIR, "train_ik.py")
SNAPSHOT = "model_source.py"


def _rel(path):
    return os.path.relpath(path, PROJECT_DIR).replace("\\", "/")


def load_module(path, tag="m"):
    """Import a .py file under a unique name (re-imported when the file changes)."""
    path = os.path.abspath(path)
    key = hashlib.sha1(f"{path}|{os.path.getmtime(path)}".encode()).hexdigest()[:10]
    name = f"_ikmodel_{tag}_{key}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    folder = os.path.dirname(path)
    sys.path.insert(0, folder)            # lets a model file import helpers next to it
    no_cache = sys.dont_write_bytecode
    sys.dont_write_bytecode = True        # keep modelset/ and run folders free of __pycache__
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = no_cache
        sys.path.remove(folder)
    sys.modules[name] = mod
    return mod


def find_class(mod):
    cls = getattr(mod, "IKNet", None)
    if isinstance(cls, type) and issubclass(cls, nn.Module):
        return cls
    cands = [c for c in vars(mod).values()
             if isinstance(c, type) and issubclass(c, nn.Module)
             and c.__module__ == mod.__name__ and hasattr(c, "decode_angles")]
    if not cands:
        raise ValueError("no model class found - define class IKNet(nn.Module) with forward() and decode_angles()")
    return cands[0]


def source_path(model_id):
    if model_id == BASE_ID:
        return BASE_FILE
    if not model_id.startswith("modelset/"):
        raise ValueError(f"unknown model id {model_id!r}")
    root = os.path.normpath(MODELSET_DIR)
    path = os.path.normpath(os.path.join(root, model_id.split("/", 1)[1]))
    if os.path.commonpath([root, path]) != root or not path.endswith(".py"):
        raise ValueError(f"bad model id {model_id!r}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"model file not found: {_rel(path)}")
    return path


def model_files():
    """Every model file in modelset/ and its sub-folders (one folder per model
    family). Files and folders starting with '_' or '.' are skipped."""
    out = []
    for root, dirs, files in os.walk(MODELSET_DIR):
        dirs[:] = sorted(d for d in dirs if not d.startswith(("_", ".")))
        out += [os.path.join(root, f) for f in sorted(files) if f.endswith(".py") and not f.startswith("_")]
    return out


def model_class_from_file(path, tag="m"):
    return find_class(load_module(path, tag))


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def check(cls, dof):
    """Instantiate and run a dummy batch; returns the parameter count or raises."""
    m = cls(dof)
    m.eval()
    with torch.no_grad():
        out = m(torch.zeros(4, 7 + dof))
        if tuple(out.shape) != (4, 2 * dof):
            raise ValueError(f"forward() returned shape {tuple(out.shape)}, expected (batch, {2 * dof}) = [sin, cos] per joint")
        if not hasattr(m, "decode_angles"):
            raise ValueError("model has no decode_angles(raw_out) method")
        ang = m.decode_angles(out)
        if tuple(ang.shape) != (4, dof):
            raise ValueError(f"decode_angles() returned shape {tuple(ang.shape)}, expected (batch, {dof})")
    return count_params(m)


def _describe(cls, mod):
    text = getattr(mod, "DESCRIPTION", None) or inspect.getdoc(cls) or ""
    return " ".join(line.strip() for line in text.strip().splitlines()[:3] if line.strip())[:160]


class robot_env:
    """Tell robot-aware models (e.g. ones that compute forward kinematics
    internally) which robot they are built for: sets IK_ROBOT while inside."""

    def __init__(self, robot):
        self.robot = robot

    def __enter__(self):
        self.prev = os.environ.get("IK_ROBOT")
        if self.robot:
            os.environ["IK_ROBOT"] = self.robot
        return self

    def __exit__(self, *exc):
        if self.prev is None:
            os.environ.pop("IK_ROBOT", None)
        else:
            os.environ["IK_ROBOT"] = self.prev


def list_models(dof, robot=None):
    """Base model + every modelset/*.py (files starting with '_' are skipped)."""
    with robot_env(robot):
        return _list_models(dof)


def _list_models(dof):
    entries = [(BASE_ID, BASE_FILE)]
    for p in model_files():
        entries.append(("modelset/" + os.path.relpath(p, MODELSET_DIR).replace("\\", "/"), p))
    out = []
    for mid, path in entries:
        sub = os.path.relpath(os.path.dirname(path), MODELSET_DIR).replace("\\", "/")
        e = {"id": mid, "file": _rel(path), "base": mid == BASE_ID,
             "group": "" if mid == BASE_ID or sub == "." else sub}
        try:
            mod = load_module(path, "list")
            cls = find_class(mod)
            e.update(ok=True, cls=cls.__name__, params=check(cls, dof), description=_describe(cls, mod))
        except Exception as ex:  # a broken model file must not break the UI
            e.update(ok=False, error=f"{type(ex).__name__}: {ex}")
        out.append(e)
    return out


def snapshot(model_id, run_dir):
    """Copy the model source into the run folder; returns the run's model record."""
    src = source_path(model_id)
    dst = os.path.join(run_dir, SNAPSHOT)
    shutil.copy2(src, dst)
    with open(dst, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return {"id": model_id, "file": _rel(src), "source": SNAPSHOT, "sha256": sha}


def run_model_class(run_dir, meta):
    """The IK class a run was trained with (from its snapshot; older runs: train_ik.IKNet)."""
    info = (meta or {}).get("model") or {}
    snap = os.path.join(run_dir, info.get("source") or SNAPSHOT)
    if info and os.path.isfile(snap):
        return model_class_from_file(snap, "run")
    import train_ik
    return train_ik.IKNet


def build_for_state_dict(cls, dof, state_dict):
    """cls(dof) loaded with state_dict. If a width-configurable model was saved
    with another width (e.g. train_ik.py's `hidden` was edited after training),
    rebuild it with the width stored in the checkpoint."""
    m = cls(dof)
    try:
        m.load_state_dict(state_dict)
        return m
    except RuntimeError:
        w = state_dict.get("net.0.weight")
        if w is None or "hidden" not in inspect.signature(cls.__init__).parameters:
            raise
        m = cls(dof, hidden=int(w.shape[0]))
        m.load_state_dict(state_dict)
        return m


def load_run_models(run_dir, meta, dof, device="cpu"):
    """(ik_net, collision_net or None) of a saved run, in eval mode."""
    import train_ik
    ik_sd = torch.load(os.path.join(run_dir, meta["models"]["ik"]), map_location=device)
    with robot_env(meta.get("robot")):
        ik = build_for_state_dict(run_model_class(run_dir, meta), dof, ik_sd).to(device).eval()
    col = None
    col_path = os.path.join(run_dir, meta["models"].get("collision") or "")
    if os.path.isfile(col_path):
        col_sd = torch.load(col_path, map_location=device)
        col = build_for_state_dict(train_ik.CollisionNet, dof, col_sd).to(device).eval()
    return ik, col
