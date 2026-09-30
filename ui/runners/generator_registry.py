"""
Which dataset generator a Generate job uses.

  "base"                -> generate_dataset.generate (the project's generator, unchanged)
  "datagen/<file>.py"   -> a generator in <project>/datagen/

A datagen file must provide a function
    generate(robot_name, n_samples, ...)
that writes an .npz with at least the keys of generate_dataset.py (joints,
positions, orientations, collision_labels, joint_limits, dof). It is used as it
is - gen_runner.py adapts to whatever it supports:
  * seed=...          passed when present
  * out_path=...      passed when present; otherwise the file is expected at
                      data/<robot>_ik_dataset.npz (relative to the working
                      directory, which is the new library entry) or at the
                      path generate() returns
  * progress=...      per-sample callback progress(i, total, q, pos, orn,
                      collided, phase); without it the live view follows the
                      file's own set_joint_config / get_ee_pose / collision
                      check functions
  * any other keyword parameter with a bool / int / float / str default is
    shown in the UI as a generator option
Optional module attributes: DESCRIPTION (else the docstring's first paragraph),
ID_TAG (short tag added to dataset ids), LAST_REPORT (shown in the UI),
OPTION_HELP ({parameter: help text}).
"""

import glob
import inspect
import os

import model_registry as MR

DATAGEN_DIR = os.path.join(MR.PROJECT_DIR, "datagen")
BASE_ID = "base"
BASE_DESCRIPTION = ("The project's generate_dataset.generate(): uniform + near-limit sampling; a pose is "
                    "colliding if any two links that are not parent / child touch.")
STANDARD_PARAMS = ("seed", "out_path", "progress")
OPTION_TYPES = {bool: "bool", int: "int", float: "float", str: "str"}


def source_path(gen_id):
    if gen_id == BASE_ID:
        return os.path.join(MR.PROJECT_DIR, "generate_dataset.py")
    if not gen_id.startswith("datagen/"):
        raise ValueError(f"unknown generator {gen_id!r}")
    path = os.path.normpath(os.path.join(DATAGEN_DIR, gen_id.split("/", 1)[1]))
    if os.path.dirname(path) != os.path.normpath(DATAGEN_DIR) or not path.endswith(".py"):
        raise ValueError(f"bad generator id {gen_id!r}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"generator file not found: {MR._rel(path)}")
    return path


def load(gen_id):
    """The generator module (base: the imported generate_dataset module)."""
    if gen_id == BASE_ID:
        import generate_dataset
        return generate_dataset
    return MR.load_module(source_path(gen_id), "gen")


def describe(mod):
    """What generate() accepts: {"accepts": {seed, out_path, progress}, "options": [...]}."""
    gen = getattr(mod, "generate", None)
    if not callable(gen):
        raise ValueError("no generate(robot_name, n_samples, ...) function")
    params = inspect.signature(gen).parameters
    positional = [p for p in params.values()
                  if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    if len(positional) < 2:
        raise ValueError("generate() must take the robot name and the number of samples as its first two parameters")
    helps = getattr(mod, "OPTION_HELP", None) or {}
    options = []
    for prm in list(params.values())[2:]:
        if prm.name in STANDARD_PARAMS or prm.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if prm.default is inspect.Parameter.empty:
            raise ValueError(f"generate() parameter {prm.name!r} has no default value, so the UI cannot fill it")
        kind = OPTION_TYPES.get(type(prm.default))
        if kind is None:
            continue                      # not settable from the UI - the generator's default is used
        options.append({"name": prm.name, "type": kind, "default": prm.default, "help": str(helps.get(prm.name, ""))})
    return {"accepts": {k: k in params for k in STANDARD_PARAMS}, "options": options}


def _description(mod):
    text = getattr(mod, "DESCRIPTION", None)
    if not text:
        text = (mod.__doc__ or "").strip().split("\n\n")[0]
    return " ".join(str(text).split())


def list_generators():
    out = [{"id": BASE_ID, "file": "generate_dataset.py", "base": True, "ok": True,
            "description": BASE_DESCRIPTION, "tag": "", "options": [], "adapted": False,
            "supports_ignore_pairs": False}]
    for path in sorted(glob.glob(os.path.join(DATAGEN_DIR, "*.py"))):
        name = os.path.basename(path)
        if name.startswith("_"):
            continue
        e = {"id": f"datagen/{name}", "file": MR._rel(path), "base": False}
        try:
            mod = MR.load_module(path, "genlist")
            info = describe(mod)
            with open(path, encoding="utf-8", errors="replace") as f:
                src = f.read()
            e.update(ok=True, description=_description(mod),
                     tag=getattr(mod, "ID_TAG", os.path.splitext(name)[0]),
                     options=info["options"],
                     # no progress callback -> gen_runner follows the file's own FK / collision calls
                     adapted=not info["accepts"]["progress"],
                     # the file reads cfg["self_collision_ignore_pairs"] -> the UI can hand it the
                     # link pairs that touch by construction (clean collision matrix)
                     supports_ignore_pairs="self_collision_ignore_pairs" in src and hasattr(mod, "load_robot"))
        except Exception as ex:  # a broken generator file must not break the UI
            e.update(ok=False, error=f"{type(ex).__name__}: {ex}")
        out.append(e)
    return out
