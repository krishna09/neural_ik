"""Shared paths and small helpers for the presentation / video pipeline.

Everything under deliverables/ is generated from the project's own code and
recorded runs; nothing in the project itself (data/, models/, ui/workspace/)
is modified by these scripts.
"""
import json
import os
import sys
from pathlib import Path

DELIV = Path(__file__).resolve().parents[1]          # deliverables/
REPO = DELIV.parent                                    # project root
SCRIPTS = DELIV / "scripts"
RESULTS = DELIV / "results"
SIMS = DELIV / "simulations"
VIDEO = DELIV / "video"
PRES = DELIV / "presentation"
DOCS = DELIV / "documentation"

# Large intermediates (datasets, frames, audio) stay outside git.
WORK = Path(os.environ.get("NIK_WORK", "/tmp/nik_work"))

# Recorded runs that ship with the repository (ui/workspace/library is committed).
BEST_RUN = "20260928-174703_xarm6"                     # 2.05 M-parameter residual MLP, 200 k samples
BEST_RUN_DIR = REPO / "ui" / "workspace" / "library" / "runs" / BEST_RUN
DATASET_200K = REPO / "ui" / "workspace" / "library" / "datasets" / "ds_20260928-163059_xarm6_n200000_s0"
L2_META = REPO / "media" / "data" / "l2_20261013-185525_xarm6_meta_n200.json"

for p in (str(REPO), str(REPO / "ui" / "runners")):
    if p not in sys.path:
        sys.path.insert(0, p)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj, indent=1):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)
