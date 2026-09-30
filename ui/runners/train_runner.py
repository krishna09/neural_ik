"""
Runs the project's train_ik.main() UNCHANGED inside a library run folder,
with a held-out test split, and streams what it is doing to the UI server.

Data flow for one run (cwd = ui/workspace/library/runs/<run_id>/):

  library dataset (.npz, N rows)
     |-- test split (test_frac, fixed split_seed) -> held out, train_ik never sees it
     '-- the rest -> data/<robot>_ik_dataset.npz in the run folder
                     -> train_ik.main() reads it (relative path, as always) and
                        makes its own random 85/15 train/validation split
                     -> writes models/ into the run folder

How it observes training without editing train_ik.py:
  * train_ik.IKNet / train_ik.CollisionNet are replaced by subclasses whose
    forward() calls the original forward() and then reports the batch. They
    add no parameters, so the saved state_dicts load straight into the
    original classes (evaluate.py keeps working on them).
  * train_ik.nn is replaced by a copy of torch.nn whose MSELoss /
    BCEWithLogitsLoss subclasses report each loss value before returning it.
  * torch.utils.data.random_split is wrapped to record which rows train_ik
    put in its train and validation subsets (it returns Subset objects that
    carry .indices), so accuracy can be reported per split afterwards.
  * Epoch boundaries are detected from model.train() / model.eval() switches.

Accuracy (see ik_metrics.py) = predicted joints -> PyBullet FK -> distance of
the reached end-effector pose from the target pose.
"""

import json
import math
import os
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pybullet as p  # noqa: E402
import torch  # noqa: E402
import torch.nn as tnn  # noqa: E402
import torch.utils.data as tud  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import ik_metrics as M  # noqa: E402
import collision_filter  # noqa: E402
import model_registry as MR  # noqa: E402
import train_ik as ti  # noqa: E402
from libio import read_json, rel, write_json  # noqa: E402
from robot_config import connect, load_robot  # noqa: E402


def emit(ev):
    sys.stdout.write("@@EV " + json.dumps(ev) + "\n")
    sys.stdout.flush()


def fl(a):
    return [round(float(v), 6) for v in a]


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------

P = json.loads(sys.argv[1])
ROBOT = P["robot"]
RUN_ID = P["run_id"]
DATASET_ID = P["dataset_id"]
DATASET_FILE = P["dataset_file"]
WORKSPACE = P["workspace"]
EPOCHS = int(P.get("epochs", 60))
CEPOCHS = int(P.get("collision_epochs", 15))
N_CASES = max(1, min(64, int(P.get("cases", 8))))
RENDER_EVERY = max(1, int(P.get("render_every", 5)))
SAMPLE_DT = 1.0 / max(0.5, float(P.get("sample_hz", 6)))
TEST_FRAC = min(0.5, max(0.0, float(P.get("test_frac", 0.15))))
SPLIT_SEED = int(P.get("split_seed", 2024))
ACC_SUBSET = int(P.get("acc_subset", 2000))
BATCH = 256  # train_ik's DataLoader batch size
# User-set accuracy tolerance: an end-effector counts as accurate when it lands
# within POS_TOL mm of the target. Reported everywhere next to the presets.
REFINE_TEST = bool(P.get("refine_test", True))      # also report network + numerical refinement on the test split
REFINE_TEST_MAX = int(P.get("refine_test_max", 20000))
REFINE_EPOCH = bool(P.get("refine_epoch", True))    # per-epoch accuracy of network + LM refinement too
REFINE_EPOCH_N = int(P.get("refine_epoch_n", 1000))  # targets per split and epoch
POS_TOL = float(P.get("pos_tol_mm", 50))
TOL_KEY = M.tkey(POS_TOL)
# Early stopping: {"metric": "pct_within" | "mean_error", "target_pct": 90,
#                  "split": "train" | "val", "patience": 2}  or None
ES = P.get("early_stop") or None

RUN_DIR = os.getcwd()
os.environ["IK_ROBOT"] = ROBOT      # robot-aware models (FK inside the network) read this

# Which network to train: train_ik.IKNet ("base") or a modelset/*.py model.
# The source is copied into the run folder first and the class is taken from
# that copy (base: train_ik's own class, identical to the copy), so the run
# can always be reloaded even if the model file is edited later.
MODEL_ID = P.get("model_id") or MR.BASE_ID
MODEL_INFO = MR.snapshot(MODEL_ID, RUN_DIR)
MODEL_CLS = ti.IKNet if MODEL_ID == MR.BASE_ID else MR.model_class_from_file(os.path.join(RUN_DIR, MR.SNAPSHOT), "run")
RENDER_DIR = "renders"
for d in (RENDER_DIR, "models", "data"):
    os.makedirs(d, exist_ok=True)

# --------------------------------------------------------------------------
# 1. Hold out the test split, write the rest where train_ik expects it
# --------------------------------------------------------------------------

with np.load(DATASET_FILE) as z:
    FULL = {k: z[k] for k in z.files}
N_ALL = len(FULL["joints"])
_perm = np.random.default_rng(SPLIT_SEED).permutation(N_ALL)
_n_test = int(round(TEST_FRAC * N_ALL))
TEST_IDX = np.sort(_perm[:_n_test])
SUB_IDX = np.sort(_perm[_n_test:])            # original index of each row train_ik sees
np.savez(
    f"data/{ROBOT}_ik_dataset.npz",
    joints=FULL["joints"][SUB_IDX], positions=FULL["positions"][SUB_IDX],
    orientations=FULL["orientations"][SUB_IDX], collision_labels=FULL["collision_labels"][SUB_IDX],
    joint_limits=FULL["joint_limits"], dof=FULL["dof"],
)

JOINTS = FULL["joints"][SUB_IDX]
LABELS = FULL["collision_labels"][SUB_IDX]
LIMITS = FULL["joint_limits"]
FREE_SUB = np.nonzero(LABELS == 0)[0]

# ti.load_dataset is deterministic (rng seed 0), so this reproduces the exact
# X that main() will build from the same file.
_X, _y, DOF, _ = ti.load_dataset(ROBOT)
X_ORIG = SUB_IDX[FREE_SUB]                    # original dataset index of each X row
Q_TRUE_X = JOINTS[FREE_SUB]                   # ground-truth joints of each X row
X_TO_INDEX = {_X[i].tobytes(): int(X_ORIG[i]) for i in range(len(_X))}
Q_TO_LABEL = {JOINTS[i].tobytes(): int(LABELS[i]) for i in range(len(JOINTS))}


def n_batches(n):
    n_val = max(1, int(0.15 * n))
    return math.ceil((n - n_val) / BATCH)


NB = {"collision": n_batches(len(JOINTS)), "ik": n_batches(len(_X))}
TOTAL_EPOCHS = {"collision": CEPOCHS, "ik": EPOCHS, "evaluating": 1}

META = read_json("meta.json", {}) or {}
META.update({
    "id": RUN_ID, "kind": "ui-train", "robot": ROBOT, "dataset_id": DATASET_ID,
    "status": "running", "created": META.get("created", time.time()),
    "model": {**MODEL_INFO, "class": MODEL_CLS.__name__, "params": MR.count_params(MODEL_CLS(DOF))},
    "params": {"epochs": EPOCHS, "collision_epochs": CEPOCHS, "cases": N_CASES,
               "render_every": RENDER_EVERY, "test_frac": TEST_FRAC, "split_seed": SPLIT_SEED,
               "pos_tol_mm": POS_TOL, "early_stop": ES, "acc_subset": ACC_SUBSET},
    "split": {"n_dataset": int(N_ALL), "n_test": int(len(TEST_IDX)),
              "n_test_free": int((FULL["collision_labels"][TEST_IDX] == 0).sum()),
              "n_given_to_train_ik": int(len(SUB_IDX)), "n_ik_rows": int(len(_X))},
    "epochs": [], "metrics": {}, "collision_metrics": {},
    "evaluations": META.get("evaluations", []),
    "models": {"ik": f"models/{ROBOT}_ik_net.pt", "collision": f"models/{ROBOT}_collision_net.pt",
               "history": f"models/{ROBOT}_history.json"},
    "manifest": f"{RENDER_DIR}/manifest.json",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
})


def save_meta():
    write_json("meta.json", META)


save_meta()

# --------------------------------------------------------------------------
# Offscreen renderer (PyBullet TinyRenderer) - also provides the FK body used
# for accuracy measurements
# --------------------------------------------------------------------------

W, H, HEAD = 420, 380, 64
TRUTH_RGBA = (0.10, 0.62, 0.44, 1)   # aqua  - dataset ground truth
PRED_RGBA = (0.85, 0.35, 0.15, 1)    # orange - network prediction
HIDE = [0, 0, -50]
ORIGIN = [0, 0, 0]
IDENT = [0, 0, 0, 1]


def _font(size):
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


class Renderer:
    def __init__(self):
        connect(gui=False)
        p.loadURDF("plane.urdf")
        self.truth, self.cfg, limits = load_robot(ROBOT)
        self.pred, _, _ = load_robot(ROBOT)
        for body in (self.truth, self.pred):
            collision_filter.apply(body, self.cfg, limits)   # ignore link pairs that touch by construction
        for body, rgba in ((self.truth, TRUTH_RGBA), (self.pred, PRED_RGBA)):
            for link in range(-1, p.getNumJoints(body)):
                p.changeVisualShape(body, link, rgbaColor=rgba)
        self.marker = self._make_marker()
        pos = FULL["positions"]
        r = float(max(np.linalg.norm(pos[:, :2], axis=1).max(), pos[:, 2].max(), 0.4))
        self.view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0.36 * r], distance=1.55 * r,
            yaw=50, pitch=-22, roll=0, upAxisIndex=2)
        self.proj = p.computeProjectionMatrixFOV(fov=45, aspect=W / H, nearVal=0.05, farVal=30)
        self.f_head = _font(15)
        self.f_small = _font(13)
        self.park()

    def _make_marker(self):
        """Target end-effector frame: red sphere + RGB axes (x, y, z)."""
        sph = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[0.9, 0.2, 0.2, 1])
        L, t = 0.09, 0.006
        boxes = [
            p.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=c)
            for he, c in (((L / 2, t, t), (1, 0.15, 0.15, 1)),
                          ((t, L / 2, t), (0.15, 0.85, 0.15, 1)),
                          ((t, t, L / 2), (0.2, 0.4, 1, 1)))
        ]
        return p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=sph, basePosition=HIDE,
            linkMasses=[0] * 3, linkCollisionShapeIndices=[-1] * 3, linkVisualShapeIndices=boxes,
            linkPositions=[(L / 2, 0, 0), (0, L / 2, 0), (0, 0, L / 2)],
            linkOrientations=[IDENT] * 3,
            linkInertialFramePositions=[(0, 0, 0)] * 3,
            linkInertialFrameOrientations=[IDENT] * 3,
            linkParentIndices=[0, 0, 0], linkJointTypes=[p.JOINT_FIXED] * 3,
            linkJointAxis=[(0, 0, 1)] * 3)

    def park(self):
        """Only the `truth` body at the origin: it doubles as the FK body for
        accuracy measurements (self-collision checks only look at itself)."""
        p.resetBasePositionAndOrientation(self.truth, ORIGIN, IDENT)
        p.resetBasePositionAndOrientation(self.pred, HIDE, IDENT)
        p.resetBasePositionAndOrientation(self.marker, HIDE, IDENT)

    def _set(self, body, q):
        for idx, a in zip(self.cfg["joint_indices"], q):
            p.resetJointState(body, idx, float(a))

    def _snap(self, truth, pred):
        p.resetBasePositionAndOrientation(self.truth, ORIGIN if truth else HIDE, IDENT)
        p.resetBasePositionAndOrientation(self.pred, ORIGIN if pred else HIDE, IDENT)
        img = p.getCameraImage(W, H, self.view, self.proj, renderer=p.ER_TINY_RENDERER,
                               lightDirection=[1, 1, 2], shadow=1)
        rgb = np.reshape(np.array(img[2], dtype=np.uint8), (H, W, 4))[:, :, :3]
        return Image.fromarray(rgb)

    def render_case(self, case, epoch, q_true, q_pred, target_pos, target_orn, out_path):
        self._set(self.truth, q_true)
        self._set(self.pred, q_pred)
        p.resetBasePositionAndOrientation(self.pred, ORIGIN, IDENT)   # FK in world frame needs it at the origin
        pred_pos, pred_orn, _ = M.fk_batch(self.pred, self.cfg["joint_indices"], self.cfg["ee_link_index"], [q_pred])
        m = {
            "pos_err_mm": float(np.linalg.norm(pred_pos[0] - np.array(target_pos)) * 1000),
            "orn_err_deg": float(M.quat_angle_deg(pred_orn[0], target_orn)[0]),
            "joint_err_deg": float(np.degrees(np.abs(M.wrap(np.array(q_pred) - np.array(q_true)))).mean()),
        }
        p.resetBasePositionAndOrientation(self.marker, target_pos, target_orn)
        panels = [("Input: target pose (dataset ground truth)", self._snap(True, False)),
                  ("Network prediction", self._snap(False, True)),
                  ("Overlay: aqua = truth, orange = prediction", self._snap(True, True))]
        self.park()

        canvas = Image.new("RGB", (W * 3, H + HEAD), (26, 26, 25))
        d = ImageDraw.Draw(canvas)
        d.text((12, 8), f"{ROBOT}  |  case {case['case']}  (dataset sample #{case['dataset_index']}, training split)  |  "
                        f"epoch {epoch}/{EPOCHS}", fill=(255, 255, 255), font=self.f_head)
        d.text((12, 32), f"end-effector position error {m['pos_err_mm']:.1f} mm   |   orientation error "
                         f"{m['orn_err_deg']:.1f} deg   |   mean joint error {m['joint_err_deg']:.1f} deg   "
                         f"|   red sphere + axes = target frame", fill=(195, 194, 183), font=self.f_small)
        for k, (title, im) in enumerate(panels):
            canvas.paste(im, (k * W, HEAD))
            d.rectangle([k * W + 6, HEAD + 6, k * W + 12 + d.textlength(title, font=self.f_small), HEAD + 26],
                        fill=(26, 26, 25))
            d.text((k * W + 9, HEAD + 8), title, fill=(255, 255, 255), font=self.f_small)
        canvas.save(out_path)
        return m


def renderer():
    if S.renderer is None:
        S.renderer = Renderer()
    return S.renderer


# --------------------------------------------------------------------------
# Instrumentation state
# --------------------------------------------------------------------------

class S:
    phase = None
    epoch = 0
    batch = 0
    prev_training = None
    open = False
    t0 = 0.0
    tr, val, pen = [], [], []
    last_emit = 0.0
    last_x = None
    ik_model = None
    col_model = None
    splits = []         # [collision (train, val) indices into subset rows, IK (train, val) indices into X]
    acc_idx = None      # fixed per-epoch accuracy subsets {"train": idx into X, "val": idx into X}
    cases = None
    cases_x = None
    manifest = None
    renderer = None
    streak = 0          # consecutive epochs meeting the early-stop target
    stop = False        # set when the target held for `patience` epochs


class EarlyStop(Exception):
    """Raised from the IK net's forward() to leave train_ik.main() early."""


def check_early_stop(acc):
    """Evaluate the user's rule on this epoch's accuracy; returns the event info."""
    s = acc[ES["split"]]
    if ES["metric"] == "mean_error":
        value, met = s["pos_mean_mm"], s["pos_mean_mm"] <= POS_TOL
    else:
        value = s["acc_pos"][TOL_KEY]
        met = value >= float(ES["target_pct"])
    S.streak = S.streak + 1 if met else 0
    if S.streak >= int(ES["patience"]) and S.epoch < EPOCHS:
        S.stop = True
    return {"met": bool(met), "value": float(value), "streak": S.streak,
            "patience": int(ES["patience"]), "stop": S.stop}


def begin_epoch():
    S.epoch += 1
    S.batch = 0
    S.tr, S.val, S.pen = [], [], []
    S.t0 = time.time()
    S.open = True


def epoch_accuracy():
    """Per-epoch IK accuracy on fixed subsets of train_ik's own train and
    validation rows (FK only - no collision check - to keep epochs fast)."""
    if S.acc_idx is None:
        ik_tr, ik_va = S.splits[1]
        rng = np.random.default_rng(SPLIT_SEED + 7)
        S.acc_idx = {
            "train": np.sort(rng.choice(ik_tr, min(ACC_SUBSET, len(ik_tr)), replace=False)),
            "val": np.sort(rng.choice(ik_va, min(ACC_SUBSET, len(ik_va)), replace=False)),
        }
    r = renderer()
    out, out_ref = {}, {}
    for name, idx in S.acc_idx.items():
        out[name], per = M.evaluate_ik(S.ik_model, r.truth, r.cfg, LIMITS, _X[idx], Q_TRUE_X[idx],
                                       collision=False, extra_pos_mm=(POS_TOL,))
        if REFINE_EPOCH:
            # the full system: network + LM refinement (refine/lm_solver.py), checked with PyBullet
            k = min(REFINE_EPOCH_N, len(idx))
            out_ref[name], _ = M.refine_ik(ROBOT, r.truth, r.cfg, LIMITS, _X[idx[:k]], per["q_pred"][:k],
                                           extra_pos_mm=(POS_TOL,), q_true=Q_TRUE_X[idx[:k]])
    return out, out_ref


def end_epoch():
    if not S.open:
        return
    S.open = False
    ev = {"type": "epoch_end", "robot": ROBOT, "run_id": RUN_ID, "phase": S.phase, "epoch": S.epoch,
          "epochs": TOTAL_EPOCHS[S.phase], "secs": round(time.time() - S.t0, 3),
          "train_loss": float(np.mean(S.tr)) if S.tr else None}
    if S.phase == "collision":
        tr, va = S.splits[0]
        ev["train_acc"] = M.collision_metrics(S.col_model, JOINTS[tr], LABELS[tr])["accuracy"]
        ev["val_acc"] = M.collision_metrics(S.col_model, JOINTS[va], LABELS[va])["accuracy"]
    else:
        ev["val_mse"] = float(np.mean(S.val)) if S.val else None
        ev["collision_penalty"] = float(np.mean(S.pen)) if S.pen else None
        ev["acc"], acc_ref = epoch_accuracy()
        if acc_ref:
            ev["acc_refined"] = acc_ref
        ev["pos_tol_mm"] = POS_TOL
        if ES:
            ev["early_stop"] = check_early_stop(ev["acc"])
    if torch.cuda.is_available():
        ev["gpu_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    emit(ev)
    META["epochs"].append({k: v for k, v in ev.items() if k not in ("type", "robot", "run_id")})
    if S.stop:
        META["early_stop_result"] = {"stopped_at_epoch": S.epoch, "max_epochs": EPOCHS, "rule": ES,
                                     "pos_tol_mm": POS_TOL, "value": ev["early_stop"]["value"]}
        emit({"type": "early_stop", "robot": ROBOT, "run_id": RUN_ID, **META["early_stop_result"]})
    save_meta()
    if S.phase == "ik" and S.cases is not None and (
            S.epoch == 1 or S.epoch % RENDER_EVERY == 0 or S.epoch == EPOCHS or S.stop):
        render_cases(S.epoch)


def set_phase(phase):
    end_epoch()
    S.phase = phase
    S.epoch = 0
    S.prev_training = None
    emit({"type": "phase", "robot": ROBOT, "run_id": RUN_ID, "phase": phase,
          "epochs": TOTAL_EPOCHS[phase], "nb": NB.get(phase, 1), "epoch": 0, "batch": 0})


def track_mode(model):
    if model.training and S.prev_training is not True:
        end_epoch()
        begin_epoch()
    if model.training:
        S.batch += 1
    S.prev_training = model.training


def throttle():
    now = time.time()
    if now - S.last_emit >= SAMPLE_DT:
        S.last_emit = now
        return True
    return False


# --------------------------------------------------------------------------
# Instrumented classes / functions (behaviour-identical to the originals)
# --------------------------------------------------------------------------

_OrigIK, _OrigCol = MODEL_CLS, ti.CollisionNet     # the selected IK network is swapped into train_ik
_orig_random_split = tud.random_split


def random_split_observed(dataset, lengths, *args, **kwargs):
    parts = _orig_random_split(dataset, lengths, *args, **kwargs)
    S.splits.append([np.asarray(part.indices, dtype=int) for part in parts])
    return parts


class IKNetObserved(_OrigIK):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        S.ik_model = self
        set_phase("ik")

    def forward(self, x):
        out = super().forward(x)
        track_mode(self)          # may close the previous epoch and trip the early-stop rule
        if S.stop:
            # first batch of the next epoch: nothing has been updated yet, so the
            # weights are exactly those at the end of the epoch that met the rule
            raise EarlyStop()
        S.last_x = x
        return out

    def raw_forward(self, x):
        """The model's own forward pass, without the training observer."""
        return _OrigIK.forward(self, x)


class CollisionNetObserved(_OrigCol):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        S.col_model = self
        set_phase("collision")

    def forward(self, q):
        out = super().forward(q)
        if S.phase == "collision":
            track_mode(self)
            if self.training and throttle():
                q0 = q[0].detach().cpu().numpy()
                emit({"type": "train_sample", "robot": ROBOT, "run_id": RUN_ID, "phase": "collision",
                      "epoch": S.epoch, "epochs": CEPOCHS, "batch": S.batch, "nb": NB["collision"],
                      "q_true": fl(q0), "label": Q_TO_LABEL.get(q0.tobytes()),
                      "prob": float(torch.sigmoid(out[0]).item())})
        elif S.phase == "ik" and S.ik_model is not None and S.ik_model.training:
            S.pen.append(float(torch.sigmoid(out.detach()).mean().item()))
        return out


class MSEObserved(tnn.MSELoss):
    def forward(self, inp, target):
        loss = super().forward(inp, target)
        if S.phase == "ik" and S.ik_model is not None:
            if S.ik_model.training:
                S.tr.append(float(loss.item()))
                if S.cases is None:
                    capture_cases(S.last_x, target)
                if throttle():
                    x0 = S.last_x[0].detach().cpu().numpy()
                    q_pred = M.decode(inp[0].detach().cpu().numpy(), DOF)
                    q_true = M.decode(target[0].detach().cpu().numpy(), DOF)
                    emit({"type": "train_sample", "robot": ROBOT, "run_id": RUN_ID, "phase": "ik",
                          "epoch": S.epoch, "epochs": EPOCHS, "batch": S.batch, "nb": NB["ik"],
                          "dataset_index": X_TO_INDEX.get(x0.tobytes()),
                          "target_pos": fl(x0[:3]), "target_orn": fl(x0[3:7]), "seed": fl(x0[7:]),
                          "q_true": fl(q_true), "q_pred": fl(q_pred),
                          "sample_mse": float(((inp[0] - target[0]) ** 2).mean().item()),
                          "batch_loss": float(loss.item())})
            else:
                S.val.append(float(loss.item()))
        return loss


class BCEObserved(tnn.BCEWithLogitsLoss):
    def forward(self, inp, target):
        loss = super().forward(inp, target)
        if S.phase == "collision":
            S.tr.append(float(loss.item()))
        return loss


# --------------------------------------------------------------------------
# Tracked training cases + rendering
# --------------------------------------------------------------------------

def capture_cases(xb, yb):
    """The first K rows of the very first training batch -> guaranteed to be
    in the training split (not validation / test)."""
    k = min(N_CASES, xb.shape[0])
    xs = xb[:k].detach().clone()
    ys = yb[:k].detach().cpu().numpy()
    xn = xs.cpu().numpy()
    S.cases_x = xs
    S.cases = []
    for c in range(k):
        S.cases.append({
            "case": c + 1, "dataset_index": X_TO_INDEX.get(xn[c].tobytes()),
            "target_pos": fl(xn[c][:3]), "target_orn": fl(xn[c][3:7]),
            "seed": fl(xn[c][7:]), "q_true": fl(M.decode(ys[c], DOF)), "images": [],
        })
    S.manifest = {"run_id": RUN_ID, "robot": ROBOT, "epochs": EPOCHS, "collision_epochs": CEPOCHS,
                  "render_every": RENDER_EVERY, "created": time.time(), "cases": S.cases}
    emit({"type": "cases", "robot": ROBOT, "run_id": RUN_ID,
          "cases": [{k2: v for k2, v in c.items() if k2 != "images"} for c in S.cases]})


def render_cases(epoch):
    r = renderer()
    q_preds = M.predict(S.ik_model, S.cases_x.cpu().numpy(), DOF)
    t0 = time.time()
    for c, q_pred in zip(S.cases, q_preds):
        fname = f"case{c['case']:02d}_epoch{epoch:03d}.png"
        path = os.path.join(RUN_DIR, RENDER_DIR, fname)
        m = r.render_case(c, epoch, c["q_true"], q_pred, c["target_pos"], c["target_orn"], path)
        rec = {"epoch": epoch, "file": rel(path, WORKSPACE), "q_pred": fl(q_pred), **m}
        c["images"].append(rec)
        emit({"type": "render", "robot": ROBOT, "run_id": RUN_ID, "case": c["case"],
              "dataset_index": c["dataset_index"], **rec})
    write_json(os.path.join(RENDER_DIR, "manifest.json"), S.manifest)
    emit({"type": "render_round", "robot": ROBOT, "run_id": RUN_ID, "epoch": epoch,
          "secs": round(time.time() - t0, 2), "n": len(S.cases)})


# --------------------------------------------------------------------------
# Final evaluation on train / validation / held-out test
# --------------------------------------------------------------------------

def inference_speed():
    m = S.ik_model
    fwd = m.raw_forward
    x1 = torch.as_tensor(_X[:1], device=next(m.parameters()).device)
    sync = torch.cuda.synchronize if x1.is_cuda else (lambda: None)
    was_training = m.training
    m.eval()
    try:
        with torch.no_grad():
            for _ in range(20):
                fwd(x1)
            sync()
            t0 = time.perf_counter()
            for _ in range(300):
                fwd(x1)
            sync()
            single = (time.perf_counter() - t0) / 300
            xb = torch.as_tensor(_X[:4096], device=x1.device)
            sync()
            t0 = time.perf_counter()
            fwd(xb)
            sync()
            batch = (time.perf_counter() - t0) / len(xb)
    finally:
        m.train(was_training)
    return {"device": str(x1.device), "single_ms": single * 1e3, "batch_us_per_sample": batch * 1e6}


def final_evaluation():
    set_phase("evaluating")
    r = renderer()
    body, cfg = r.truth, r.cfg
    col_tr, col_va = S.splits[0]
    ik_tr, ik_va = S.splits[1]
    t0 = time.time()

    tol = (POS_TOL,)
    metrics = {}
    for name, rows in (("train", ik_tr), ("val", ik_va)):
        metrics[name], per_s = M.evaluate_ik(S.ik_model, body, cfg, LIMITS, _X[rows], Q_TRUE_X[rows], extra_pos_mm=tol)
        if REFINE_TEST:
            # training accuracy of the full system (network + LM) on up to REFINE_TEST_MAX rows
            sub = np.sort(np.random.default_rng(SPLIT_SEED + 5).choice(len(rows), min(REFINE_TEST_MAX, len(rows)),
                                                                        replace=False))
            metrics[name + "_refined"], _ = M.refine_ik(ROBOT, body, cfg, LIMITS, _X[rows][sub], per_s["q_pred"][sub],
                                                        extra_pos_mm=tol, q_true=Q_TRUE_X[rows][sub])
    labels_all = FULL["collision_labels"]
    col = {
        "train": M.collision_metrics(S.col_model, JOINTS[col_tr], LABELS[col_tr]),
        "val": M.collision_metrics(S.col_model, JOINTS[col_va], LABELS[col_va]),
    }
    if len(TEST_IDX):
        test_free = TEST_IDX[labels_all[TEST_IDX] == 0]
        seeds = M.noisy_seeds(FULL["joints"][test_free], SPLIT_SEED + 1)
        Xt = M.make_inputs(FULL["positions"][test_free], FULL["orientations"][test_free], seeds)
        metrics["test"], per = M.evaluate_ik(S.ik_model, body, cfg, LIMITS, Xt, FULL["joints"][test_free], extra_pos_mm=tol)
        col["test"] = M.collision_metrics(S.col_model, FULL["joints"][TEST_IDX], labels_all[TEST_IDX])
        extra = {}
        if REFINE_TEST:
            # network + numerical refinement (refine/lm_solver.py) on (up to REFINE_TEST_MAX) test rows:
            # warm start (the seeds above) and cold start (zero-pose seed), each checked with PyBullet
            sub = np.sort(np.random.default_rng(SPLIT_SEED + 3).choice(
                len(test_free), min(REFINE_TEST_MAX, len(test_free)), replace=False))
            q_sub = FULL["joints"][test_free][sub]
            print(f"[refine] network + LM refinement on {len(sub)} held-out test targets (warm and cold start)",
                  flush=True)
            metrics["test_refined"], pr = M.refine_ik(ROBOT, body, cfg, LIMITS, Xt[sub], per["q_pred"][sub],
                                                      extra_pos_mm=tol, q_true=q_sub)
            Xc = M.make_inputs(FULL["positions"][test_free][sub], FULL["orientations"][test_free][sub],
                               np.zeros_like(q_sub, dtype=np.float32))
            metrics["test_cold"], pc = M.evaluate_ik(S.ik_model, body, cfg, LIMITS, Xc, q_sub, extra_pos_mm=tol)
            metrics["test_cold_refined"], prc = M.refine_ik(ROBOT, body, cfg, LIMITS, Xc, pc["q_pred"],
                                                            extra_pos_mm=tol, q_true=q_sub)
            extra = {"refined_subset": sub, **{k: v for k, v in pr.items()},
                     **{k + "_cold": v for k, v in prc.items()}, "q_pred_cold": pc["q_pred"]}
        np.savez("test_results.npz", index=test_free, seed=seeds, **{k: v for k, v in per.items()}, **extra)

    np.savez("split.npz", test=TEST_IDX, ik_train=X_ORIG[ik_tr], ik_val=X_ORIG[ik_va],
             col_train=SUB_IDX[col_tr], col_val=SUB_IDX[col_va])
    META["split"].update({"n_ik_train": int(len(ik_tr)), "n_ik_val": int(len(ik_va)),
                          "n_col_train": int(len(col_tr)), "n_col_val": int(len(col_va))})
    META["metrics"] = metrics
    META["collision_metrics"] = col
    META["inference_speed"] = inference_speed()
    META["eval_secs"] = round(time.time() - t0, 2)
    emit({"type": "final_metrics", "robot": ROBOT, "run_id": RUN_ID, "metrics": metrics,
          "collision_metrics": col, "inference_speed": META["inference_speed"]})


# --------------------------------------------------------------------------

def main():
    nn_proxy = types.ModuleType("nn_observed")
    nn_proxy.__dict__.update(tnn.__dict__)
    nn_proxy.MSELoss = MSEObserved
    nn_proxy.BCEWithLogitsLoss = BCEObserved
    ti.nn = nn_proxy
    ti.IKNet = IKNetObserved
    ti.CollisionNet = CollisionNetObserved
    tud.random_split = random_split_observed   # train_ik calls torch.utils.data.random_split

    emit({"type": "train_start", "robot": ROBOT, "run_id": RUN_ID, "dataset_id": DATASET_ID,
          "epochs": EPOCHS, "collision_epochs": CEPOCHS, "n_samples": int(N_ALL),
          "n_test": int(len(TEST_IDX)), "n_given": int(len(SUB_IDX)), "n_free": int(len(_X)),
          "pos_tol_mm": POS_TOL, "early_stop": ES, "acc_subset": ACC_SUBSET, "model": META["model"],
          "device": META["device"], "gpu": META["gpu"]})
    t0 = time.time()
    try:
        ti.main(ROBOT, EPOCHS, CEPOCHS)   # the project's own training entry point
        end_epoch()                        # close (and render) the final epoch
    except EarlyStop:
        # The epoch opened by the interrupted batch never ran - discard it.
        S.open = False
        S.epoch -= 1
        # Save exactly what train_ik.main() saves after training (same files / format).
        torch.save(S.ik_model.state_dict(), f"models/{ROBOT}_ik_net.pt")
        torch.save(S.col_model.state_dict(), f"models/{ROBOT}_collision_net.pt")
        with open(f"models/{ROBOT}_history.json", "w") as f:
            json.dump({"val_mse": [e["val_mse"] for e in META["epochs"] if e.get("phase") == "ik"]}, f)
        print(f"early stop after epoch {S.epoch}/{EPOCHS}: target held for {ES['patience']} epochs - "
              f"saved models/{ROBOT}_ik_net.pt and models/{ROBOT}_collision_net.pt", flush=True)
    final_evaluation()
    META["status"] = "done"
    META["train_secs"] = round(time.time() - t0, 1)
    META["finished"] = time.time()
    save_meta()
    emit({"type": "train_done", "robot": ROBOT, "run_id": RUN_ID, "secs": META["train_secs"]})


if __name__ == "__main__":
    main()
