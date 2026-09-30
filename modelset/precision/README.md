# precision/ – network + numerical refinement for < 1 mm / < 1°

## Why the earlier models could not reach 1 mm / 1°

1. **Regression networks plateau at a few millimetres.** The training loop (`train_ik.py`) minimises
   the error in joint angles, not in end-effector position. Even so, the best network here (M10,
   1 M samples, 512 epochs, 7 h) ends at a 3.8 mm / 1.1° median. Only 2 % of targets are within
   1 mm **and** 1°.
2. **The published results agree.** Networks without refinement report errors of a few
   millimetres or more:
   - CycleIK: 5.7 mm / 1.95°
   - IKFlow: about 10 mm

   The IROS 2024 Learning-by-Example paper reaches sub-millimetre error only when its example pose
   is about 1° from the answer; at 20° it is back to 1–2 mm. With an arbitrary starting pose,
   sub-millimetre accuracy is reported only with a numerical solver after the network. Lu, Zou &
   Jiang 2022 did this on the xArm6 (network + Newton–Raphson, 1e-6 m).
3. **Angles decoded outside the joint limits.** The network decodes angles with `atan2`
   (−180…180°), while xArm6 joint 3 spans −225…+11°. A solver started from such an angle gets
   stuck at the limit unless the angle is first shifted by 360°. That step alone took success from
   77 % to 99.8 %.

## The approach used here

| Part | File | Role |
|---|---|---|
| Data | `datagen/precision_generator.py` | Clean collision labels for every arm (including the Panda), Sobol coverage, near-limit samples, one quaternion sign |
| Network | `precision_ik_net.py` (this folder) | Gives a starting point close to a correct solution, for any seed |
| Solver | `refine/lm_solver.py` | Finishes the answer: LM from the network output and the seed, random restarts, collision-free retry |
| Check | `ui/runners/ik_metrics.py` | PyBullet re-checks every answer; **Success** = ≤ 1 mm and ≤ 1° and within limits and collision-free |

### `precision_ik_net.py`

- **Trunk:** M10's, the best-measured network here: 8 pre-norm SwiGLU residual blocks, width 384, 7.1 M parameters.
- **Inputs:** position, 6D rotation and sin/cos of the seed.
- **Two output heads:**
  - seed + correction, which is precise when the seed is close;
  - a direct angle prediction, which is usable when it is not.

  A learned per-joint gate blends the two.
- **Seed augmentation during training.** `train_ik` always feeds seed = truth + N(0, 0.3), so on its own a network learns to rely on a good seed. This model replaces the seed of each training row:
  - 60 % keep the given seed;
  - 25 % get N(0, 0.7) extra noise, about 45° per joint;
  - 15 % get a random pose within the limits.

  The network therefore also learns what to do without a good seed (cold start). Evaluation and inference use the seed as given.

## Results

*(filled in after the training run – see below)*

## How to reproduce in the UI

1. **Generate Dataset:**
   - Generator `datagen/precision_generator.py`, 500 000 samples, seed 0, Natural split.
   - Again with **20 000 samples, seed 1**, as an independent test set.
2. **Train Model:**
   - Model `precision/precision_ik_net.py`, held-out test 10 %, IK epochs 60, position tolerance 1 mm.
   - Automatic stop off. The network only has to land near the right solution, and the solver does the rest.
3. **Run details:**
   - Set Position tolerance **1** and Orientation **1**, and read the **Success** column.
   - The "↳ network + LM refinement" rows are the final system.
   - Evaluate on the seed-1 set with seed *warm*, *zero pose* and *random pose*, with LM refinement ticked.
4. **Run Inference:** Refinement *LM solver*, 30 iterations, 16 random restarts.
