# Model experiments – guide

How the models in `modelset/` relate to the accuracy goal, and how to compare them fairly. The
architectures themselves are explained in each folder's `README.md` (index in
[`README.md`](README.md)). None of the models in `mlp/`, `conditioned/`, `structured/` and
`paper_iros2024_lbe/` has been trained or tested for accuracy yet.

## Goal and what is fixed

- **Goal:** ≥ 90 % of predictions land within **30 mm** of the target (end-effector position),
  measured on the held-out test split.
- **Fixed training loop:** every model is trained by the unchanged `train_ik.main()`:
  loss = MSE on `[sin q, cos q]` + 0.05 × collision penalty, Adam lr 1e-3, batch 256, and
  train_ik's own 85 / 15 train / validation split. Only the network class differs, so any gain
  has to come from the architecture, the data and the number of epochs.
- **Input** `x` = `[x, y, z, qx, qy, qz, qw, seed q1 … q_dof]`. In training and testing the seed
  is the true joint solution + N(0, 0.3 rad) noise (about 14° per joint on average).

## What limits accuracy today (measured on your earlier runs)

| Finding | Evidence |
|---|---|
| The base network mostly copies its seed | 12 % within 50 mm vs 4 % for "just output the seed"; wrist joints 4 and 6 no better than the seed |
| Data is the biggest lever | same model, 60 epochs: 15 k samples → 12 % within 50 mm, 80 k → 47 % |
| The inputs are discontinuous | 31 % of quaternions have w < 0 (q and −q are the same rotation); ±360° joints wrap around |
| The loss is in joint space | position error is dominated by joints 1–3 (65–83 mm each), but all joints weigh the same in the loss |

## Design choices shared by mlp/, conditioned/ and structured/

1. **Continuous rotation input.** The quaternion is converted to the first two columns of the
   rotation matrix ("6D rotation", Zhou et al. 2019): continuous and identical for q and −q.
2. **Seed as sin / cos.** No jump at ±180° / ±360°.
3. **Position scaled ×1.25** so it spans roughly −1 … 1 for these arms.
4. **Seed-residual output** (all except M02): the network predicts a correction, q = seed + Δ,
   and the Δ head is **initialised to zero**. An untrained model outputs the seed exactly; training
   only has to learn how to remove the seed's error for the given pose.
5. **Exact unit-circle output.** The model outputs `[sin q, cos q]` of real angles, so
   train_ik's MSE becomes 2·(1 − cos(q̂ − q)) per joint – a clean angular error.
6. **Pre-norm residual blocks, LayerNorm, no dropout / batch-norm.** Stable at a fixed Adam
   lr 1e-3 without a schedule, and dropout only adds noise to a noise-free regression target.

The `paper_iros2024_lbe/` models follow the paper instead (RPY input, min-max scaling, direct
joint-angle output) – see their README.

## How to run the comparison

1. **Data:** a large dataset of the robot (200 k–1 M samples, natural split; for the Panda use
   `datagen/clean_generator.py`). For an independent test, generate one more dataset with
   **10,000 samples, seed 1**.
2. **Train Model**, same settings for every model:
   - Held-out test split **10 %**, IK epochs (max) **200**, collision-net epochs **15**
   - Position tolerance **30 mm**
   - Automatic stop: **≥ 90 %** within tolerance, measured on the **validation split**,
     **3 epochs in a row**
   - Render every **20** epochs, live samples / s **2**
3. **Order:** `mlp/` M01 first (the reference), then `structured/` M05 and M07, `conditioned/` M04;
   then the rest. Train the base model and `custom/` models on the same dataset as baselines.
4. **Afterwards:** Run details → Evaluate on the seed-1 dataset. Compare in the Library
   (Model / Tolerance / Val acc / Test acc columns) and note test accuracy ≤ 30 mm, mean error,
   epochs used and time per epoch.

## Expectations

- The seed-residual models start where the base model is weakest (they begin at "output the
  seed" and learn only the correction), so they should beat the base model on the same data.
- **≥ 90 % within 30 mm is ambitious** with train_ik's joint-space loss and 0.3 rad seed noise;
  these models are designed towards it, not verified to reach it.
- **Accuracy depends on the seed.** Test seeds are truth + N(0, 0.3). With a poor seed (e.g.
  the zero pose in Run Inference), residual models lose more than M02 or M10.
- **If a model plateaus below the target**, the remaining levers sit outside the model file –
  see the next section.

## Levers outside the model file (from the literature)

Published neural IK methods that reach millimetre or sub-millimetre accuracy share three ideas;
all three need something beyond a new model file:

1. **Train on the end-effector error through differentiable forward kinematics**, not on joint
   angles (CycleIK, Habekost et al., ICANN 2023 / IROS 2024). This needs a training loop other than
   `train_ik.py`'s joint-space MSE. Our own in-memory experiment confirmed a differentiable FK of the
   xArm6 matches PyBullet to 0.0002 mm.
2. **Network + numerical refinement – built: [`refine/`](../refine/README.md).** The network gives
   the starting point and Levenberg–Marquardt iterations finish the job (as in Lu, Zou & Jiang,
   *Sensors* 2022, on the xArm6, and Tenhumberg et al., Humanoids 2023). It is available in
   Evaluate, in Run Inference, and automatically on the held-out test of every training run. With
   your M10 run it reaches 100 % (warm seed) and 99.9 % (zero-pose seed) at < 1 mm / < 1°, against
   2 % for the network alone. For this goal use [`precision/`](precision/README.md) together with
   `datagen/precision_generator.py`.
3. **One strict success criterion – built:** the *Success* column in Run details. It counts
   position ≤ tolerance **and** orientation ≤ tolerance **and** within joint limits **and**
   collision-free, reported separately for the network alone and for network + refinement.
