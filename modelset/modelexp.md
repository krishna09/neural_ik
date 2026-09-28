# Model experiments – `modelset/`

Ten IK network architectures (`m01` … `m10`) built to push towards the goal below. They are
**untrained and untested for accuracy** – they only pass the UI's load check (build + one dummy
batch, for 6- and 7-joint arms). Your own files (`model_set.py`, `base_512.py`) are untouched.

## Goal and what is fixed

- **Goal:** ≥ 90 % of predictions land within **30 mm** of the target (end-effector position),
  measured on the held-out test split.
- **Fixed training loop:** every model is trained by the unchanged `train_ik.main()`:
  loss = MSE on `[sin q, cos q]` + 0.05 × collision penalty, Adam lr 1e-3, batch 256, and
  train_ik's own 85 / 15 train / validation split. Only the network class differs, so any gain
  has to come from the architecture, the data and the number of epochs.
- **Input** `x` = `[x, y, z, qx, qy, qz, qw, seed q1 … q_dof]`. In training and testing the seed
  is the true joint solution + N(0, 0.3 rad) noise (about 14° per joint).

## What limits accuracy today (measured on your earlier runs)

| Finding | Evidence |
|---|---|
| The base network mostly copies its seed | 12 % within 50 mm vs 4 % for "just output the seed"; wrist joints 4 and 6 no better than the seed |
| Data is the biggest lever | same model, 60 epochs: 15 k samples → 12 % within 50 mm, 80 k → 47 % |
| The inputs are discontinuous | 31 % of quaternions have w < 0 (q and −q are the same rotation); ±360° joints wrap around |
| The loss is in joint space | position error is dominated by joints 1–3 (65–83 mm each), but all joints weigh the same in the loss |

## Design choices shared by all ten models

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
7. **Self-contained files.** A run stores a copy of only its own model file.

## The ten models

| # | File | Architecture | Params (6-DOF) | Output |
|---|---|---|---|---|
| M01 | `m01_seed_residual_mlp.py` | Pre-norm ResMLP, 6 blocks × 384 | 3.56 M | seed + Δ |
| M02 | `m02_direct_wide_mlp.py` | Wide MLP 4 × 1024, LayerNorm + SiLU | 3.19 M | direct angles |
| M03 | `m03_fourier_features.py` | Random Fourier features of the pose + ResMLP 5 × 512 | 5.40 M | seed + Δ |
| M04 | `m04_film_pose_conditioned.py` | Pose encoder FiLM-modulates a seed trunk, 6 × 384 | 4.88 M | seed + Δ |
| M05 | `m05_iterative_refiner.py` | 4 unrolled, weight-shared refinement steps (ResMLP 3 × 512) | 3.40 M | seed + Σ Δₖ |
| M06 | `m06_kinematic_chain.py` | Trunk 4 × 512 + one head per joint, conditioned on earlier joints | 5.41 M | seed + Δ per joint |
| M07 | `m07_joint_transformer.py` | Transformer, 4 layers, d = 128, pose token + joint tokens | 0.81 M | seed + Δ per token |
| M08 | `m08_mixture_of_experts.py` | Shared encoder + softmax gate over 8 experts | 3.71 M | seed + gated Δ |
| M09 | `m09_dense_skip_mlp.py` | DenseNet-style MLP, 8 layers × 128 growth | 0.76 M | seed + Δ |
| M10 | `m10_swiglu_seed_gate.py` | SwiGLU ResNet 8 × 384, seed-trust gate between two heads | 7.12 M | blend of seed + Δ and direct |

For comparison: base `train_ik.IKNet` (768 wide) 0.90 M, your `model_set.py` 2.05 M.

### M01 – Seed-residual ResMLP (the reference)
Stem → 6 pre-norm residual blocks (LN → Linear 384→768 → SiLU → Linear 768→384, + skip) → LN → Δ.
**Why:** the simplest form of the seed-residual idea on a deep but stable network; the model every
other one should be compared against. **Watch:** if the seed is far from the answer (e.g. zero
pose at inference), a residual model has further to go than a direct one.

### M02 – Direct wide MLP (control experiment)
Same continuous inputs, but predicts the angles directly as unit-circle `[sin, cos]` pairs.
**Why:** isolates how much the new input encoding alone helps, and it does not rely on a good seed.
**Watch:** has to learn the seed → answer mapping from scratch, so expect it to need more data
and epochs than M01.

### M03 – Fourier features + ResMLP
The 9 pose numbers are also mapped to 128 random frequencies [sin(2πBv), cos(2πBv)] (σ = 2,
fixed buffer saved with the model) before the ResMLP. **Why:** MLPs are biased towards smooth
functions; Fourier features let them fit the fine detail millimetre accuracy needs (Tancik et al.
2020). **Watch:** too many high frequencies can overfit on small datasets – use the 200 k set.

### M04 – FiLM pose conditioning
A pose encoder produces a scale γ and shift β for every block of a trunk that processes the seed
(LN(h)·(1 + γ) + β). **Why:** turns "correct this seed" into "correct this seed *for this target*"
at every depth instead of only at the input. FiLM layers start as identity.

### M05 – Learned iterative solver
q₀ = seed; 4 times: q ← q + f(pose, sin/cos q, sin/cos seed, step), one shared network with a step
embedding, trained end to end. **Why:** mirrors how classical IK (damped least squares) converges –
each step sees where the previous one landed and can fix what is left. Strong candidate for the
30 mm goal. **Watch:** each training step runs the network 4× – about 4× slower per epoch than M01.

### M06 – Kinematic chain
A shared trunk makes a context vector; joint i is then predicted by its own head that also sees the
already-predicted joints 1 … i−1. **Why:** in a serial arm the wrist angles depend on what the
shoulder and elbow did. **Watch:** an error in an early joint propagates to later heads.

### M07 – Joint transformer
One token per joint (sin/cos seed + learned joint embedding) plus one pose token, 4 pre-norm
Transformer layers, per-joint Δ. **Why:** attention learns joint-to-joint couplings (e.g. elbow
compensating shoulder) explicitly; small (0.81 M) so it trains fast. **Watch:** transformers can
be sensitive to a fixed lr of 1e-3 – if its loss is noisy early on, give it more epochs.

### M08 – Mixture of experts
A softmax gate weighs the Δ of 8 small experts. **Why:** IK is piecewise – different workspace
regions and elbow-up / elbow-down branches need different local corrections, which experts can
specialise on. **Watch:** without a load-balancing loss (the training loop is fixed) a few experts
may end up doing most of the work; it still works, just with less specialisation.

### M09 – Dense skip MLP
Each of 8 layers sees the input plus all earlier layers' outputs. **Why:** the raw pose and seed
stay visible at every depth (like NeRF's skip connection) – good for precise regression, and small
(0.76 M). **Watch:** the concatenated width grows with depth, so deeper versions get slow.

### M10 – SwiGLU with a seed-trust gate
Deep SwiGLU residual blocks with two heads – seed + Δ and a direct prediction – blended per joint
by a learned gate (starts at ~5 % direct). **Why:** keeps the accuracy of the residual form when the
seed is good, and can lean on the direct head when it is not – the most robust option for real use
with poor seeds. **Watch:** the largest model (7.1 M); slowest per epoch.

## How to run the comparison

1. **Data:** your 200 k-sample xArm6 dataset (natural split). For an independent test, generate
   one more dataset with **10,000 samples, seed 1**.
2. **Train Model**, same settings for every model:
   - Held-out test split **10 %**, IK epochs (max) **200**, collision-net epochs **15**
   - Position tolerance **30 mm**
   - Automatic stop: **≥ 90 %** within tolerance, measured on the **validation split**,
     **3 epochs in a row**
   - Render every **20** epochs, live samples / s **2**
3. **Order:** M01 first (reference), then M05, M07 and M04; then the rest. Train the base model
   and `base_512.py` on the same dataset as baselines.
4. **Afterwards:** Run details → Evaluate on the seed-1 dataset. Compare in the Library
   (its Tolerance / Val acc / Test acc columns) and note test accuracy ≤ 30 mm, mean error,
   epochs used and time per epoch.

## Expectations

- The seed-residual models start where the base model is weakest (they begin at "output the
  seed" and learn only the correction), so they should beat the base model on the same data.
- **≥ 90 % within 30 mm is ambitious** with train_ik's joint-space loss and 0.3 rad seed noise;
  these models are designed towards it, not verified to reach it.
- **Accuracy depends on the seed.** Test seeds are truth + N(0, 0.3). With a poor seed (e.g.
  the zero pose in Run Inference), residual models lose more than M02 or M10.
- **If a model plateaus below the target**, the remaining levers sit outside the model file and
  would need a change to the training loop: an end-effector (forward-kinematics) loss, a
  learning-rate schedule, a larger batch. Without touching training, the inference panel's
  **DLS refinement (5–20 iterations)** closes the last tens of millimetres.
