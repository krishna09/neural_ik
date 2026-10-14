# Implementation audit — what the code actually does

**Project:** Neural Kinematics and Reinforcement Learning for Collision-Free Space Manipulator Path Planning (STP Team 06)
**Branch audited:** `main` at commit `4f9242d` ("after video and ppt generation")
**Robot shown in all deliverables:** UFactory xArm6 (6-DOF), PyBullet simulation only

This audit was written before building the presentation and video. Every claim on a slide or in the
narration maps back to a row in this file.

## 1. Corrections to the earlier (ChatGPT) notes

The notes pasted into the request were based on an older view of the repository. Three points were wrong:

| Earlier note | What the repository actually contains |
|---|---|
| "PPO is not implemented; present it only as a Level 2 proposal" | **PPO is implemented and trained.** `level2/ppo.py` (PPO in plain PyTorch), `level2/env.py` (obstacle and hardware-fault environment), `level2/evaluate.py`, `level2/train.py`. The team's best agent has a recorded 200-episode evaluation (`media/data/l2_20261013-185525_xarm6_meta_n200.json`). A 20 M-step reproduction was retrained for these deliverables. Level 2 is therefore presented as **implemented, in simulation**. |
| "Architecture: 256 → 256 → 128" is the model | That is the **baseline** network (`train_ik.IKNet`, 103,820 parameters). The best recorded run uses a **2.05 M-parameter residual MLP** (`modelset/custom/model_set.py`) trained on 200,000 samples. |
| Accuracy table 128 mm / 5 mm / 45 mm is the result | That is the **original plan baseline** (README). The project now finishes the network's answer with a **Levenberg–Marquardt solver** (`refine/lm_solver.py`): 100 % of 2,000 held-out targets land within 1 mm and 1°, with 0 % self-collisions (measured again here). |

## 2. Source map

| Functionality | Source | Notes |
|---|---|---|
| Robot registry, loading, joint lookup by name | `robot_config.py` | xArm6 URDF is a local copy (`urdf/xarm/`). Joint indices are resolved by name because the xArm6 URDF has a fixed `world_joint` at index 0. Self-collision needs `URDF_USE_SELF_COLLISION \| URDF_USE_SELF_COLLISION_EXCLUDE_PARENT`. |
| Dataset generation | `generate_dataset.py` | 70 % uniform joint samples + 30 % with every joint in the outer 15 % of its range. FK with `getLinkState`. Self-collision label from `getContactPoints(body, body)`. |
| Alternative generators | `datagen/*.py` | Clean collision matrix (needed for the Panda), Sobol sampling, clearance labels. Not used for the xArm6 results shown. |
| Preprocessing | `train_ik.load_dataset` | Only collision-free rows train the IK net. Seed = true joints + N(0, 0.3 rad). Input = [position 3, quaternion 4, seed 6] = 13. Target = [sin q 6, cos q 6] = 12. |
| Baseline IK network | `train_ik.IKNet` | 13 → 256 → 256 → 128 → 12, ReLU. Decoded with `atan2(sin, cos)`. |
| Best IK network | `modelset/custom/model_set.py` | 13 → 256 → 512 → 3 residual blocks of 512 (LayerNorm, SiLU / GELU / LeakyReLU, dropout 0.05) → 384 → 256 → 128 → 12. Each sin/cos pair is normalised before `atan2`. 2,050,316 parameters. |
| Collision classifier | `train_ik.CollisionNet` | 6 → 64 → 64 → 1, ReLU, BCE-with-logits loss. 4,673 parameters. |
| Collision-aware loss | `train_ik.train_ik_net` | `loss = MSE(sin/cos) + 0.05 × mean(sigmoid(CollisionNet(atan2-decoded angles)))`. The classifier is frozen and the penalty is differentiable. |
| Training loop | `train_ik.main` | Adam, lr 1e-3, batch 256 (validation batch 512). Random 85 / 15 train / validation split. 15 classifier epochs, 60 IK epochs by default. |
| Studio training (best run) | `ui/runners/train_runner.py` | Runs `train_ik.main()` unchanged with the model class swapped. Holds out a 15 % test split first (seed 2024). Logs per-epoch metrics. Refines the test split with LM at the end. |
| Accuracy metrics | `ui/runners/ik_metrics.py` | Predicted joints → PyBullet FK → position / orientation error. Threshold rates, self-collision, joint limits. |
| Classical baseline | `evaluate.py` | PyBullet `calculateInverseKinematics` (damped least squares, 100 iterations), given the same seed as the network. |
| Hybrid (original plan) | `evaluate.py` | Network output + 5 DLS iterations. |
| Hybrid (final) | `refine/lm_solver.py`, `refine/kinematics.py`, `ik_metrics.refine_ik` | Batched LM with Δq = (JᵀJ + λI)⁻¹Jᵀe, λ = ‖e‖² + 1e-6. Starts from the network output and the seed (wrapped into the joint limits). 30 iterations, 16 random restarts if outside 1 mm / 1°. Collision-free retry on another IK branch if PyBullet finds a self-collision. |
| Inference API | `ui/server.py: infer()` | Network → atan2 → optional LM / DLS refinement → PyBullet FK, limit and self-collision check. |
| Level 2 environment | `level2/env.py` | `ReachAvoidEnv`: start → goal in joint space. Sphere obstacles (static, sudden, drifting). Actuator gain / bias error and a sudden joint slip. Arm moved kinematically. |
| Level 2 agent | `level2/ppo.py` | Residual RL: command = potential field + 2 × policy correction. Actor 2 × 64 tanh. PPO with clip 0.2, GAE (γ 0.995, λ 0.95), horizon 128, 6 epochs, minibatch 2048, lr 3e-4 linear decay. `NumpyPolicy` deploys the actor without PyTorch. |
| Level 2 evaluation | `level2/evaluate.py` | Agent vs straight-line vs potential field on identical seeds. Six scenarios. |
| Level 1 + 2 mission | `ui/server.py: mission()` | Target pose → network + LM IK (seeded with the current joints) → straight-path check → three controllers on the same mission → final servo to the IK goal. |
| Studio (web UI) | `ui/server.py`, `ui/static/*` | Dataset browser, training monitor, inference, Level 2 and mission pages (Three.js). |

## 3. Answers to the audit questions (xArm6)

| # | Question | Answer (source) |
|---|---|---|
| 1 | Objective and scope | Level 1: neural IK that returns joint angles for a target pose without self-collision. Level 2: a light DRL agent that re-routes in real time around unexpected obstacles and hardware position errors (Project Brief PDF). Validated on standard arms because the space manipulator's URDF is not available (Level 1 plan PDF). |
| 2 | Simulator and robot | PyBullet 3.2.7, DIRECT (headless). xArm6, 6 revolute joints. Limits: J1 ±360°, J2 −118…120°, J3 −225…11°, J4 ±360°, J5 −97…180°, J6 ±360°. |
| 3 | Libraries | PyBullet, PyTorch, NumPy, Matplotlib (+ psutil and Pillow for the Studio, Three.js in the browser). |
| 4 | Dataset generation | See §2. Labels = PyBullet self-contact. |
| 5 | Dataset size | README recipe: 20,000 samples, 16.1 % colliding (reproduced here: 3,216). Best run: 200,000 samples (seed 0), 16.2 % colliding (32,392). |
| 6 | Preprocessing | Seed noise N(0, 0.3 rad); sin/cos target; quaternion xyzw input; colliding rows excluded from IK training. |
| 7 | Splits | `train_ik`: random 85 / 15 train / validation (both networks). Best run: 15 % test held out first → 25,090 collision-free test targets, 121,141 train and 21,377 validation IK rows. |
| 8 | Architecture | Baseline MLP 256-256-128 (103,820 params). Best: residual MLP (2,050,316 params). Classifier 64-64-1 (4,673 params). |
| 9 | Optimiser | Adam, learning rate 1e-3, no schedule. |
| 10 | Losses | Classifier: BCE-with-logits. IK: MSE on sin/cos + 0.05 × mean classifier probability. |
| 11 | Epochs / batch | Classifier 15 epochs. IK 60 (baseline) / 120 (best run). Batch 256. |
| 12 | Collision handling | (a) labels in the data; (b) IK net trained on collision-free rows only; (c) classifier penalty in the loss; (d) PyBullet re-check of every answer; (e) LM collision-free retry; (f) Level 2: classifier risk in the reward, contact ends the episode. |
| 13 | Model input / output | 13 numbers in, 12 out; `atan2` → 6 angles in radians. |
| 14 | Evaluation method | FK re-check of the predicted joints (pose actually reached). Self-collision, joint limits, wall-clock time. Same seed for every solver. |
| 15 | Inference method | `ui/server.py: infer()` (§2). |
| 16 | Comparison algorithms | PyBullet DLS; network + 5 DLS (plan hybrid); network + LM (final). Level 2: straight-line and potential-field controllers. |
| 17 | Results | `deliverables/results/` and §4. |
| 18 | Implemented vs proposed | §5. |

## 4. Results used in the deliverables, and where each comes from

**Measured here** (Python 3.13, PyTorch 2.14.1, PyBullet 3.2.7, 4-core CPU, no GPU):

- **Baseline recipe**, unchanged scripts (`results/baseline_repro/`): neural 123 mm / 33.0°, DLS 5.07 mm / 0.04°, hybrid 45.0 mm / 0.77°, on evaluate.py's 200 targets. README: 128 mm / 5 mm / 45 mm.
- **Best network**, whole held-out test split (25,090 targets): 23.06 mm mean / 5.76° — identical to the recorded run. Classifier test accuracy 98.37 %.
- **All methods on 2,000 identical held-out targets** (`results/evaluation_metrics.csv`):
  - network + LM: 100 % within 1 mm and 1° for warm, home-pose and random seeds. 0 % self-collision, 0 % limit violations.
  - LM without the network also reaches 100 %. From the home pose it needs random restarts for 57 % of targets instead of 19 %, and is slower (50.5 vs 37.8 ms per target; 1.18 vs 0.59 ms batched).
- **Level 2 reproduction**: see `results/level2_evaluation.csv` (filled in by `pipeline/evaluate_level2_repro.py`).

**Recorded by the team** (cited with source, not re-run):

- Best run training history and timing (RTX A6000, 718.8 s): `ui/workspace/library/runs/20260928-174703_xarm6/meta.json`.
- Level 2 best agent (20 M steps, 134 min, GPU): `media/data/l2_20261013-185525_xarm6_meta_n200.json`.

## 5. Implemented vs not implemented

| Implemented (simulation) | Not implemented / out of scope |
|---|---|
| Neural IK for xArm6 (and KUKA iiwa / Panda configs) | The actual space manipulator and satellite body (no URDF yet). Swapping it in = one `ROBOTS` entry. |
| Self-collision labels, classifier, collision-aware loss | Collision with a satellite body or external objects in Level 1 (self-collision only) |
| LM refinement to < 1 mm / < 1° | Continuous trajectory collision checking in Level 1 (configurations are checked one at a time) |
| PPO re-routing agent around sphere obstacles, joint slip, gain errors | Dynamics, torques, free-floating base, real hardware, non-spherical obstacles |
| Combined mission pipeline (Level 1 IK → Level 2 steering) | Flight-qualified or real-time-OS implementation |

## 6. Evaluation caveats carried into the slides

1. Warm seeds are perturbed true solutions (σ = 0.3 rad). They model "the arm is already close", not an arbitrary start. Home-pose and random seeds are therefore reported as well.
2. Self-collision is checked per configuration, not along a continuous path.
3. PyBullet DLS was run with fixed settings (100 iterations, no joint limits or collision information), as in `evaluate.py`. Other settings would change its numbers.
4. Timings are hardware-dependent. Here they are single-target, one CPU thread. The team's runs used GPUs.
5. The orientation error floor (~0.01°) comes from the quaternion-angle check itself (arccos near 1), not from the solver.
