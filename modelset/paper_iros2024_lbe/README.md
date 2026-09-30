# paper_iros2024_lbe/ – Learning-by-Example (Demby's, Farag & DeSouza, IROS 2024)

*"Inverse Kinematics of Robotic Manipulators Using a New Learning-by-Example Method."*
The paper feeds the network an **example** joint-pose pair (Qe, De) next to the query pose D
and reports sub-millimetre accuracy for 7-DoF arms; without the example the same networks
stay at tens of millimetres. Its three networks are reproduced here, each in both input
schemes, with layer layouts taken from the authors' code
([github.com/jacketdembys/iksolver](https://github.com/jacketdembys/iksolver)) rather than read
off the figure.

| File | Network (authors' class) | Input | Params (xArm6) |
|---|---|---|---|
| `paper_lbe_mlp.py` | Plain MLP (`utils.MLP`): 12 hidden × 1024, ReLU | [De, Qe, D] | 11.6 M |
| `paper_lbe_rmlp.py` | RMLP (`ResMLPSum`): input FC + 5 residual blocks × 1024 | [De, Qe, D] | 10.5 M |
| `paper_lbe_dmlp.py` | DMLP (`DenseNet` = "DenseMLP3"): 2 dense blocks × 5 layers × 1024 | [De, Qe, D] | 37.8 M |
| `paper_noex_mlp.py` | Plain MLP | D only | 11.6 M |
| `paper_noex_rmlp.py` | RMLP | D only | 10.5 M |
| `paper_noex_dmlp.py` | DMLP | D only | 37.8 M |

## The three networks, exactly as in the authors' code
- **Plain MLP** – `input_fc` → ReLU, 11 × (`Linear 1024` → ReLU), `output_fc` → Q (12 hidden
  layers, as in the paper). The reference class also creates SELU / PReLU / LeakyReLU / Sigmoid /
  BatchNorm modules that its `forward()` never uses; they are left out.
- **RMLP** – `input` Linear with **no activation**, then 5 blocks of
  `Linear → ReLU → Linear`, `+ skip`, `ReLU`, then `out` → Q.
- **DMLP** – `input` Linear → ReLU; dense block 1 (5 layers, each fed the concatenation of the
  block input and all earlier layers, 1024 each); transition `Linear(6144 → 1024)` → ReLU; dense
  block 2; `classifier Linear(6144 → dof)` → Q. That is 12 hidden layers, the paper's "same
  number of layers". It is the `DenseMLP3` class selected in the authors' `train.yaml`
  (NUM_BLOCKS 2, NUM_HIDDEN_LAYERS 5, 1024 neurons). The repository's `forward()` calls
  `self.classifier` without the definition showing in its `__init__`; it is implemented as the
  output Linear from the final concatenated width.

## How the input maps onto this project
The column order is the paper's "seq" dataset: **[De (6), Qe (dof), D (6)]**, with each pose
given as `[x, y, z, roll, pitch, yaw]`.

| Paper | Here |
|---|---|
| Qe – example joints: the previous step of a trajectory, each joint ±v away (v = 1…20°) | the **seed** train_ik already provides: true joints + N(0, 0.3 rad) |
| De – the pose of the example joints | computed **inside the model** as De = FK(Qe), from the robot's URDF (matches PyBullet to 0.0002 mm) |
| D – query pose with RPY orientation | query position + quaternion converted to RPY, "XYZ" convention (the paper generator's `matrix_to_euler_angles(R, "XYZ")`) |
| sklearn `MinMaxScaler` to [0, 1], fitted on the training set | [0, 1] min-max with ranges fitted once from 20,000 random configurations (saved with the model) |
| output Q in radians, loss = MSE on Q | output Q in radians, returned as [sin Q, cos Q] – train_ik's loss is MSE on [sin, cos] |

The model needs to know its robot to compute De. Training, evaluation and inference in the UI
pass it automatically. A model built without that information assumes xArm6 for 6 joints and
KUKA iiwa for 7 (`DEFAULT_ROBOT_FOR_DOF` at the top of each file).

## What cannot match the paper (fixed by train_ik, not by the model file)
- **Example distance.** The paper's best numbers use examples 1° away; it shows RMLP / DMLP stay
  around 1–2 mm up to 20°, while the plain MLP degrades beyond about 12°. Here the seed noise
  is fixed by `train_ik.load_dataset` at σ = 0.3 rad (≈ 17°, and Gaussian, so some seeds are
  30–50° off). Expect results closer to the paper's large-v end than to its headline numbers.
- **Data and training.** The paper trains on 800 k samples with batch 128 for up to 1000 epochs,
  early stopping with patience 100. train_ik uses batch 256, and stopping is the UI's
  target-based automatic stop.
- **Loss.** MSE on [sin Q, cos Q] instead of on Q; for small errors the two behave alike.
- **Robots.** The paper uses a 7R Panda and a 7-DoF 2RP4R arm. Here: xArm6, KUKA iiwa and the
  Panda – for the Panda, generate the dataset with `datagen/clean_generator.py` (the base
  generator labels every Panda pose colliding).

## Suggested settings to reproduce the paper's comparison
1. **Data:** generate **1,000,000** samples (natural split) – the paper's size. Also generate a
   10,000-sample seed-1 set for an independent test.
2. **Train Model**, identical for all six:
   - held-out test 10 % (the paper's 80 / 10 / 10)
   - IK epochs (max) 300, collision-net epochs 15
   - position tolerance **1 mm** to measure the paper's metric (or 30 mm for this project's goal)
   - automatic stop on the validation split, 5 epochs in a row
   - render every 25
3. **Order:** `paper_lbe_rmlp` first (the paper's best), then `paper_lbe_dmlp`, `paper_lbe_mlp`,
   then the three `paper_noex_*` baselines to see the Learning-by-Example effect.
4. **Cost:** the DMLP has 37.8 M parameters – expect it to be several times slower per epoch
   than the others.
