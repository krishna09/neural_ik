# Speaker notes — Neural Kinematics for Intelligent Space Manipulators (STP Team 06)

Generated from the notes inside `Neural_IK_STP_Team06_Level1.pptx` (scripts/deck/build_deck.js). About 18–20 minutes in total; timings per slide are suggestions.

## Slide 1 · Neural Kinematics for Intelligent Space Manipulators

Opening (about 45 s).
Good morning. We are STP Team 06. Our project is Neural Kinematics and Reinforcement Learning for Collision-Free Space Manipulator Path Planning.
In this talk we show what we actually built and measured: a complete Level 1 pipeline that turns a target pose into collision-free joint angles, and a Level 2 reinforcement-learning agent that re-routes the arm around unexpected obstacles. Everything runs in PyBullet simulation on a UFactory xArm6, a six-joint arm standing in for the space manipulator.
All numbers in this deck come from result files in deliverables/results (re-measured for this presentation) or from the team's recorded runs, which are cited where they appear.

## Slide 2 · Reach any pose: precisely, legally, safely and fast

Problem statement (about 1 min).
Inverse kinematics asks: given where the gripper must be, and how it must be oriented, which six joint angles get it there?
For a space manipulator the answer must also be legal (inside every joint limit) and safe (the arm must never hit itself), and it has to come fast enough for real-time use.
The project brief splits the work into two levels. Level 1: a kinematics-aware AI model that computes joint angles and prevents self-collision. Level 2: a lightweight deep-reinforcement-learning agent that re-routes the arm in real time when obstacles or hardware position errors appear.
Both levels are implemented in simulation. The mentors approved using a standard arm, because the real manipulator's model is not available yet.
Source: Project Brief - Team 6.pdf; Level 1 Implementation Plan PDF.

## Slide 3 · Forward is easy; inverse is the challenge

Fundamentals (about 1 min).
Forward kinematics, left: chain the six joint rotations and you get exactly one pose. PyBullet computes this with getLinkState; our renderer uses the same chain and matches PyBullet to 0.005 mm.
Inverse kinematics goes the other way. There is no general closed-form answer, and for the target in the clip our solver found 6 different joint configurations that all reach it exactly; four are shown.
That is why classical IK solvers iterate from a starting guess, and why our network also takes a seed: the arm's current joints decide which solution we want.
Clip: simulations/inverse_kinematics.mp4 (Simulation 3). Code: robot_config.py, refine/lm_solver.py (random_candidates).

## Slide 4 · From plan to a working, reproducible pipeline

Development approach (about 1 min).
We followed the Level 1 implementation plan step by step, and each step is one script in the repository.
1 robot setup with joints resolved by name; 2 dataset generation; 3 the collision classifier; 4 the IK network with the collision-aware loss; 5 the benchmark against PyBullet's classical solver; 6 Levenberg-Marquardt refinement, added after the benchmark showed that a network alone stays tens of millimetres off.
Beyond the plan: a deeper residual network, a web Studio that runs every step, and Level 2.
Reproducibility: the README recipe was re-run unchanged for this presentation; classical DLS reproduced exactly (5.07 mm, 9 % self-collision) and the neural baseline within noise (123 mm vs 128 mm in the README).

## Slide 5 · Technology and simulation environment

Technology (about 45 s).
PyBullet is the simulator: forward kinematics, Jacobians and the contact checks that label self-collision. It runs headless in DIRECT mode.
PyTorch trains both networks and also runs our batched Levenberg-Marquardt solver. NumPy and Matplotlib handle data and figures; the web Studio uses Three.js to show the robot from PyBullet's link frames.
The right column says where each result was computed. The team's best Level 1 network trained in 719 s on an NVIDIA RTX A6000; the Level 2 agent trained for 20 million steps in 134 minutes with GPU updates. For this presentation, everything was re-run on a 4-core CPU with no GPU: Python 3.13, PyTorch 2.14.1, PyBullet 3.2.7.

## Slide 6 · Building the xArm6 digital robot

The robot (about 1 min).
The xArm6 comes from its URDF: meshes, six revolute joints and their limits, shown in the table.
Two details mattered. First, the URDF starts with a fixed world_joint at index 0, so assuming joints 0 to 5 silently shifts every index by one. robot_config.py resolves joints by name instead, which is also what lets a new robot drop in as one registry entry.
Second, PyBullet skips self-collision checks unless the URDF is loaded with URDF_USE_SELF_COLLISION; EXCLUDE_PARENT stops neighbouring links from always counting as touching.
The pose we control is the frame of link 6: a position and a quaternion.
Clip: simulations/robot_intro.mp4 (Simulation 1).

## Slide 7 · 200,000 labelled samples, generated in simulation

Dataset (about 1 min).
No external dataset is needed: generate_dataset.py samples random joint angles, 70 % uniformly inside the limits and 30 % close to the limits where self-collisions concentrate.
For each sample PyBullet computes the end-effector pose and checks every link pair for contact; the result is one row: six angles, a position, a quaternion and a collision label.
The main dataset has 200,000 rows, of which 32,392 (16.2 %) are self-colliding; the clip ends with 7,000 of their end-effector positions: the red cluster is near the base, where the wrist can hit the body.
Re-checking the first 120 rows in PyBullet gives the same labels (100 % agreement).
The IK network is trained only on collision-free rows; the colliding rows train the classifier.

## Slide 8 · What the network sees and what it predicts

Representation (about 1 min).
Input: 13 numbers. The target position (3), its orientation as a quaternion (4), and a seed of 6 joint angles. In training, the seed is the true answer plus Gaussian noise of 0.3 radians, standing in for the arm's current joints.
Output: 12 numbers, the sine and cosine of each joint angle. On a circle, plus and minus 180 degrees are neighbours, so the training targets have no artificial jump; atan2 turns each pair back into an angle.
Because atan2 returns -180 to +180 degrees while joint 3 runs from -225 to +11, each angle is then shifted by 360 degrees into the joint range when possible: same pose, legal value.
Code: train_ik.load_dataset, IKNet.decode_angles, refine/lm_solver.wrap_to_limits.

## Slide 9 · Neural IK architecture

Architecture (about 1 min).
The clip shows the baseline network from the plan, with its real activations for one held-out target: 13 inputs, three hidden layers of 256, 256 and 128 ReLU units, and 12 outputs.
The best network keeps exactly the same inputs, outputs and training loop, but has a deeper residual body: 2,050,316 parameters instead of 103,820.
On the target in the clip the baseline lands 92 mm away; the best network 16.7 mm. Over the whole held-out test split the best network averages 23.06 mm, re-measured here and identical to the recorded run.
Code: train_ik.IKNet (baseline), modelset/custom/model_set.py (best). Clip: simulations/neural_architecture.mp4 (Simulation 5).

## Slide 10 · Collision-aware learning

Collision awareness (about 1 min).
A small classifier, 6 inputs, two hidden layers of 64, one output, learns to predict self-collision from the joint angles alone. On 30,000 held-out poses it is 98.4 % accurate, with 95.9 % precision and 94.1 % recall (re-measured here).
In the clip the arm leans forward until its wrist touches the base: PyBullet reports contact at 80 % of the motion, exactly where the classifier's probability crosses 0.5.
During IK training the classifier is frozen and its probability for the predicted angles is added to the loss with weight 0.05. Because it is differentiable, it pushes the network away from risky poses during training, not only afterwards.
It is a soft penalty, not a guarantee: every final answer is still checked in PyBullet. Code: train_ik.train_ik_net.

## Slide 11 · Training and validation

Training (about 1 min).
Both networks are trained by the same train_ik loop: Adam, learning rate 0.001, batch 256, a random 85/15 train/validation split, 15 classifier epochs then the IK network.
The chart shows the mean validation position error after every epoch, checked with forward kinematics. The baseline (20,000 samples, 60 epochs, our CPU reproduction in 26 seconds) levels off near 115 mm. The residual network (200,000 samples, 120 epochs, the team's run on an RTX A6000 in 719 s) reaches 23.4 mm.
Training and validation errors fall together for most of training. Over the last epochs the residual network's validation error stops improving (best 21.5 mm at epoch 112, 23.4 mm at the end) while its training loss keeps falling: more epochs would not help, more data or regularisation might. Both histories are recorded, not reconstructed.
Sources: results/training_history.csv; ui/workspace/library/runs/20260928-174703_xarm6/meta.json.

## Slide 12 · Inference: guess, polish, check

Inference workflow (about 1 min).
One target is solved in five steps. The network makes a guess in about a millisecond on one CPU thread (0.93 ms measured), typically about two centimetres off.
The guess is shifted into the joint limits, then refined with Levenberg-Marquardt: the step solves (JᵀJ + λI) Δq = Jᵀe with λ = |e|² + 10⁻⁶, from the network output and from the seed, for up to 30 iterations; targets still outside 1 mm and 1 degree get 16 random restarts.
PyBullet then checks the result independently: the pose error, the joint limits and self-collision. If the solution collides, another IK branch is searched from fresh random starts and the first one PyBullet confirms is kept.
Measured on 2,000 held-out targets: 100 % within 1 mm and 1 degree, 12.9 ms per target from a warm start on one CPU thread.
Code: ui/server.py infer(), ui/runners/ik_metrics.refine_ik, refine/lm_solver.py.

## Slide 13 · The robot reaching held-out targets

Demonstration (about 1 min).
Three held-out test poses. For each, the green ghost is the network's guess, which misses by 56, 32, 31 millimetres; the solid arm then executes the refined answer, which PyBullet confirms below 0.001 mm, without self-collision and inside the joint limits.
The input panel shows the actual 13 numbers given to the network for each target.
Clip: simulations/inference_demo.mp4 (Simulation 8). Data: export_sim_data.py inference.

## Slide 14 · Testing on 2,000 unseen targets

Evaluation method and accuracy (about 1 min).
Targets: 2,000 poses from the held-out test split of the best run, restricted to dataset rows neither network saw in training.
Every method receives the same target and the same starting joints. Three starting conditions: warm (true joints plus 0.3-radian noise, as in training), the home pose, and a random pose.
Every answer is re-checked in PyBullet: forward-kinematics pose error, joint limits, self-collision. An answer is valid only if it passes all three.
Left: how close each method gets, warm start. The network alone reaches 50 mm on 95 % of targets but never 1 mm; classical DLS reaches 1 mm on 26 %; network + LM on 100 %.
Right: valid answers for every starting condition. LM alone, without the network, also reaches 100 %: the precision comes from the solver; the network's value is a better starting point.
Source: results/evaluation_metrics.csv, scripts/pipeline/evaluate_methods.py.

## Slide 15 · Neural vs classical vs hybrid: the trade-off

Comparison (about 1.5 min).
Cold start: all three methods start from the home pose, on the same 2,000 targets.
The network alone is fast, about one millisecond, but it is not precise, and from a cold start 6.4 % of its answers collide.
PyBullet's classical damped-least-squares solver is the fastest, and from the home pose it gets within 5 cm of 68 % of targets, but it knows nothing about joint limits or self-collision: 22.8 % of its answers collide and 39.3 % break a joint limit; only 2.6 % are valid within 1 mm and 1 degree.
Network plus LM refinement with PyBullet checks: 100 % valid, no collisions, no limit violations, at 37.8 ms per target on one CPU thread.
Honest caveat: the LM solver alone also reaches 100 %. The network makes it about a quarter faster (50.5 → 37.8 ms) and cuts random restarts from 57 % to 19 % of targets.
From a warm start it is the other way round: LM alone is faster (8.7 vs 12.9 ms) because the seed is already close. The network pays off when the arm starts far from the answer.
Clip: simulations/method_comparison.mp4 (Simulation 9). Source: results/evaluation_metrics.csv (seed_mode = zero).

## Slide 16 · Level 2: steering around the unexpected

Level 2 (about 1.5 min).
Level 2 is implemented and trained in simulation. A PPO agent outputs a correction for each of the six joints, added to a classical potential-field controller (residual reinforcement learning), so it only has to learn when and how to deviate.
It observes 101 numbers: joint state, goal error, end-effector and link positions, and for each obstacle its distance, velocity and a kinematics-aware escape direction. Its reward favours progress, penalises getting close to obstacles, collisions, timeouts and the Level 1 classifier's self-collision risk.
The policy has 11,078 parameters and needs 17.5 microseconds per decision on one CPU core, in plain NumPy.
In the hardest scenario, everything at once, over 200 unseen episodes with identical seeds: the agent reached 84.0 % of goals with 2.0 % collisions; the potential field 71.0 % / 2.5 %, and the straight path 50.0 % / 50.0 %. The agent still gets stuck in 14.0 % of episodes.
Reproduction for this presentation (20 M steps on a CPU, same code and settings): 81.0 % reached, 2.0 % collisions on the same scenario, within 3 points of the team's agent in all six scenarios. The straight-path and potential-field numbers reproduce exactly. The clip shows this reproduced agent, because the team's agent file is not in the repository.
Clip: simulations/level2_ppo_rerouting.mp4 (Simulation 10). Source: media/data/l2_20261013-185525_xarm6_meta_n200.json; level2/*.py.

## Slide 17 · What we achieved, and what is still open

Conclusion (about 1.5 min).
Achievements: a complete and reproducible Level 1 pipeline; every one of 2,000 held-out targets solved within 1 mm and 1 degree with no self-collision and no limit violation; a collision classifier that is 98.4 % accurate and is used both in training and in the Level 2 reward; and a Level 2 agent that cuts collisions from 50 % (straight path) to 2 %.
Limitations: everything is simulation with kinematic motion; the xArm6 stands in for the space manipulator and there is no satellite body yet; self-collision is checked per configuration, not along a continuous path; warm-start results assume the arm starts near the answer; the network alone is not precise, the precision comes from the solver; the Level 2 agent still gets stuck in 14 % of the hardest episodes.
Next steps: the space manipulator's URDF with the satellite body as collision geometry; continuous path checking; an IK-branch fallback when the agent is stuck; dynamics and a free-floating base.
Thank the audience and open questions.
