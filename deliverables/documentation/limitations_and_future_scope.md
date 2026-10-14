# Limitations and future scope

The results in the deck and the video are real, but they hold under specific conditions. This page
states those conditions plainly, with the numbers behind them, so the results are not read as more
than they are. Numbers come from `results/evaluation_metrics.csv` (2,000 held-out targets, xArm6),
`results/eval_summary.json` and the Level 2 evaluation files.

## 1. Scope of everything shown

- **Simulation only.** PyBullet, kinematic motion: joint angles are set directly; there are no
  torques, no dynamics and no control loop running on hardware.
- **xArm6 stands in for the space manipulator.** The space arm's URDF is not available yet, so the
  framework was built and validated on a standard 6-DOF arm. There is no satellite body, no
  free-floating base and no external obstacle in Level 1 (self-collision only).

## 2. Level 1: neural inverse kinematics

| Limitation | Evidence | Consequence |
|---|---|---|
| **The network alone is not precise.** | Warm start: 22.9 mm / 5.8° mean error, 0 % of targets within 1 mm and 1°. | The network is a fast first guess. Precision comes from the LM solver. |
| **The network alone is weak from a cold start.** | Home pose (all joints 0°): 128.8 mm mean, 6.4 % self-collision, 14.5 % joint-limit violations. Random start: 158.9 mm, 12.4 %, 31.5 %. | Without refinement and the PyBullet check its answers are unsafe from a cold start. |
| **The LM solver alone also reaches 100 %.** | LM without the network: 100 % within 1 mm and 1° for every starting pose. | The network's value is speed and fewer restarts, not reachability: from the home pose it cuts random restarts from 57 % to 19 % of targets and time from 50.5 to 37.8 ms per target. From a warm start, LM alone is faster (8.7 vs 12.9 ms). |
| **Warm seeds are optimistic.** | Warm seeds are the true joints plus N(0, 0.3 rad) noise. | They model "the arm is already close to the answer". Home-pose and random seeds are always reported next to them. |
| **Collisions are checked per configuration.** | PyBullet checks the final joint vector; the classifier scores single poses. | A motion between two collision-free poses can still pass through a collision. Continuous checking is future work. |
| **The classifier misses some collisions.** | 98.37 % accuracy, 95.9 % precision, 94.1 % recall on 30,000 held-out poses: about 6 in 100 colliding poses are scored as free. | It is used as a soft training penalty and a risk signal, never as the final safety check. Every answer is re-checked in PyBullet. |
| **The classical baseline uses fixed settings.** | PyBullet DLS, 100 iterations, residual threshold 1e-5, no joint limits or collision information, as in `evaluate.py`. | Its 22.8 % self-collision and 39.3 % limit-violation rates from the home pose describe this configuration, not every classical solver. A limit-aware solver (for example TRAC-IK) would avoid the limit violations; self-collision would still need its own check. |
| **Timings depend on the hardware.** | One target at a time, one CPU thread, 4-core Xeon. Batched: 0.11 ms per target for 2,000 targets. | Compare ratios between methods, not absolute times. The team's training runs used GPUs. |
| **One solution per query.** | The seed decides which of the arm's IK branches is returned (several exist for most poses). | No explicit preference such as maximum clearance or minimum joint travel yet. |

## 3. Level 2: DRL re-routing

| Limitation | Evidence | Consequence |
|---|---|---|
| **Not every episode succeeds.** | Team agent, "everything at once" scenario, 200 unseen episodes: 84.0 % reached the goal, 2.0 % collided, 14.0 % timed out. | The agent reduces collisions sharply (the straight path collides in 50.0 %) but can still get stuck. A fallback (another IK branch, a sampling planner) is needed. |
| **Simple world model.** | Sphere obstacles (static, appearing, drifting); faults modelled as gain and bias errors and a sudden joint slip; kinematic motion. | Real obstacles, contact dynamics and sensor noise are not represented. |
| **Perfect state.** | The policy observes obstacle positions and joint angles exactly. | Perception and estimation errors are not covered. |
| **Training is not bit-reproducible.** | A reproduction with the same code and settings was trained on a CPU for these deliverables. | Its rates differ from the team's agent; both are reported side by side in `results/level2_evaluation.csv`. |

## 4. What a space manipulator adds

These are outside the current implementation and are the main gap between the simulation results and
an on-orbit system:

1. **Free-floating base.** Without a fixed base, arm motion moves and rotates the satellite
   (conservation of momentum). IK must use the generalised Jacobian, and the end-effector pose depends
   on the motion history, not only on the final joint angles.
2. **Satellite body and payloads as collision geometry**, not only the arm itself.
3. **Flight computing.** Radiation-tolerant processors are much slower than the CPU used here; the
   1 ms network and 13–38 ms refinement times must be re-measured on representative hardware.
4. **Verification and validation** to space-software standards, hardware-in-the-loop testing, and a
   real-time operating system.

## 5. Future scope, in order

1. Load the space manipulator's URDF (one entry in `robot_config.ROBOTS`) and add the satellite body as
   collision geometry.
2. Continuous collision checking along paths (interpolated or swept-volume checks) in Level 1.
3. Fallback when the Level 2 agent is stuck: switch to another IK branch or call a sampling-based
   planner.
4. A preference between IK branches (clearance, joint travel), and an uncertainty estimate for the
   network (for example an ensemble) to decide when its guess can be trusted.
5. Free-floating dynamics in simulation (generalised Jacobian, reaction-null-space control).
6. Domain randomisation, sensor noise and hardware-in-the-loop tests; timing on flight-like processors.
