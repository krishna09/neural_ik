# level2/ – real-time re-routing with Deep Reinforcement Learning

Level 2 of the Team 06 project: *a lightweight DRL policy agent, optimised for low compute
overhead, that dynamically re-routes the manipulator path in real time when unexpected
geometric obstacles or hardware position errors occur.*

Level 1 (neural IK + LM refinement) turns a target pose into a goal joint configuration.
Level 2 drives the arm from where it is to that goal while the world changes under it.

| File | What it does |
|---|---|
| `env.py` | PyBullet environment `ReachAvoidEnv` (xArm6, KUKA iiwa): obstacles, hardware errors, reward |
| `ppo.py` | PPO trainer (plain PyTorch), parallel CPU simulation, GPU updates, `NumpyPolicy` for deployment |
| `evaluate.py` | Agent vs. straight-line baseline on identical seeded scenarios; recorded episodes for the UI |
| `train.py` | Command-line training |

## Task

Each episode: a start and a goal configuration (both self-collision free). During the motion:

* **Unexpected obstacles** – spheres (3.5–7.5 cm radius) placed on the straight path from the start,
  a sphere that **appears suddenly** on the path ahead mid-motion, and drifting spheres.
* **Hardware position errors** – an actuator gain error (±20 %) and bias for the whole episode, and a
  **sudden joint slip** (one joint jumps by N(0, 0.25 rad)).

Success = every joint within 0.06 rad of the goal **and** end effector within 2 cm.
Failure = contact with an obstacle or self-collision, or 300 steps without arriving.

## Agent

* **Residual RL:** command = potential-field controller + 2 × policy correction, clipped to ±1, then
  × 0.03 rad per step. With a zero correction the agent *is* the potential field, so it only has to
  learn when and how to deviate – mainly how to get out of the potential field's local minima.
  (`base="straight"` in the env config trains on top of the straight-line controller instead.)
* **Observation:** normalised joints, joint error to the goal, EE position,
  EE-to-goal vector, link positions relative to the EE, previous action, and per obstacle slot:
  position relative to the EE, the closest-point vector robot → obstacle, clearance, radius, active flag,
  and the kinematics-aware escape direction w·Jᵀn (Jacobian at the closest robot point × obstacle normal).
  87 numbers for the xArm6.
* **Reward:** progress towards the goal − time − action size/jerk − proximity (clearance < 8 cm)
  − 0.3 · σ(C(θ)) with **C = the Level 1 collision classifier** − 25 on collision + 15 on success − 10 on timeout.
* **Policy:** MLP 2 × 64 (tanh), 10,182 actor parameters (xArm6). Deployed as plain NumPy (`NumpyPolicy`):
  no PyTorch, no GPU – one decision is three small matrix products.

## Compute

Training: PyBullet is CPU-only, so 80 environments run in 10 spawned worker processes (NumPy +
PyBullet only, ~380 MB each); the PPO gradient
updates run on the GPU (CUDA when available). Per-step inference during rollouts uses a CPU copy of
the policy, because for 80 tiny observations a GPU call costs more in launch latency than it saves.

## Results (xArm6, 200 unseen episodes per scenario, identical seeds)

Best agent `l2_20261013-185525_xarm6`: 20 M steps (134 min, PPO updates on the GPU), observation v2.

| Scenario | DRL agent success / collisions | Previous agent (5 M, obs v1) | Potential field | Straight path |
|---|---|---|---|---|
| Everything at once | **84.0 % / 2.0 %** | 78.0 % / 2.5 % | 71.0 % / 2.5 % | 50.0 % / 50.0 % |
| Obstacles from the start | **89.5 % / 1.5 %** | 83.0 % / 1.0 % | 79.0 % / 1.5 % | 55.0 % / 45.0 % |
| Random mix | **88.5 % / 2.0 %** | 79.0 % / 2.0 % | 77.5 % / 2.5 % | 57.0 % / 43.0 % |
| Sudden obstacle | 94.0 % / 1.5 % | 94.5 % / 0.5 % | 91.0 % / 1.0 % | 80.5 % / 19.5 % |
| Drifting obstacles | 94.5 % / 2.0 % | 94.0 % / 2.0 % | **95.5 %** / 2.5 % | 50.0 % / 50.0 % |
| Joint slip | 98.5 % / 1.5 % | 98.5 % / 1.5 % | 98.5 % / 1.5 % | 98.5 % / 1.5 % |

11,078 parameters, 17.5 µs per decision (NumPy, one CPU core). Remaining failures are mostly timeouts (stuck), not collisions.

**Observation v2** (`obs_version: 2`, used by new agents): per obstacle also its velocity and how fast its clearance is
shrinking; globally the number of steps without progress (“am I stuck?”) and the elapsed time. 101 numbers for the xArm6.

## Run

```bash
nik                                                  # project folder + GPU venv (WSL)
python -m level2.train --robot xarm6 --steps 5000000
python -m level2.evaluate --policy level2/runs/<id>/policy.pt --n 200
```

Or use the Studio: **Level 2 · DRL re-routing** tab → Train agent → Test scenario → Evaluate.
Agents trained in the Studio are stored in `ui/workspace/level2/policies/`.
