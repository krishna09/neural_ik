# refine/ – numerical IK refinement (network + solver)

A neural network alone does not reach **< 1 mm and < 1°**: the best network in this project
(M10, 1 M samples, 512 epochs) stops at a 3.8 mm / 1.1° median. Published neural IK methods show
the same limit. The systems that do reach sub-millimetre accuracy use the network only as a
**starting point** and finish with a numerical solver:

- Lu, Zou & Jiang 2022 (xArm6): network + Newton–Raphson, all targets converged to 1e-6 m.
- IKFlow (Ames et al.): about 10 mm from the network, then a fast classical refinement.

This folder is that solver.

| File | What it does |
|---|---|
| `kinematics.py` | Forward kinematics and the 6 × n geometric Jacobian of an arm in PyTorch (batched, CPU / GPU). The chain is read from the URDF through PyBullet and matches PyBullet's FK to about 1e-4 mm / 3e-5° on the xArm6, KUKA iiwa and Panda. |
| `lm_solver.py` | Batched Levenberg–Marquardt: step Δq = (JᵀJ + λI)⁻¹ Jᵀe, where e = [position error; rotation error]. Damping is λ = ‖e‖² + 1e-6 (Sugihara 2011), and joints are clamped to their limits every step. `solve()` tries all starting points at once, keeps the best per target, and adds random restarts only for targets still outside tolerance. |

## How the UI uses it

1. **Network answer:** the network predicts the joints from the target pose and the seed.
2. **Wrap into limits:** each angle is shifted by ±360° to the copy inside the joint limits that is closest to the seed (`wrap_to_limits`). The pose is the same. Without this step, xArm6 joint 3 (−225°…+11°) gets stuck at its limit, because `atan2` only returns −180…180°. That alone took success from 77 % to 99.8 %.
3. **LM refinement:** 30 iterations, starting from the network output and from the seed. The better result per target is kept.
4. **Random restarts:** targets still outside 1 mm / 1° get 16 random restarts.
5. **Independent check:** PyBullet re-checks every solution for FK error, joint limits and self-collision.
6. **Collision-free retry:** if the solution collides, another solution (a different arm posture) is searched from fresh random starts. The first one PyBullet confirms is kept.

Where it runs:

- **Evaluation (Run details → Evaluate):** tick *LM refinement*. Each evaluation gets a second row, "↳ network + LM refinement".
- **End of every training run:** the held-out test split, up to 20 000 targets, is refined automatically. This is done for a warm start (seed = truth + N(0, 0.3)) and a cold start (zero-pose seed).
- **Run Inference:** set Refinement to *LM solver*. It runs on the CPU, so it never competes with a training job.

**Success** in Run details is the strict criterion: position ≤ tolerance **and** orientation ≤
tolerance **and** inside the joint limits **and** collision-free. Set both tolerances to 1 to read the
< 1 mm / < 1° result.

## Measured (xArm6, network M10, 2 510 unseen targets, strict success at 1 mm / 1°)

| Seed | Network alone | Network + LM refinement |
|---|---|---|
| true joints + N(0, 0.3) (warm) | 2.0 % | **100 %** |
| zero pose (cold) | 0 % | **99.9 %** |

The solver takes about 0.03 ms per target on the GPU (2 000 targets at once, warm start, 10
iterations). The UI's timing also includes the PyBullet check.
