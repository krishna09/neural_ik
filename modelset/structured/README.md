# structured/ – networks shaped like the IK problem

Four networks whose structure mirrors something about inverse kinematics: iterating like a
numerical solver, following the kinematic chain, modelling joint-to-joint couplings, or
specialising per workspace region. All use the shared design choices described in
[`../modelexp.md`](../modelexp.md#design-choices-shared-by-mlp-conditioned-and-structured) and
predict zero-initialised corrections to the seed.

| # | File | Architecture | Params (6-DOF) | Output |
|---|---|---|---|---|
| M05 | `m05_iterative_refiner.py` | 4 unrolled, weight-shared refinement steps (ResMLP 3 × 512) | 3.40 M | seed + Σ Δₖ |
| M06 | `m06_kinematic_chain.py` | Trunk 4 × 512 + one head per joint, conditioned on earlier joints | 5.41 M | seed + Δ per joint |
| M07 | `m07_joint_transformer.py` | Transformer, 4 layers, d = 128, pose token + joint tokens | 0.81 M | seed + Δ per token |
| M08 | `m08_mixture_of_experts.py` | Shared encoder + softmax gate over 8 experts | 3.71 M | seed + gated Δ |

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
