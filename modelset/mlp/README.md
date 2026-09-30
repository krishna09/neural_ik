# mlp/ – MLP variants

Four fully connected networks that differ in how layers are connected. All use the shared design
choices described in [`../modelexp.md`](../modelexp.md#design-choices-shared-by-mlp-conditioned-and-structured)
(6D rotation input, seed as sin / cos, zero-initialised seed correction, unit-circle output).

| # | File | Architecture | Params (6-DOF) | Output |
|---|---|---|---|---|
| M01 | `m01_seed_residual_mlp.py` | Pre-norm ResMLP, 6 blocks × 384 | 3.56 M | seed + Δ |
| M02 | `m02_direct_wide_mlp.py` | Wide MLP 4 × 1024, LayerNorm + SiLU | 3.19 M | direct angles |
| M09 | `m09_dense_skip_mlp.py` | DenseNet-style MLP, 8 layers × 128 growth | 0.76 M | seed + Δ |
| M10 | `m10_swiglu_seed_gate.py` | SwiGLU ResNet 8 × 384, seed-trust gate between two heads | 7.12 M | blend of seed + Δ and direct |

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

### M09 – Dense skip MLP
Each of 8 layers sees the input plus all earlier layers' outputs. **Why:** the raw pose and seed
stay visible at every depth (like NeRF's skip connection) – good for precise regression, and small
(0.76 M). **Watch:** the concatenated width grows with depth, so deeper versions get slow.

### M10 – SwiGLU with a seed-trust gate
Deep SwiGLU residual blocks with two heads – seed + Δ and a direct prediction – blended per joint
by a learned gate (starts at ~5 % direct). **Why:** keeps the accuracy of the residual form when the
seed is good, and can lean on the direct head when it is not – the most robust option for real use
with poor seeds. **Watch:** the largest model (7.1 M); slowest per epoch.
