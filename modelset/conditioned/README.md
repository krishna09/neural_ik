# conditioned/ – special encodings of the target pose

Two networks that change how the target pose enters the network. Both use the shared design
choices described in [`../modelexp.md`](../modelexp.md#design-choices-shared-by-mlp-conditioned-and-structured)
and predict a zero-initialised correction to the seed.

| # | File | Architecture | Params (6-DOF) | Output |
|---|---|---|---|---|
| M03 | `m03_fourier_features.py` | Random Fourier features of the pose + ResMLP 5 × 512 | 5.40 M | seed + Δ |
| M04 | `m04_film_pose_conditioned.py` | Pose encoder FiLM-modulates a seed trunk, 6 × 384 | 4.88 M | seed + Δ |

### M03 – Fourier features + ResMLP
The 9 pose numbers are also mapped to 128 random frequencies [sin(2πBv), cos(2πBv)] (σ = 2,
fixed buffer saved with the model) before the ResMLP. **Why:** MLPs are biased towards smooth
functions; Fourier features let them fit the fine detail millimetre accuracy needs (Tancik et al.
2020). **Watch:** too many high frequencies can overfit on small datasets – use a dataset of
200 k samples or more.

### M04 – FiLM pose conditioning
A pose encoder produces a scale γ and shift β for every block of a trunk that processes the seed
(LN(h)·(1 + γ) + β). **Why:** turns "correct this seed" into "correct this seed *for this target*"
at every depth instead of only at the input. FiLM layers start as identity.
