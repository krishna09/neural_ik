# modelset/ – IK network architectures

Every `.py` model file in this folder and its sub-folders appears in the UI
(**Train Model → Model**), grouped by folder, next to the base model from `train_ik.py`.
Press **↻** after adding or editing a file.

Similar models live together in one folder, and each folder has a `README.md` that explains its
models. `modelexp.md` (this folder) is the experiment guide: the goal, what limits accuracy today,
and how to compare the models fairly.

| Folder | Models | What they have in common |
|---|---|---|
| [`custom/`](custom/README.md) | `model_set.py`, `base_512.py` | Your own models |
| [`mlp/`](mlp/README.md) | M01, M02, M09, M10 | MLP variants: residual, wide direct, densely connected, SwiGLU with a seed-trust gate |
| [`conditioned/`](conditioned/README.md) | M03, M04 | The target pose is encoded specially: Fourier features, FiLM conditioning |
| [`structured/`](structured/README.md) | M05, M06, M07, M08 | Networks shaped like the IK problem: iterative solver, kinematic chain, joint transformer, mixture of experts |
| [`paper_iros2024_lbe/`](paper_iros2024_lbe/README.md) | 6 `paper_*` models | Reproduction of the IROS 2024 Learning-by-Example paper (MLP / RMLP / DMLP, with and without the example) |
| [`precision/`](precision/README.md) | `precision_ik_net.py` | **For < 1 mm / < 1°:** a robust starting-point network, finished by the LM solver in `refine/` |

None of the models in `mlp/`, `conditioned/`, `structured/` and `paper_iros2024_lbe/` has been
trained or tested for accuracy; they only pass the UI's load check (build + one dummy batch, for
6- and 7-joint arms).

## What a model file must provide

Every model is trained by the unchanged `train_ik.main()` loop (same loss, optimiser, batch size
and splits); only the network class is swapped in. So a file needs a class with the same
interface as `train_ik.IKNet`:

```python
import torch
import torch.nn as nn

class IKNet(nn.Module):                 # any nn.Module name works; "IKNet" is picked first
    def __init__(self, dof):            # dof = number of arm joints (6 xArm6, 7 KUKA / Panda)
        ...
    def forward(self, x):               # x: (B, 7 + dof) = [x, y, z, qx, qy, qz, qw, seed joints]
        ...                             # -> (B, 2 * dof) = [sin q1..q_dof, cos q1..q_dof]
    def decode_angles(self, raw_out):   # (B, 2 * dof) -> (B, dof) joint angles in radians
        ...
```

Optional: a module-level `DESCRIPTION = "..."` is shown in the UI.

## Adding a model

1. Put the file in the folder of its family, or create a new folder with a short `README.md`
   explaining the idea.
2. Keep each file self-contained: a training run stores a copy of that one file only, so imports
   of other files in this folder would be missing when an old run is reloaded. (Imports of
   project modules such as `robot_config` are fine.)
3. Files and folders starting with `_` or `.` are ignored.

The UI checks each file (builds it and runs a dummy batch) and shows the error next to the file if
it does not fit. Moving or editing a file never breaks older runs – each run reloads its own copy.
