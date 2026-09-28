# modelset/ – your IK network architectures

Every `.py` file in this folder appears in the UI (**Train Model → Model**) next to the
base model from `train_ik.py`. Press **↻** after adding or editing a file. Files starting
with `_` are ignored. Keep each model file self-contained: a training run stores a copy of
that one file only, so imports of other files in this folder would be missing when an old
run is reloaded. See `modelexp.md` for the ten architectures `m01` … `m10`.

The selected network is trained by the unchanged `train_ik.main()` loop, so a file only
needs a class with the same interface as `train_ik.IKNet`:

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

The UI checks each file (builds it and runs a dummy batch) and shows the error next to the
file if it does not fit. Every training run stores a copy of the model file it was trained
with, so editing or deleting a file here never breaks older runs.
