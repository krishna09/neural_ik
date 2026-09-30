# custom/ – your own models

| File | Network | Params (6-DOF) |
|---|---|---|
| `model_set.py` | Residual MLP: input 256 → expand to 512 (LayerNorm, GELU, dropout 0.05) → 3 residual blocks of 512 (SiLU / GELU / LeakyReLU, LayerNorm, dropout 0.05) → compress 384 → 256 → 128 → `[sin, cos]` output; `decode_angles` normalises each sin/cos pair before `atan2` | 2.05 M |
| `base_512.py` | The base `train_ik.IKNet` layout at width 512: 512 → 512 → 256, ReLU, raw `[sin, cos]` output | 0.40 M |

Both take the raw input `[xyz, quaternion, seed joints]` as it comes from `train_ik`.
