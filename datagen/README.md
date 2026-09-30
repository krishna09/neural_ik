# datagen/ – alternative dataset generators

Every `.py` file here appears in the UI (**Generate Dataset → Generator**) next to the base
generator `generate_dataset.py`, which stays unchanged. Files starting with `_` are ignored.

| File | What it does |
|---|---|
| `clean_generator.py` | Builds a self-collision matrix before sampling – the way MoveIt's Setup Assistant does – so that link pairs which touch by construction are not counted as collisions: rigidly attached links, links one movable joint apart, and pairs touching in ≥ 95 % of random poses. Samples the joint space with a scrambled Sobol sequence plus 20 % near the joint limits. Made for 7-axis arms with grippers: on the Panda the base rule labels **100 %** of poses colliding (hand bolted to link 7, closed fingers touching), this generator about **17–25 %**, while real collisions such as link 5 ↔ link 7 are still detected. For the xArm6 and KUKA iiwa it gives the same collision rate as the base rule (on the same poses). |
| `precision_generator.py` | For the **< 1 mm / < 1°** goal, used with `modelset/precision/` and `refine/`. It uses the clean collision matrix (so it works for the xArm6, KUKA and Panda) and Sobol coverage of the joint space. `near_limit_frac` (default 15 %) of the samples have every joint within the outer 15 % of its range. Quaternions are stored with w ≥ 0, so q and −q never both appear. The optional `clearance_mm` also labels poses closer than that to a self-collision as colliding. Output keys are the same as `generate_dataset.py`. Speed: about 500 k xArm6 samples in 30 s. |
| `chatgpt_generator_code_v1.py` | Keeps only collision-safe samples in the dataset: a pose is rejected if two non-parent/child links touch **or** come closer than `clearance_mm`. The rejected poses are stored separately in `rejected_*` arrays. It samples uniformly, with `oversample_frac` of the samples pushing only some joints near their limits. It also saves extra fields (`orientations_6d`, `sample_weights`, `split`, …). In the UI it runs as it is (see below). **Panda:** tick *Skip link pairs that touch by construction*, otherwise every sample is rejected. |

## How the UI runs a generator

The file is never edited – the UI adapts to what `generate()` offers:

| `generate()` has … | The UI … |
|---|---|
| `robot_name, n_samples` (first two parameters) | required |
| `seed` | passes the Seed field |
| `out_path` | passes `data/<robot>_ik_dataset.npz`; without it the file must end up there (relative to the working directory, which is the new library entry) or at the path `generate()` returns |
| `progress` | uses it for the live 3D view; without it the live view follows the file's own `set_joint_config`, `get_ee_pose` and collision check (`is_self_colliding` / `check_self_collision` / …) |
| other keyword parameters with a bool / int / float / str default | shows them as **Generator options** (`OPTION_HELP = {name: text}` adds tooltips) |
| reads `cfg["self_collision_ignore_pairs"]` | offers *Skip link pairs that touch by construction*: the pairs from the clean collision matrix are added to that list |

**Movement split.** *Natural* keeps the file exactly as generated. With a collision %, the UI draws the requested numbers of
collision-free and colliding samples from the generator's output. Rows in `rejected_joints` / `rejected_positions` /
`rejected_orientations` count as colliding when `rejected_reason` is 1 (real contact); "too close only" rows are left out.
Such a re-drawn file keeps only the keys training needs.

If the first 3,000 samples are all rejected, the job stops with a message instead of waiting for the
generator's own attempt limit.

## Adding a generator

The simplest complete form:

```python
def generate(robot_name, n_samples, seed=0, out_path=None, progress=None):
    ...                                   # writes out_path (default data/<robot>_ik_dataset.npz)
    return out_path
```

- The `.npz` must have the same keys as `generate_dataset.py`: `joints`, `positions`,
  `orientations`, `collision_labels`, `joint_limits`, `dof` – then `train_ik.py` reads it unchanged.
- Call `progress(i, n_samples, q, pos, orn, collided, phase)` after each sample to get the live 3D
  view and progress bar in the UI.
- Optional module constants: `DESCRIPTION` (shown in the UI; otherwise the docstring's first paragraph),
  `ID_TAG` (short tag added to dataset ids), `LAST_REPORT` (a dict saved in the dataset's
  `meta.json` and summarised in the UI) and `OPTION_HELP`.
