"""
Self-collision matrix for the UI's own collision checks: the red link highlight
in the 3D view, the dataset "FK re-check", and the self-collision figures in
training / evaluation / inference.

It uses datagen/clean_generator.build_collision_matrix, so the UI judges a pose
exactly like the clean generator labels it: link pairs that touch by
construction (rigidly attached, one movable joint apart, touching in >= 95 % of
random poses) are ignored. For the xArm6 and KUKA iiwa this changes nothing
measurable; on the Panda it stops the hand and the closed fingers from always
showing as colliding. Datasets themselves are untouched - their labels come
from whichever generator made them.
"""

import generator_registry as GR

CLEAN_ID = "datagen/clean_generator.py"


def apply(body, cfg, limits):
    """Disable structural link pairs on `body` (default PyBullet client). Returns
    the matrix report, or None if the clean generator is not available."""
    try:
        builder = getattr(GR.load(CLEAN_ID), "build_collision_matrix", None)
    except Exception:
        return None
    if builder is None:
        return None
    return builder(body, cfg, limits, seed=0)
