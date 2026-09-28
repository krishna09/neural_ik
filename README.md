# Neural IK for 6DOF/7DOF Manipulators (STP Team 06 - Level 1)

Tested, working pipeline for the Level 1 scope: solve inverse kinematics for
a 6DOF and a 7DOF arm in PyBullet using a neural network, with a
self-collision-aware training objective.

## Setup

```bash
pip install -r requirements.txt
```

Both robots used here (xarm6 = 6DOF, kuka_iiwa = 7DOF) ship inside the
`pybullet_data` package, so no external URDF or dataset download is needed
to reproduce this. A Franka Panda config (7DOF) is also included.

## Files

- `robot_config.py` - robot registry (URDF path + joint names) and the
  `load_robot()` helper. Joints are resolved **by name**, not by hardcoded
  index - see the comment there about the `xarm6` fixed `world_joint` bug
  this avoids.
- `generate_dataset.py` - samples random joint configurations, computes the
  resulting end-effector pose via PyBullet's forward kinematics, and labels
  each sample as self-colliding or not using PyBullet's collision checker.
- `train_ik.py` - trains (1) a small collision classifier network and (2)
  the main IK network (pose + seed config -> joint angles), using the
  classifier as an auxiliary loss so the IK network is pushed away from
  self-colliding solutions during training, not just filtered afterward.
- `evaluate.py` - compares three approaches on held-out targets: the neural
  net alone, PyBullet's classical damped-least-squares IK (given the same
  warm-start "seed" configuration, for a fair comparison), and a hybrid
  (neural warm start + a handful of classical refinement iterations).

## Running it end to end

```bashStep 3 prepares the information that the neural network will learn from. **The seed, sine/cosine, and `atan2` have three different jobs:**

| Item | Its job |
|---|---|
| **Seed** | Gives the model a starting arm configuration to consider. |
| **Sine and cosine** | Represent each joint angle in a form that handles circular angles better during training. |
| **`atan2`** | Converts the predicted sine/cosine pair back into an angle. |

I’ll use **degrees** in the examples because they are easier to understand. Your code uses **radians**: 180° = π radians.

**The seed is an initial guess for the joint angles.**

Imagine reaching for a cup. You can often keep your hand at the same place while changing the position of your elbow.

A robot can also have different joint-angle combinations that put its gripper at the same target pose. This is why an IK problem can have multiple answers. :chatgpt-content-reference{index="0"}

If we give the model only:

> “Put the gripper at this position and orientation.”

There may be several valid answers.

Adding a seed provides more information:

> “Put the gripper at this position and orientation, preferably using a joint configuration close to these starting angles.”

The seed helps distinguish between different solutions. **It encourages a nearby solution; it does not guarantee one.**

For example, suppose a generated training sample has these correct joint angles:

```python
true_angles = [30, -60, 45, 0, 60, -30]   # degrees, for explanation
```

The program uses FK to calculate the target gripper pose for those angles. It then makes a slightly disturbed copy:

```python
seed_angles = [35, -52, 40, 6, 55, -38]
```

The training example effectively says:

> “Here is the target gripper pose. Here is a starting guess. The answer I want you to learn is `[30, -60, 45, 0, 60, -30]`.”

The model learns to use **both the target pose and the starting guess** to predict suitable angles.

Without the seed, a simple regression network can struggle when similar target poses have very different correct joint configurations. It may produce a compromise between solutions, and that compromise may fail to reach the target.

**During actual use, the seed usually comes from the robot’s current joint angles.**

This distinction is particularly important:

| Situation | Where the seed comes from |
|---|---|
| Training in your example | Known correct angles + artificial random noise |
| Evaluation in your example | Known correct angles + artificial random noise |
| Actual robot operation | Measured current joint angles, or the previous IK solution |

In actual operation, you do **not** know the correct target angles yet. Those are what IK must find.

For a sequence of small movements, the previous joint configuration is often a useful starting guess for the next target. This is also a standard approach in numerical IK. :chatgpt-content-reference{index="1"}

However, your evaluation supplies a seed already close to a known answer. Therefore, its results mainly demonstrate **solving IK with a nearby starting guess**. They do not establish performance when the robot starts far away.

A seed is not compulsory for every possible IK method. It is an input chosen for **this model design**.

---

**Sine and cosine are used because an angle goes around a circle.**

Consider these two angles:

- +179°
- −179°

Their ordinary numerical difference is:

\[
179-(-179)=358^\circ
\]

But their directions are only **2° apart around the circle**.

If we train a network using ordinary squared error directly on these angle numbers, predicting −179° when the answer is +179° receives a huge penalty, even though the directions are nearly identical.

Sine and cosine avoid this artificial jump in the representation.

Imagine a pointer of length 1 rotating around a circle:

- **Cosine** tells you its horizontal position.
- **Sine** tells you its vertical position.

Together, they describe the pointer’s direction.

Try **+179°** and **−179°** below. The point barely moves, even though the written angle changes sharply.



::chatgpt-content-reference{index="2" is_streaming="false"}



**Both values are needed because either one alone can be ambiguous.**

For example:

\[
\sin(30^\circ)=0.5,\qquad \sin(150^\circ)=0.5
\]

If the model predicts only `0.5` for sine, we cannot tell whether it means 30° or 150°.

Cosine distinguishes them:

\[
\cos(30^\circ)\approx+0.866,\qquad
\cos(150^\circ)\approx-0.866
\]

Therefore:

```python
30°  is represented by  [0.5,  0.866]   # [sin, cos]
150° is represented by  [0.5, -0.866]
```

Cosine alone also has ambiguity: +30° and −30° have the same cosine. Their sine values distinguish them.

**The pair identifies the angular direction within one complete turn.**

That is why your six-joint model produces **12 numbers**:

```python
[
    sin_joint1, sin_joint2, sin_joint3,
    sin_joint4, sin_joint5, sin_joint6,

    cos_joint1, cos_joint2, cos_joint3,
    cos_joint4, cos_joint5, cos_joint6
]
```

There are still only six joint angles. Each angle is represented using two numbers.

This is a useful design choice, rather than a requirement of IK itself. Other angle representations and loss functions are possible.

---

**`atan2` recovers the angle from those two numbers.**

Your code uses **`atan2`**, pronounced “a-tan-two.”

Its meaning here is:

> “Given the vertical and horizontal components, find the direction angle.”

The argument order matters:

```python
angle = atan2(vertical, horizontal)
```

Because vertical = sine and horizontal = cosine:

```python
angle = atan2(sin_value, cos_value)
```

NumPy calls this function `np.arctan2`. It uses both signs to identify the correct part of the circle and returns radians between −π and +π. :chatgpt-content-reference{index="3"}

For example:

```python
atan2(0.5,  0.866)   # approximately 30°, after conversion to degrees
atan2(0.5, -0.866)   # approximately 150°, after conversion to degrees
```

The **“2” refers to its two inputs**.

Ordinary inverse tangent, `atan`, receives only one number, usually a ratio. Dividing the components can lose information: `(1, 1)` and `(-1, -1)` both give a ratio of `1`, although they point in opposite directions. `atan2` receives the two components separately and can distinguish them.

Here is a complete, small Python example:

```python
import math

# Start with a known angle.
original_degrees = 150
theta = math.radians(original_degrees)

# Represent it using two numbers.
sin_value = math.sin(theta)
cos_value = math.cos(theta)

# Recover the angle from those two numbers.
recovered_radians = math.atan2(sin_value, cos_value)
recovered_degrees = math.degrees(recovered_radians)

print(round(sin_value, 3))          # 0.5
print(round(cos_value, 3))          # -0.866
print(round(recovered_degrees, 1))  # 150.0
```

In this demonstration, we calculate the sine and cosine ourselves. **In your trained model, the network predicts those two values.** `atan2` then turns its predictions into an angle.

`atan2` does not solve the IK problem by itself. It only decodes the network’s output.

---

Now we can read **your Step 3** directly.

For one xArm6 training example, the input contains:

```python
X = [
    x, y, z,                   # Desired gripper position: 3 numbers
    qx, qy, qz, qw,             # Desired gripper orientation: 4 numbers
    seed1, seed2, seed3,
    seed4, seed5, seed6         # Starting joint-angle guess: 6 numbers
]
```

That gives **3 + 4 + 6 = 13 input numbers**.

Here, `qx, qy, qz, qw` describe the gripper’s orientation. They are **not** the six joint angles.

The training target contains:

```python
y = [
    sin(true_joint1), ..., sin(true_joint6),
    cos(true_joint1), ..., cos(true_joint6)
]
```

That gives **12 target numbers**.

During training, the network adjusts its weights so its 12 predictions get closer to those 12 target values.

During use, the conversion is:

```python
raw_output = ik_net(X)

sin_predictions = raw_output[:6]  # First six numbers
cos_predictions = raw_output[6:]  # Last six numbers

predicted_joint_angles = np.arctan2(
    sin_predictions,
    cos_predictions
)
```

The result is **six predicted joint angles in radians**. These are then checked using FK to see whether they actually put the gripper at the requested pose.

One practical detail remains: sine/cosine represents direction, so it cannot distinguish angles separated by a complete revolution, such as 30° and 390°. It also does not enforce mechanical joint limits. **The decoded angles still need joint-limit and collision checks; nearby directions do not necessarily mean the robot can move directly between them.**
python generate_dataset.py --robot xarm6 --n 20000
python train_ik.py --robot xarm6 --epochs 60
python evaluate.py --robot xarm6 --n 200

python generate_dataset.py --robot kuka_iiwa --n 20000
python train_ik.py --robot kuka_iiwa --epochs 60
python evaluate.py --robot kuka_iiwa --n 200
```

## What the evaluation actually showed (run on this dataset)

|                         | pos err | orient err | self-collision | inference time |
|-------------------------|--------:|-----------:|----------------:|----------------:|
| Neural IK alone (xarm6) | 128 mm  | 33.7 deg   | 0.0%            | 0.07 ms |
| Classical DLS (xarm6)   | 5 mm    | 0.04 deg   | 9.0%            | 0.23 ms |
| Hybrid (xarm6)          | 45 mm   | 0.73 deg   | 6.5%            | 0.12 ms |

The plain feedforward network is fast but not accurate enough on its own -
this matches what the robotics/ML literature generally finds for pure
feedforward IK nets, and is worth stating plainly in the report rather than
overselling it. The practically useful result is the **hybrid**: the
network's output as a warm start for a *short* classical refinement gets
most of the accuracy back at roughly half the runtime of running the
classical solver to full convergence, and the collision-aware training
measurably reduces self-collision rate versus the collision-blind classical
solver. That three-way tradeoff (speed vs. accuracy vs. safety) is the
central finding to build the Level 1 report and diagrams around.

## Extending to the actual space manipulator later

Swapping in the real manipulator once its URDF/kinematic model exists is a
matter of adding one entry to `ROBOTS` in `robot_config.py` (URDF path +
joint names in kinematic order) - nothing else in the pipeline needs to
change, which is why it was worth validating this on standard arms first.
