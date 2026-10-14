# Voice-over script — Neural IK, STP Team 06 (≈ 5 min 1 s)

Voice: Kokoro-82M `af_heart` (text-to-speech, Apache-2.0), speed 1.0. Timestamps are from the final video. Every number below is checked against `deliverables/results/` by `scripts/video/narration.py` before audio is generated.

## 00:00 – 00:18 · opening (`opening`)

- **00:00** A robot arm must place its gripper precisely, stay inside its joint limits, and never collide with itself.
- **00:07** This is STP Team 06's collision-aware neural inverse kinematics, demonstrated on a simulated xArm6.

## 00:18 – 00:32 · objectives (`objectives`)

- **00:18** Our project brief has two levels.
- **00:20** Level one: a kinematics-aware model that computes joint angles without self-collision.
- **00:26** Level two: a lightweight reinforcement-learning agent that re-routes the arm in real time.

## 00:32 – 00:50 · fk (`forward_kinematics`)

- **00:33** Forward kinematics is the easy direction.
- **00:36** Each joint rotation is chained onto the next, and PyBullet returns one exact pose for the gripper: a position and an orientation.

## 00:50 – 01:10 · ik (`inverse_kinematics`)

- **00:51** Inverse kinematics runs backwards, from a target pose to the joint angles, and it is harder.
- **00:57** For this target, our solver found six different joint configurations that reach it exactly.
- **01:03** The arm's current joints, called the seed, decide which one we want.

## 01:10 – 01:37 · dataset (`dataset_generation`)

- **01:11** We generate the training data ourselves.
- **01:14** The generator samples random joint angles: seventy percent uniformly, and thirty percent close to the joint limits, where self-collisions concentrate.
- **01:23** PyBullet computes each pose and checks every pair of links for contact.
- **01:27** Our main dataset has two hundred thousand samples; sixteen percent are self-colliding.

## 01:37 – 02:08 · network (`neural_architecture`)

- **01:38** The network reads thirteen numbers: the target position, its orientation as a quaternion, and the six seed angles.
- **01:46** Three hidden layers turn them into twelve outputs, the sine and cosine of every joint.
- **01:51** The atan2 function turns each pair back into a joint angle.
- **01:55** Why sine and cosine? On a circle, plus and minus one hundred and eighty degrees are neighbours, so the training targets have no artificial jump.

## 02:08 – 02:28 · collision (`collision_awareness`)

- **02:09** A small classifier learns to predict self-collision from the joint angles alone; on held-out poses, it is right ninety-eight percent of the time.
- **02:18** During training, its probability is added to the loss, which pushes predictions away from colliding configurations.

## 02:28 – 02:47 · training (`training_validation`)

- **02:29** Training uses the Adam optimiser and a fifteen percent validation split.
- **02:34** The baseline network levels off at about one hundred and fifteen millimetres; a deeper residual network, trained on ten times more data, reaches twenty-three.

## 02:47 – 03:16 · inference (`inference_demo`)

- **02:48** At run time, the network answers in about a millisecond, but on its own it still misses by centimetres.
- **02:55** Its answer becomes the starting point for a Levenberg–Marquardt solver, which removes the remaining error.
- **03:01** PyBullet then checks the result independently: pose error, joint limits and self-collision.
- **03:07** Only an answer that passes every check is accepted.

## 03:16 – 03:57 · comparison (`method_comparison`)

- **03:17** We compared the methods on the same two thousand held-out targets, with the arm starting from its home pose.
- **03:23** The network alone never came within one millimetre. PyBullet's classical solver found a valid answer within one millimetre and one degree for under three percent of targets; nearly a quarter of its answers collided, and thirty-nine percent broke a joint limit.
- **03:39** Network plus refinement reached every target within one millimetre and one degree, with no collisions and no limit violations.
- **03:47** To be fair, the solver alone also gets there; the network makes it about a quarter faster and cuts random restarts threefold.

## 03:57 – 04:14 · workflow (`workflow`)

- **03:58** Level one is complete and reproducible: robot configuration, data generation, collision classifier, network training, refinement and evaluation.
- **04:08** Moving to the real space manipulator means adding one entry to the robot registry.

## 04:14 – 04:47 · level2 (`level2_ppo`)

- **04:15** Level two is implemented as well: a PPO agent with eleven thousand parameters corrects a potential-field controller in real time.
- **04:24** It learns in simulation, rewarded for progress and penalised for getting close to obstacles.
- **04:30** Here, an obstacle appears and a joint slips: the straight path collides, the potential field crawls, and the agent arrives.
- **04:38** Over two hundred unseen episodes of the hardest scenario, it reached eighty-four percent of goals, with two percent collisions.

## 04:47 – 05:00 · closing (`closing`)

- **04:48** Fast neural guesses, exact refinement, independent checks, and a learned controller on top.
- **04:54** Thank you from STP Team 06.
