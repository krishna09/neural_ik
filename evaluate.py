"""
Level 1 - Step 3: Evaluation - neural IK vs PyBullet's classical IK.

For a held-out set of target poses, this compares:
  - Position error (meters) and orientation error (degrees) between the
    requested pose and the pose actually reached by the predicted joints
    (re-verified via forward kinematics, not just trusted from the model).
  - Self-collision rate of the predicted configuration.
  - Joint-limit violation rate.
  - Wall-clock inference time per solution.

The classical baseline is PyBullet's built-in calculateInverseKinematics
(damped least squares / SDLS). This produces the head-to-head numbers you
need for the write-up: where the network wins (speed, redundancy
consistency, built-in collision awareness) and where the classical solver
still wins (raw accuracy on a single pose with unlimited iterations).
"""

import argparse
import time

import numpy as np
import pybullet as p
import torch

from robot_config import connect, load_robot
from train_ik import IKNet


def quat_angle_deg(q1, q2):
    """Angle (deg) between two orientation quaternions."""
    dot = np.abs(np.dot(q1, q2))
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(2 * np.arccos(dot))


def forward_kinematics(body_id, joint_indices, ee_link_index, q):
    for idx, angle in zip(joint_indices, q):
        p.resetJointState(body_id, idx, angle)
    state = p.getLinkState(body_id, ee_link_index, computeForwardKinematics=True)
    return np.array(state[4]), np.array(state[5])


def check_violations(body_id, joint_indices, ee_link_index, q, joint_limits):
    for idx, angle in zip(joint_indices, q):
        p.resetJointState(body_id, idx, angle)
    p.performCollisionDetection()
    colliding = len(p.getContactPoints(bodyA=body_id, bodyB=body_id)) > 0
    limit_violation = any(
        angle < lo - 1e-3 or angle > hi + 1e-3
        for angle, (lo, hi) in zip(q, joint_limits)
    )
    return colliding, limit_violation


def run_eval(robot_name, n_targets=200, seed=1):
    rng = np.random.default_rng(seed)
    connect(gui=False)
    body_id, cfg, joint_limits = load_robot(robot_name)
    joint_indices = cfg["joint_indices"]
    ee_link_index = cfg["ee_link_index"]
    dof = cfg["dof"]

    # Build target poses from *known-reachable* random configs, so both
    # solvers are being asked for something that has a real solution. Each
    # target also carries one fixed "seed" configuration (a noisy version
    # of the true joints, standing in for wherever the arm currently is in
    # a trajectory) that BOTH solvers receive identically below - otherwise
    # a difference in warm-start information, not solver quality, would
    # explain any gap in the results.
    targets = []
    for _ in range(n_targets):
        q_true = np.array([rng.uniform(lo, hi) for lo, hi in joint_limits])
        pos, orn = forward_kinematics(body_id, joint_indices, ee_link_index, q_true)
        seed_q = q_true + rng.normal(0, 0.3, size=dof)
        targets.append((pos, orn, q_true, seed_q))

    # ---- Neural IK ----
    ik_net = IKNet(dof)
    ik_net.load_state_dict(torch.load(f"models/{robot_name}_ik_net.pt"))
    ik_net.eval()

    neural_results = {"pos_err": [], "orn_err": [], "colliding": [], "limit_violation": [], "time": []}
    with torch.no_grad():
        for pos, orn, q_true, seed_q in targets:
            x = torch.tensor(
                np.concatenate([pos, orn, seed_q]).astype(np.float32)
            ).unsqueeze(0)

            t0 = time.perf_counter()
            raw_out = ik_net(x)
            q_pred = ik_net.decode_angles(raw_out).squeeze(0).numpy()
            dt = time.perf_counter() - t0

            reached_pos, reached_orn = forward_kinematics(body_id, joint_indices, ee_link_index, q_pred)
            colliding, limit_violation = check_violations(body_id, joint_indices, ee_link_index, q_pred, joint_limits)

            neural_results["pos_err"].append(np.linalg.norm(reached_pos - pos))
            neural_results["orn_err"].append(quat_angle_deg(reached_orn, orn))
            neural_results["colliding"].append(colliding)
            neural_results["limit_violation"].append(limit_violation)
            neural_results["time"].append(dt)

    # ---- Classical IK (PyBullet damped least squares) ----
    # Seeded with the SAME "previous configuration" info given to the neural
    # net (via currentPositions), so this is an apples-to-apples comparison
    # of what each solver does with identical warm-start information - not
    # a plain-DLS-from-zero straw man. Deliberately NOT given joint limits
    # or collision info: that is the whole point of the comparison, since
    # a classical numerical IK solver is blind to both unless you bolt on
    # extra constraint-handling logic, whereas our network was trained to
    # respect them.
    classical_results = {"pos_err": [], "orn_err": [], "colliding": [], "limit_violation": [], "time": []}

    for pos, orn, q_true, seed_q in targets:
        for idx, angle in zip(joint_indices, seed_q):
            p.resetJointState(body_id, idx, angle)

        t0 = time.perf_counter()
        q_pred = p.calculateInverseKinematics(
            body_id, ee_link_index, pos, orn,
            currentPositions=list(seed_q),
            maxNumIterations=100, residualThreshold=1e-5,
        )
        dt = time.perf_counter() - t0
        q_pred = np.array(q_pred[:dof])

        reached_pos, reached_orn = forward_kinematics(body_id, joint_indices, ee_link_index, q_pred)
        colliding, limit_violation = check_violations(body_id, joint_indices, ee_link_index, q_pred, joint_limits)

        classical_results["pos_err"].append(np.linalg.norm(reached_pos - pos))
        classical_results["orn_err"].append(quat_angle_deg(reached_orn, orn))
        classical_results["colliding"].append(colliding)
        classical_results["limit_violation"].append(limit_violation)
        classical_results["time"].append(dt)

    def summarize(name, r):
        print(f"\n-- {name} --")
        print(f"  mean position error:     {np.mean(r['pos_err'])*1000:.2f} mm")
        print(f"  mean orientation error:  {np.mean(r['orn_err']):.2f} deg")
        print(f"  self-collision rate:     {np.mean(r['colliding'])*100:.1f}%")
        print(f"  joint-limit violation:   {np.mean(r['limit_violation'])*100:.1f}%")
        print(f"  mean inference time:     {np.mean(r['time'])*1e3:.3f} ms")

    # ---- Hybrid: neural prediction as warm start, then a SHORT classical
    # refinement (few iterations instead of the full 100) ----
    # This is the practical recipe worth recommending: the network gets you
    # close in one fast forward pass, and a handful of DLS iterations from
    # that starting point cleans up the residual error - cheaper than
    # running DLS to convergence from an arbitrary/rest starting pose, and
    # far more accurate than the raw network output alone.
    hybrid_results = {"pos_err": [], "orn_err": [], "colliding": [], "limit_violation": [], "time": []}
    with torch.no_grad():
        for pos, orn, q_true, seed_q in targets:
            x = torch.tensor(np.concatenate([pos, orn, seed_q]).astype(np.float32)).unsqueeze(0)

            t0 = time.perf_counter()
            raw_out = ik_net(x)
            q_neural = ik_net.decode_angles(raw_out).squeeze(0).numpy()

            for idx, angle in zip(joint_indices, q_neural):
                p.resetJointState(body_id, idx, angle)
            q_pred = p.calculateInverseKinematics(
                body_id, ee_link_index, pos, orn,
                currentPositions=list(q_neural),
                maxNumIterations=5, residualThreshold=1e-5,
            )
            dt = time.perf_counter() - t0
            q_pred = np.array(q_pred[:dof])

            reached_pos, reached_orn = forward_kinematics(body_id, joint_indices, ee_link_index, q_pred)
            colliding, limit_violation = check_violations(body_id, joint_indices, ee_link_index, q_pred, joint_limits)

            hybrid_results["pos_err"].append(np.linalg.norm(reached_pos - pos))
            hybrid_results["orn_err"].append(quat_angle_deg(reached_orn, orn))
            hybrid_results["colliding"].append(colliding)
            hybrid_results["limit_violation"].append(limit_violation)
            hybrid_results["time"].append(dt)

    print(f"\n=== {robot_name} ({dof}DOF) - {n_targets} held-out targets ===")
    summarize("Neural IK alone", neural_results)
    summarize("Classical IK (PyBullet DLS, 100 iters, same warm start)", classical_results)
    summarize("Hybrid: neural warm start + 5-iter DLS refinement", hybrid_results)

    return neural_results, classical_results, hybrid_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="xarm6", choices=["xarm6", "kuka_iiwa", "panda"])
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()
    run_eval(args.robot, args.n)
