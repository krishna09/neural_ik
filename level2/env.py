"""
Level 2 environment: real-time re-routing of a manipulator around unexpected
obstacles and hardware position errors (STP Team 06).

Task. The arm starts at a configuration and must reach a goal configuration
(in the full system the goal comes from the Level 1 neural IK + LM solver).
Each control step the policy outputs a small joint change. During the motion

  * obstacles (spheres) block the straight path - some are there from the start,
    some appear suddenly on the path ahead of the arm, some drift;
  * hardware position errors happen - an actuator gain / bias error for the whole
    episode, and sudden joint slips (a joint jumps by a random angle).

The policy sees the measured joint state, the goal, the end-effector and link
positions and, per obstacle, where it is and how close it is to the robot surface.
It must re-route on the fly and still arrive.

The arm is moved kinematically (resetJointState), like the Level 1 code. Every
env has its own PyBullet client, so many envs can run in one process.
"""

import os
import sys

import numpy as np
import pybullet as p
import pybullet_data

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from robot_config import ROBOTS  # noqa: E402

SUPPORTED = ("xarm6", "kuka_iiwa")   # the Panda needs the clean collision matrix (Level 1 datagen), not used here

DEFAULTS = {
    "max_steps": 300,
    "dq_max": 0.03,            # rad per step at |action| = 1  (~1.7 deg)
    "n_obstacles": 3,          # obstacle slots in the observation
    "d_safe": 0.08,            # m, clearance below which the proximity penalty starts
    "goal_tol_rad": 0.06,      # max |joint error| to count as arrived
    "goal_tol_m": 0.02,        # EE distance to count as arrived
    "p_static": 0.8,           # chance of 1-2 obstacles on the path from the start
    "p_sudden": 0.7,           # chance of an obstacle appearing mid-motion on the path ahead
    "p_moving": 0.3,           # chance that obstacles drift
    "p_slip": 0.5,             # chance of a sudden joint slip during the episode
    "slip_std": 0.25,          # rad
    "gain_err": 0.2,           # actuator gain error ~ U(-g, g)
    "bias_std": 0.004,         # rad per step actuator bias
    "start_spread": 1.4,       # rad, max per-joint distance start <-> goal
    "use_collision_net": True, # Level 1 collision classifier as a self-collision-risk penalty
    "residual": True,          # action = straight-line controller + learned correction (residual RL)
    "residual_scale": 2.0,     # correction range: can fully reverse the base command
    "base": "apf",             # base controller under the correction: "apf" (potential field) or "straight"
    "obs_version": 1,          # 2 adds obstacle velocities, clearance rates, stall counter and time (new agents)
}


RADII = [0.035, 0.045, 0.055, 0.065, 0.075]   # m, obstacle sphere sizes


def _load_robot(robot, cid):
    cfg = ROBOTS[robot]
    body = p.loadURDF(cfg["urdf"], useFixedBase=True,
                      flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
                      physicsClientId=cid)
    names = {p.getJointInfo(body, j, physicsClientId=cid)[1].decode(): j
             for j in range(p.getNumJoints(body, physicsClientId=cid))}
    idx = [names[n] for n in cfg["joint_names"]]
    lim = []
    for j in idx:
        info = p.getJointInfo(body, j, physicsClientId=cid)
        lo, hi = info[8], info[9]
        if lo >= hi:
            lo, hi = -np.pi, np.pi
        lim.append((lo, hi))
    return body, idx, np.array(lim, dtype=np.float64)


class CollisionNetNumpy:
    """Level 1 collision classifier (train_ik.CollisionNet, 64-64-1) evaluated in NumPy.

    The weights are read from a NumPy cache (level2/.cache/) so simulation workers never need
    PyTorch; ensure_cnet_cache() creates it from models/<robot>_collision_net.pt."""

    def __init__(self, robot):
        z = np.load(ensure_cnet_cache(robot))
        self.w = [z[f"w{i}"] for i in range(3)]
        self.b = [z[f"b{i}"] for i in range(3)]

    def prob(self, q):
        h = np.maximum(self.w[0] @ q + self.b[0], 0)
        h = np.maximum(self.w[1] @ h + self.b[1], 0)
        z = float((self.w[2] @ h + self.b[2])[0])
        return 1.0 / (1.0 + np.exp(-z))


def collision_net_path(robot):
    path = os.path.join(PROJECT_DIR, "models", f"{robot}_collision_net.pt")
    return path if os.path.exists(path) else None


def ensure_cnet_cache(robot):
    src = collision_net_path(robot)
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", f"{robot}_collision_net.npz")
    if not os.path.exists(cache) or os.path.getmtime(cache) < os.path.getmtime(src):
        import torch   # only here, once
        sd = torch.load(src, map_location="cpu")
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, **{f"w{k}": sd[f"net.{i}.weight"].numpy().astype(np.float64) for k, i in enumerate((0, 2, 4))},
                 **{f"b{k}": sd[f"net.{i}.bias"].numpy().astype(np.float64) for k, i in enumerate((0, 2, 4))})
    return cache


class ReachAvoidEnv:
    """Gym-style env: reset(seed) -> obs, step(action) -> obs, reward, done, info."""

    def __init__(self, robot="xarm6", seed=0, **overrides):
        if robot not in SUPPORTED:
            raise ValueError(f"Level 2 supports {SUPPORTED}, not {robot}")
        self.robot = robot
        self.cfg = dict(DEFAULTS, **overrides)
        self.cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        self.body, self.jidx, self.lim = _load_robot(robot, self.cid)
        self.dof = len(self.jidx)
        self.nlinks = p.getNumJoints(self.body, physicsClientId=self.cid)
        self.link_ids = list(range(self.nlinks))
        self.ee = self.jidx[-1]
        self.rng = np.random.default_rng(seed)
        K = self.cfg["n_obstacles"]
        # sphere bodies are pre-built in a few sizes per slot (rebuilding a body is slow);
        # an obstacle uses the pool sphere closest to its sampled radius
        self.pool = []
        for k in range(K):
            slot = {}
            for i, r in enumerate(RADII):
                col = p.createCollisionShape(p.GEOM_SPHERE, radius=r, physicsClientId=self.cid)
                slot[r] = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                            basePosition=[10 + 3 * k, 3 * i, -50], physicsClientId=self.cid)
            self.pool.append(slot)
        self.obs_bodies = [self.pool[k][RADII[0]] for k in range(K)]
        self.obs_radius = np.full(K, RADII[0])
        self.cnet = None
        if self.cfg["use_collision_net"] and collision_net_path(robot):
            self.cnet = CollisionNetNumpy(robot)
        self.obs_dim = len(self.reset(seed=seed))

    # ------------------------------------------------------------------ helpers
    def close(self):
        try:
            p.disconnect(self.cid)
        except p.error:
            pass

    def _set_q(self, q):
        for j, a in zip(self.jidx, q):
            p.resetJointState(self.body, j, float(a), physicsClientId=self.cid)

    def _ee(self):
        return np.array(p.getLinkState(self.body, self.ee, computeForwardKinematics=True,
                                       physicsClientId=self.cid)[4])

    def _link_pos(self):
        sts = p.getLinkStates(self.body, self.link_ids, computeForwardKinematics=True, physicsClientId=self.cid)
        return np.array([s[4] for s in sts])

    def _self_collision(self):
        p.performCollisionDetection(physicsClientId=self.cid)
        return len(p.getContactPoints(bodyA=self.body, bodyB=self.body, physicsClientId=self.cid)) > 0

    def _set_obstacle(self, k, pos, radius):
        r = min(RADII, key=lambda x: abs(x - radius))
        if r != self.obs_radius[k]:
            old = self.obs_bodies[k]
            p.resetBasePositionAndOrientation(old, [10 + 3 * k, 3 * RADII.index(self.obs_radius[k]), -50],
                                              [0, 0, 0, 1], physicsClientId=self.cid)
            self.obs_bodies[k] = self.pool[k][r]
            self.obs_radius[k] = r
        p.resetBasePositionAndOrientation(self.obs_bodies[k], list(pos), [0, 0, 0, 1], physicsClientId=self.cid)
        self.obs_pos[k] = pos

    def _hide_obstacle(self, k):
        self.active[k] = False
        self.obs_vel[k] = 0
        self._set_obstacle(k, np.array([10.0 + 3 * k, 3.0 * RADII.index(self.obs_radius[k]), -50.0]), self.obs_radius[k])

    def _clearance(self, k, max_d=0.5):
        """(distance robot surface -> obstacle k, point on robot, point on obstacle, link)."""
        cps = p.getClosestPoints(self.body, self.obs_bodies[k], max_d, physicsClientId=self.cid)
        if not cps:
            return max_d, None, None, -1, None
        c = min(cps, key=lambda c: c[8])
        return c[8], np.array(c[5]), np.array(c[6]), c[3], np.array(c[7])

    def _min_clearance(self):
        ds = [self._clearance(k)[0] for k in range(len(self.obs_bodies)) if self.active[k]]
        return min(ds) if ds else 0.5

    def _path_points(self, q_from, q_to, n=12, t0=0.15):
        """EE positions along the straight joint-space path (what a naive controller would follow)."""
        pts = []
        for t in np.linspace(t0, 0.85, n):
            self._set_q(q_from + t * (q_to - q_from))
            pts.append(self._ee())
        self._set_q(self.q)
        return pts

    def _sample_q(self):
        lo, hi = self.lim[:, 0], self.lim[:, 1]
        return lo + (hi - lo) * self.rng.random(self.dof)

    def _place_on_path(self, k, q_from, q_to, min_clear=0.10, t0=0.25):
        """Put obstacle k on the remaining straight path, far enough from the arm to be avoidable."""
        pts = self._path_points(q_from, q_to, t0=t0)
        for _ in range(12):
            c = pts[self.rng.integers(len(pts))] + self.rng.normal(0, 0.025, 3)
            r = self.rng.uniform(RADII[0], RADII[-1])
            if c[2] - r < 0.05:         # keep above the floor / base plate
                continue
            self._set_obstacle(k, c, r)
            self.active[k] = True
            # the goal pose itself keeps a margin from every obstacle (Level 1 picks collision-free goals)
            if self._clearance(k)[0] > min_clear and self._clear_at(self.q_goal, k, margin=0.12):
                if self.rng.random() < self.p_moving_ep:
                    self.obs_vel[k] = self.rng.normal(0, 0.0025, 3)
                return True
        self._hide_obstacle(k)
        return False

    def _clear_at(self, q, k, margin=0.03):
        self._set_q(q)
        ok = self._clearance(k)[0] > margin
        self._set_q(self.q)
        return ok

    # ------------------------------------------------------------------ API
    def reset(self, seed=None, scenario=None, start=None, goal=None):
        """scenario (optional, for evaluation): dict with keys static / sudden / slip / moving (bools).
        start / goal (optional): fixed joint configurations, e.g. the goal solved by Level 1 IK."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c = self.cfg
        K = len(self.obs_bodies)
        self.active = np.zeros(K, bool)
        self.obs_pos = np.zeros((K, 3))
        self.obs_vel = np.zeros((K, 3))
        for k in range(K):
            self._hide_obstacle(k)
        # start and goal: both self-collision free, start within start_spread of the goal
        for _ in range(200):
            g = np.clip(np.asarray(goal, float), self.lim[:, 0], self.lim[:, 1]) if goal is not None else self._sample_q()
            s = (np.clip(np.asarray(start, float), self.lim[:, 0], self.lim[:, 1]) if start is not None else
                 np.clip(g + self.rng.uniform(-c["start_spread"], c["start_spread"], self.dof), self.lim[:, 0], self.lim[:, 1]))
            if start is not None and goal is not None:
                self._set_q(g)
                ee_g = self._ee()
                break
            self._set_q(g)
            if self._self_collision():
                continue
            ee_g = self._ee()
            self._set_q(s)
            if self._self_collision():
                continue
            if np.linalg.norm(self._ee() - ee_g) > 0.15 and ee_g[2] > 0.1:
                break
        self.q_goal, self.q = g, s.copy()
        self._set_q(self.q)
        self.ee_goal = ee_g
        sc = scenario or {}
        flag = lambda name, prob: sc.get(name, self.rng.random() < prob)
        self.scenario = {"static": flag("static", c["p_static"]), "sudden": flag("sudden", c["p_sudden"]),
                         "slip": flag("slip", c["p_slip"]), "moving": bool(sc.get("moving", False))}
        self.p_moving_ep = 1.0 if self.scenario["moving"] else c["p_moving"]
        k = 0
        if self.scenario["static"]:
            for _ in range(1 + int(self.rng.random() < 0.4)):
                if k < K and self._place_on_path(k, self.q, self.q_goal):
                    k += 1
        # events are timed relative to how long the straight motion would take
        n_est = int(np.ceil(np.max(np.abs(self.q_goal - self.q)) / c["dq_max"]))
        self.sudden_at = int(self.rng.integers(2, max(3, n_est // 2))) if self.scenario["sudden"] and k < K else -1
        self.slip_at = int(self.rng.integers(3, max(4, (4 * n_est) // 5))) if self.scenario["slip"] else -1
        self.gain = 1 + self.rng.uniform(-c["gain_err"], c["gain_err"], self.dof)
        self.bias = self.rng.normal(0, c["bias_std"], self.dof)
        self.t = 0
        self.prev_a = np.zeros(self.dof)
        self.events = []
        self._dist = self._goal_dist()
        self._best_dist, self._stall = self._dist, 0
        self._clr = {}
        self._update_clearances()
        return self._obs()

    def _update_clearances(self):
        self._prev_clr = {k: v[0] for k, v in getattr(self, "_clr", {}).items()}
        self._clr = {k: self._clearance(k) for k in range(len(self.obs_bodies)) if self.active[k]}
        self._esc = {k: self._escape(*v) for k, v in self._clr.items()}

    def _escape(self, d, pa, pb, link, normal, reach=0.25):
        """Kinematics-aware escape direction in joint space: g = J(closest point)^T n.

        J is the 3 x n position Jacobian of the robot point closest to the obstacle and n the
        unit normal from the obstacle towards that point, so moving the joints along g pushes the
        robot away fastest. Returned unit-length, weighted by w = 1 - d / reach (0 when far)."""
        w = float(np.clip(1.0 - d / reach, 0.0, 1.0))
        if pa is None or link < 0 or w == 0.0:
            return np.zeros(self.dof), 0.0
        cid = self.cid
        ls = p.getLinkState(self.body, link, computeForwardKinematics=True, physicsClientId=cid)
        ipos, iorn = p.invertTransform(ls[0], ls[1])                      # world -> link COM frame
        local = p.multiplyTransforms(ipos, iorn, pa.tolist(), [0, 0, 0, 1])[0]
        zeros = [0.0] * self.dof
        jlin, _ = p.calculateJacobian(self.body, link, list(local), [float(v) for v in self.q], zeros, zeros,
                                      physicsClientId=cid)
        g = np.array(jlin).T @ normal
        n = np.linalg.norm(g)
        return (g / n if n > 1e-9 else np.zeros(self.dof)), w

    def _goal_dist(self):
        return float(np.linalg.norm(self.q_goal - self.q) + 2.0 * np.linalg.norm(self.ee_goal - self._ee()))

    def _obs(self):
        lo, hi = self.lim[:, 0], self.lim[:, 1]
        qn = 2 * (self.q - lo) / (hi - lo) - 1
        dq = (self.q_goal - self.q) / np.pi
        links = self._link_pos()
        ee = links[self.ee]
        parts = [qn, dq, ee, self.ee_goal - ee, links[1:].reshape(-1) - np.tile(ee, len(links) - 1), self.prev_a]
        for k in range(len(self.obs_bodies)):
            v2 = self.cfg["obs_version"] >= 2
            if not self.active[k]:
                parts.append(np.zeros(9 + self.dof + (4 if v2 else 0)))
                continue
            d, pa, pb, link, _ = self._clr[k]
            vec = (pb - pa) if pa is not None else (self.obs_pos[k] - ee)
            n = np.linalg.norm(vec) + 1e-9
            g, w = self._esc[k]
            feat = [self.obs_pos[k] - ee, vec / n * min(d, 0.5), [min(d, 0.5), self.obs_radius[k], 1.0], w * g]
            if v2:   # obstacle velocity (m/step x 100) and how fast the clearance is shrinking (m/step x 20)
                prev = self._prev_clr.get(k, d)
                feat += [100.0 * self.obs_vel[k], [20.0 * float(np.clip(min(d, 0.5) - min(prev, 0.5), -0.05, 0.05))]]
            parts.append(np.concatenate(feat))
        if self.cfg["obs_version"] >= 2:   # "am I stuck?" and "how much time is left?"
            parts.append([min(self._stall / 50.0, 2.0), self.t / self.cfg["max_steps"]])
        return np.concatenate(parts).astype(np.float32)

    def step(self, action, absolute=False):
        """action in [-1, 1]^n. With residual=True it is a correction added to the straight-line
        controller (zero action = baseline behaviour); otherwise it is the joint command itself.
        absolute=True: the action is the joint command (used by the baselines)."""
        c = self.cfg
        a_pol = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        if absolute or not c["residual"]:
            a, a_pol = a_pol, np.zeros_like(a_pol)
        else:
            base = apf_action(self) if c["base"] == "apf" else straight_line_action(self)
            a = np.clip(base + c["residual_scale"] * a_pol, -1, 1)
        self.t += 1
        info = {}
        # hardware: actuator executes the command with a gain / bias error
        dq = c["dq_max"] * a * self.gain + self.bias
        q_new = np.clip(self.q + dq, self.lim[:, 0], self.lim[:, 1])
        # hardware: sudden joint slip
        if self.t == self.slip_at:
            j = int(self.rng.integers(self.dof))
            q_new[j] = np.clip(q_new[j] + self.rng.normal(0, c["slip_std"]), self.lim[j, 0], self.lim[j, 1])
            self.events.append({"t": self.t, "type": "joint_slip", "joint": j})
        self.q = q_new
        self._set_q(self.q)
        # obstacles drift / appear
        for k in range(len(self.obs_bodies)):
            if self.active[k] and np.any(self.obs_vel[k]):
                pos = self.obs_pos[k] + self.obs_vel[k]
                pos[2] = max(pos[2], 0.08)
                self._set_obstacle(k, pos, self.obs_radius[k])
        if self.t == self.sudden_at:
            free = [k for k in range(len(self.obs_bodies)) if not self.active[k]]
            if free and self._place_on_path(free[0], self.q, self.q_goal, min_clear=0.12, t0=0.35):
                self.events.append({"t": self.t, "type": "obstacle_appeared", "slot": free[0]})
        # reward
        d_new = self._goal_dist()
        if d_new < self._best_dist - 0.01:          # real progress towards the goal
            self._best_dist, self._stall = d_new, 0
        else:
            self._stall += 1
        r = 3.0 * (self._dist - d_new) - 0.01 - 0.02 * float(a_pol @ a_pol) - 0.02 * float(np.sum((a - self.prev_a) ** 2))
        self._dist = d_new
        self._update_clearances()
        clear = min((v[0] for v in self._clr.values()), default=0.5)
        if clear < c["d_safe"]:
            r -= 1.0 * (c["d_safe"] - clear) / c["d_safe"]
        if self.cnet is not None:
            pc = self.cnet.prob(self.q)
            r -= 0.3 * pc
            info["collision_risk"] = pc
        self.prev_a = a
        done = False
        hit_obs = clear <= 0.0
        hit_self = self._self_collision()
        ee_err = float(np.linalg.norm(self.ee_goal - self._ee()))
        if hit_obs or hit_self:
            r -= 25.0
            done = True
            info["outcome"] = "collision"
            info["collision"] = "obstacle" if hit_obs else "self"
        elif np.max(np.abs(self.q_goal - self.q)) < c["goal_tol_rad"] and ee_err < c["goal_tol_m"]:
            r += 15.0
            done = True
            info["outcome"] = "success"
        elif self.t >= c["max_steps"]:
            r -= 10.0                    # stuck (e.g. in a local minimum) is a failure too
            done = True
            info["outcome"] = "timeout"
        info.update(clearance=clear, ee_err=ee_err, t=self.t)
        return self._obs(), r, done, info

    # ------------------------------------------------------------------ for the UI
    def link_frames(self):
        """World pose of every link's URDF frame, same format as the Studio's /api/fk 'links'."""
        cid = self.cid
        bpos, born = p.getBasePositionAndOrientation(self.body, physicsClientId=cid)
        dyn = p.getDynamicsInfo(self.body, -1, physicsClientId=cid)
        ipos, iorn = p.invertTransform(dyn[3], dyn[4])
        lpos, lorn = p.multiplyTransforms(bpos, born, ipos, iorn)
        r = lambda v: [round(float(x), 5) for x in v]
        frames = {"-1": [r(lpos), r(lorn)]}
        for j, s in enumerate(p.getLinkStates(self.body, self.link_ids, computeForwardKinematics=True, physicsClientId=cid)):
            frames[str(j)] = [r(s[4]), r(s[5])]
        return frames

    def obstacles(self):
        return [{"pos": [round(float(v), 4) for v in self.obs_pos[k]], "r": round(float(self.obs_radius[k]), 4)}
                for k in range(len(self.obs_bodies)) if self.active[k]]


def straight_line_action(env):
    """Baseline: move straight towards the goal in joint space (no obstacle awareness).

    This is what a planner that only uses the Level 1 IK answer and interpolates would do.
    """
    d = env.q_goal - env.q
    return np.clip(d / env.cfg["dq_max"], -1, 1)


def apf_action(env, gain=2.0):
    """Classical baseline: artificial potential field = straight line + repulsion along the
    kinematics-aware escape directions of nearby obstacles (Khatib 1986 style, in joint space)."""
    a = straight_line_action(env)
    for g, w in env._esc.values():
        a = a + gain * w * w * g
    return np.clip(a, -1, 1)
