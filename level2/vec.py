"""Parallel PyBullet environments for training: several envs per worker process."""

import multiprocessing as mp
import os

import numpy as np

from .env import ReachAvoidEnv


def _worker(conn, robot, seeds, env_kw):
    envs = [ReachAvoidEnv(robot, seed=sd, **env_kw) for sd in seeds]
    ep_ret, ep_len = [0.0] * len(envs), [0] * len(envs)
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "step":
                out = []
                for i, (env, a) in enumerate(zip(envs, data)):
                    o, r, d, info = env.step(a)
                    ep_ret[i] += r
                    ep_len[i] += 1
                    if d:
                        info = dict(info, ep_return=ep_ret[i], ep_len=ep_len[i])
                        ep_ret[i], ep_len[i] = 0.0, 0
                        o = env.reset()
                    out.append((o, r, d, info))
                conn.send(out)
            elif cmd == "reset":
                ep_ret, ep_len = [0.0] * len(envs), [0] * len(envs)
                conn.send([env.reset(seed=data + 7919 * i) for i, env in enumerate(envs)])
            elif cmd == "spaces":
                conn.send((envs[0].obs_dim, envs[0].dof))
            elif cmd == "close":
                break
    finally:
        for env in envs:
            env.close()
        conn.close()


class VecEnv:
    """n environments spread over worker processes (several envs per worker = fewer round trips)."""

    def __init__(self, robot, n, seed=0, env_kw=None, workers=None):
        # "spawn": fresh worker processes that import only NumPy + PyBullet (not the
        # trainer's PyTorch / CUDA memory) - forked copies used ~680 MB each
        ctx = mp.get_context("spawn")
        workers = max(1, min(n, workers or max(1, (os.cpu_count() or 4) - 4)))
        split = np.array_split(np.arange(n), workers)
        self.conns, self.procs, self.sizes = [], [], []
        for ids in split:
            a, b = ctx.Pipe()
            pr = ctx.Process(target=_worker, args=(b, robot, [seed + 1000 * int(i) for i in ids], env_kw or {}), daemon=True)
            pr.start()
            b.close()
            self.conns.append(a)
            self.procs.append(pr)
            self.sizes.append(len(ids))
        self.conns[0].send(("spaces", None))
        self.obs_dim, self.act_dim = self.conns[0].recv()
        self.n = n

    def reset(self, seed=0):
        for i, c in enumerate(self.conns):
            c.send(("reset", seed + 101 * i))
        return np.concatenate([np.stack(c.recv()) for c in self.conns])

    def step(self, actions):
        k = 0
        for c, m in zip(self.conns, self.sizes):
            c.send(("step", actions[k:k + m]))
            k += m
        res = [x for c in self.conns for x in c.recv()]
        o, r, d, info = zip(*res)
        return np.stack(o), np.array(r, np.float32), np.array(d), list(info)

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for pr in self.procs:
            pr.join(timeout=2)
            if pr.is_alive():
                pr.terminate()


