"""
Level 2 DRL agent: PPO (Schulman et al. 2017) with a small MLP policy, written in
plain PyTorch (no RL library needed), plus a NumPy-only policy for deployment.

    policy  pi(a | o) = N( tanh(MLP(o_norm)), diag(sigma^2) )      a in [-1, 1]^n
    value   V(o)      = MLP(o_norm)
    PPO loss          = -E[ min(r A, clip(r, 1-eps, 1+eps) A) ] + c_v (V - R)^2 - c_e H
    r = pi(a|o) / pi_old(a|o),   A = GAE(gamma, lambda) advantage

Environments run in worker processes (one PyBullet client each) so training uses
several CPU cores. The policy itself is tiny (2 x 64 hidden units, ~9k weights in
the actor), which is what "low compute overhead" asks for: one decision is a few
small matrix products (tens of microseconds on a CPU, no GPU needed).
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from .env import ReachAvoidEnv, ensure_cnet_cache
from .vec import VecEnv

HIDDEN = 64


# --------------------------------------------------------------------------- model
class RunningNorm:
    def __init__(self, dim):
        self.mean, self.var, self.count = np.zeros(dim), np.ones(dim), 1e-4

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0), len(x)
        d = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + d * bc / tot
        self.var = (self.var * self.count + bv * bc + d ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -8, 8)


def mlp(i, o, h=HIDDEN):
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=HIDDEN):
        super().__init__()
        self.actor = mlp(obs_dim, act_dim, hidden)
        self.critic = mlp(obs_dim, 1, hidden)
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.0))
        with torch.no_grad():
            self.actor[-1].weight.mul_(0.01)

    def dist(self, o):
        mu = torch.tanh(self.actor(o))
        return torch.distributions.Normal(mu, self.log_std.exp())

    def value(self, o):
        return self.critic(o).squeeze(-1)


# --------------------------------------------------------------------------- deploy
class NumpyPolicy:
    """The trained actor as plain NumPy: no PyTorch / GPU needed on the target computer."""

    def __init__(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["actor"]
        self.W = [sd[f"{i}.weight"].numpy().astype(np.float32) for i in (0, 2, 4)]
        self.b = [sd[f"{i}.bias"].numpy().astype(np.float32) for i in (0, 2, 4)]
        self.mean = np.asarray(ck["obs_mean"], np.float32)
        self.std = np.sqrt(np.asarray(ck["obs_var"], np.float32) + 1e-8)
        self.meta = {k: ck[k] for k in ("robot", "obs_dim", "act_dim", "hidden", "env_cfg") if k in ck}
        self.n_params = sum(w.size for w in self.W) + sum(b.size for b in self.b)

    def act(self, obs):
        x = np.clip((obs - self.mean) / self.std, -8, 8)
        h = np.tanh(self.W[0] @ x + self.b[0])
        h = np.tanh(self.W[1] @ h + self.b[1])
        return np.tanh(self.W[2] @ h + self.b[2])


# --------------------------------------------------------------------------- training
def train(robot="xarm6", total_steps=3_000_000, n_envs=80, seed=0, out_dir=None, on_progress=None,
          should_stop=None, hidden=HIDDEN, env_kw=None, horizon=128, epochs=6, minibatch=2048, workers=None,
          lr=3e-4, gamma=0.995, lam=0.95, clip=0.2, ent_coef=0.0, vf_coef=0.5, device=None,
          on_live=None, live_every=4):
    """Train a PPO policy; saves policy.pt (+ history.json) into out_dir. Returns the history."""
    from .env import DEFAULTS
    # stored with the policy: how it must be run (new agents use the richer observation, v2)
    env_kw = dict({"base": DEFAULTS["base"], "obs_version": 2}, **(env_kw or {}))
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Where each part runs (measured on the RTX 3060 laptop under WSL2):
    #   PPO updates (batched gradient steps on ~8k samples)  -> GPU
    #   PyBullet simulation (CPU-only library)               -> CPU worker processes
    #   per-step policy inference on 64 observations         -> CPU copy of the policy
    #     (a GPU call costs ~5 ms launch/sync latency here, the CPU copy ~1 ms)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_num_threads(2)
    ensure_cnet_cache(robot)                                  # workers read the classifier without torch
    venv = VecEnv(robot, n_envs, seed, env_kw, workers or 10)
    obs_dim, act_dim = venv.obs_dim, venv.act_dim
    ac = ActorCritic(obs_dim, act_dim, hidden).to(dev)
    ac_roll = ActorCritic(obs_dim, act_dim, hidden)          # CPU copy for rollouts
    ac_roll.load_state_dict({k: v.cpu() for k, v in ac.state_dict().items()})
    T = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(ac.parameters(), lr=lr)
    norm = RunningNorm(obs_dim)
    o = venv.reset(seed)
    norm.update(o)
    history, steps, t0 = [], 0, time.time()
    n_updates = max(1, total_steps // (horizon * n_envs))
    best_success = -1.0

    def save(tag="policy.pt"):
        if not out_dir:
            return
        os.makedirs(out_dir, exist_ok=True)
        cpu_sd = lambda m: {k: v.detach().cpu() for k, v in m.state_dict().items()}
        torch.save({"actor": cpu_sd(ac.actor), "critic": cpu_sd(ac.critic), "log_std": ac.log_std.detach().cpu(),
                    "obs_mean": norm.mean, "obs_var": norm.var, "robot": robot, "obs_dim": obs_dim, "act_dim": act_dim,
                    "hidden": hidden, "env_cfg": env_kw or {}, "steps": steps}, os.path.join(out_dir, tag))

    try:
        for upd in range(1, n_updates + 1):
            if should_stop and should_stop():
                break
            frac = 1 - (upd - 1) / n_updates
            for g in opt.param_groups:
                g["lr"] = lr * frac
            buf_o = np.zeros((horizon, n_envs, obs_dim), np.float32)
            buf_a = np.zeros((horizon, n_envs, act_dim), np.float32)
            buf_lp = np.zeros((horizon, n_envs), np.float32)
            buf_r = np.zeros((horizon, n_envs), np.float32)
            buf_d = np.zeros((horizon, n_envs), np.float32)
            buf_v = np.zeros((horizon, n_envs), np.float32)
            finished = []
            for t in range(horizon):
                on = norm(o).astype(np.float32)
                with torch.no_grad():
                    ot = torch.from_numpy(on)
                    dist = ac_roll.dist(ot)
                    a = dist.sample()
                    lp = dist.log_prob(a).sum(-1)
                    v = ac_roll.value(ot)
                a_np = a.cpu().numpy()
                o2, r, d, infos = venv.step(a_np)
                buf_o[t], buf_a[t], buf_lp[t], buf_r[t], buf_d[t], buf_v[t] = on, a_np, lp.cpu().numpy(), r, d, v.cpu().numpy()
                finished += [i for i in infos if "outcome" in i]
                o = o2
                norm.update(o)
            steps += horizon * n_envs
            with torch.no_grad():
                last_v = ac_roll.value(torch.from_numpy(norm(o).astype(np.float32))).numpy()
            # GAE advantages
            adv = np.zeros_like(buf_r)
            gae = np.zeros(n_envs, np.float32)
            for t in reversed(range(horizon)):
                nv = last_v if t == horizon - 1 else buf_v[t + 1]
                nonterm = 1.0 - buf_d[t]
                delta = buf_r[t] + gamma * nv * nonterm - buf_v[t]
                gae = delta + gamma * lam * nonterm * gae
                adv[t] = gae
            ret = adv + buf_v
            B = horizon * n_envs
            fo, fa = T(buf_o.reshape(B, -1)), T(buf_a.reshape(B, -1))
            flp, fadv, fret = (T(x.reshape(B)) for x in (buf_lp, adv, ret))
            fadv = (fadv - fadv.mean()) / (fadv.std() + 1e-8)
            for _ in range(epochs):
                perm = torch.randperm(B, device=dev)
                for s in range(0, B, minibatch):
                    idx = perm[s:s + minibatch]
                    dist = ac.dist(fo[idx])
                    lp = dist.log_prob(fa[idx]).sum(-1)
                    ratio = (lp - flp[idx]).exp()
                    pg = -torch.min(ratio * fadv[idx], ratio.clamp(1 - clip, 1 + clip) * fadv[idx]).mean()
                    vl = ((ac.value(fo[idx]) - fret[idx]) ** 2).mean()
                    loss = pg + vf_coef * vl - ent_coef * dist.entropy().sum(-1).mean()
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
                    opt.step()
            ac_roll.load_state_dict({k: v.detach().cpu() for k, v in ac.state_dict().items()})
            n_fin = max(1, len(finished))
            rate = lambda k: 100.0 * sum(i["outcome"] == k for i in finished) / n_fin
            rec = {
                "update": upd, "updates": n_updates, "steps": steps, "secs": round(time.time() - t0, 1),
                "fps": round(steps / (time.time() - t0)), "episodes": len(finished),
                "success_pct": rate("success"), "collision_pct": rate("collision"), "timeout_pct": rate("timeout"),
                "self_collision_pct": 100.0 * sum(i.get("collision") == "self" for i in finished) / n_fin,
                "ep_return": float(np.mean([i["ep_return"] for i in finished])) if finished else None,
                "ep_len": float(np.mean([i["ep_len"] for i in finished])) if finished else None,
                "value_loss": float(vl.item()), "std": float(ac.log_std.exp().mean().item()), "device": str(dev),
                "envs": n_envs, "workers": len(venv.conns),
            }
            history.append(rec)
            if on_progress:
                on_progress(rec)
            if on_live and (upd == 1 or upd % live_every == 0 or upd == n_updates):
                on_live(_live_episode(robot, env_kw, ac_roll, norm, seed + upd), upd, n_updates)
            # keep the best policy seen so far (smoothed over the last 5 updates)
            sm = np.mean([h["success_pct"] - h["collision_pct"] for h in history[-5:]])
            if upd > 5 and sm > best_success:
                best_success = sm
                save("policy.pt")
        save("policy_last.pt")
        if out_dir and not os.path.exists(os.path.join(out_dir, "policy.pt")):
            save("policy.pt")
        if out_dir:
            with open(os.path.join(out_dir, "history.json"), "w") as f:
                json.dump(history, f)
    finally:
        venv.close()
    return history


_LIVE_ENV = {}


def _live_episode(robot, env_kw, ac, norm, seed):
    """One recorded episode of the current (deterministic) policy, for the Studio's live view."""
    from .evaluate import run_episode
    key = (robot, json.dumps(env_kw, sort_keys=True))
    if key not in _LIVE_ENV:
        _LIVE_ENV[key] = ReachAvoidEnv(robot, seed=seed, **env_kw)
    env = _LIVE_ENV[key]

    class _Ctrl:
        def act(self, obs):
            with torch.no_grad():
                x = torch.from_numpy(norm(obs).astype(np.float32))
                return torch.tanh(ac.actor(x)).numpy()

    ep = run_episode(env, _Ctrl(), seed, None, record=True)
    env.reset(seed=seed)
    env._set_q(env.q_goal)
    ep["goal_links"] = env.link_frames()
    ep["goal_ee"] = [round(float(v), 4) for v in env.ee_goal]
    return ep
