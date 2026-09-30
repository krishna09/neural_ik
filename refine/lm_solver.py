"""
Numerical IK refinement: batched damped least squares with Levenberg-Marquardt
damping (Sugihara 2011: lambda = |e|^2 + small), joint limits enforced, on the
GPU or CPU.

The network gives a starting point close to a solution; a few LM iterations
then remove the last millimetres / degrees (the "network + numerical
refinement" scheme of Lu, Zou & Jiang 2022 on the xArm6, IKFlow, etc.).

    solve(chain, pos, quat, starts, ...) -> best refined joints per target

Every call is batched over all targets and all starting points at once.
"""

import math

import numpy as np
import torch

try:
    from .kinematics import quat_to_matrix, rot_err
except ImportError:                    # refine/ itself on sys.path
    from kinematics import quat_to_matrix, rot_err

DEG = math.pi / 180.0


def _t(a, chain):
    return torch.as_tensor(np.asarray(a) if not torch.is_tensor(a) else a, dtype=chain.dtype).to(chain.device)


def wrap_to_limits(q, lo, hi, ref=None):
    """Shift every angle by k * 360 deg (k = -2..2) to the copy that lies inside
    [lo, hi] and is closest to `ref` (default: the angle itself). Same pose,
    legal joint value. Angles with no copy inside the limits are clamped.
    Needed because atan2 decoding returns -180..180 deg while e.g. xArm6
    joint 3 spans -225..+11 deg."""
    ref = q if ref is None else ref
    ks = torch.arange(-2, 3, dtype=q.dtype, device=q.device) * (2 * math.pi)
    cand = q[..., None] + ks
    ok = (cand >= lo[:, None] - 1e-9) & (cand <= hi[:, None] + 1e-9)
    d = torch.where(ok, (cand - ref[..., None]).abs(), torch.full_like(cand, float("inf")))
    out = cand.gather(-1, d.argmin(-1, keepdim=True))[..., 0]
    clamped = torch.maximum(torch.minimum(q, hi), lo)
    return torch.where(ok.any(-1), out, clamped)


def lm(chain, q0, p_t, R_t, iters=30, tol_m=1e-6, tol_rad=1e-5, q_rest=None, rest_weight=0.0):
    """Levenberg-Marquardt on all rows of q0 (B, dof) towards targets p_t (B,3), R_t (B,3,3).
    Rows stop updating once below tol. Returns q, position error (m), rotation error (rad)."""
    q = q0.clone()
    n = q.shape[1]
    eye = torch.eye(n, dtype=q.dtype, device=q.device)
    active = torch.arange(len(q), device=q.device)
    for _ in range(iters):
        if len(active) == 0:
            break
        qa = q[active]
        pe, Re, J = chain.fk_jac(qa)
        e = torch.cat([p_t[active] - pe, rot_err(R_t[active], Re)], dim=1)
        done = (e[:, :3].norm(dim=1) < tol_m) & (e[:, 3:].norm(dim=1) < tol_rad)
        if done.any():
            keep = ~done
            active, qa, e, J = active[keep], qa[keep], e[keep], J[keep]
            if len(active) == 0:
                break
        Jt = J.transpose(1, 2)
        lam = (e * e).sum(1) + 1e-6
        H = Jt @ J + lam[:, None, None] * eye
        g = (Jt @ e[:, :, None])[:, :, 0]
        if q_rest is not None and rest_weight > 0:
            H = H + rest_weight * eye
            g = g + rest_weight * (q_rest[active] - qa)
        step = torch.linalg.solve(H, g[:, :, None])[:, :, 0]
        q[active] = torch.maximum(torch.minimum(qa + step, chain.hi), chain.lo)
    pe, Re = chain.fk(q)
    return q, (p_t - pe).norm(dim=1), rot_err(R_t, Re).norm(dim=1)


def _best(q_all, pe_all, re_all, K, N, tol_mm, tol_deg):
    """Per target, the candidate with the lowest error relative to the tolerance."""
    score = (pe_all * 1000.0 / tol_mm) + (re_all / DEG / tol_deg)
    k = score.view(K, N).argmin(0)
    ar = torch.arange(N, device=q_all.device)
    pick = lambda a: a.view(K, N, *a.shape[1:])[k, ar]
    return pick(q_all), pick(pe_all), pick(re_all), k


def solve(chain, pos, quat, starts, ref=None, iters=30, restarts=0, restart_iters=None,
          tol_mm=1.0, tol_deg=1.0, rng_seed=0, chunk=20000, rest_weight=0.0):
    """
    pos (N,3), quat (N,4 xyzw): target poses.
    starts: list of (N,dof) starting joint vectors (e.g. [network output, seed]); each is
            wrapped into the joint limits (nearest `ref`, default the start itself).
    restarts: extra random starting points (uniform within limits) - used only for the
              targets that are still outside tol_mm / tol_deg after the given starts.
    Returns numpy dict: q (N,dof), pos_err_mm, orn_err_deg, winner (index of the start that
    won; len(starts)+k = random restart k), converged (within tolerance).
    """
    N = len(pos)
    outs = {k: [] for k in ("q", "pos_err_mm", "orn_err_deg", "winner")}
    gen = torch.Generator(device="cpu").manual_seed(int(rng_seed))
    for a in range(0, N, chunk):
        b = min(N, a + chunk)
        p_t = _t(pos[a:b], chain)
        R_t = quat_to_matrix(_t(quat[a:b], chain))
        r = _t(ref[a:b], chain) if ref is not None else None
        cands = [wrap_to_limits(_t(s[a:b], chain), chain.lo, chain.hi, r) for s in starts]
        K, n = len(cands), b - a
        q_rest = r.repeat(K, 1) if (r is not None and rest_weight > 0) else None
        q_all, pe, re = lm(chain, torch.cat(cands), p_t.repeat(K, 1), R_t.repeat(K, 1, 1), iters,
                           q_rest=q_rest, rest_weight=rest_weight)
        q, pe, re, win = _best(q_all, pe, re, K, n, tol_mm, tol_deg)
        if restarts > 0:
            bad = ((pe * 1000 > tol_mm) | (re / DEG > tol_deg)).nonzero()[:, 0]
            if len(bad):
                m = len(bad)
                u = torch.rand(restarts * m, chain.dof, generator=gen, dtype=torch.float64).to(chain.device, chain.dtype)
                q0 = chain.lo + u * (chain.hi - chain.lo)
                qr, per, rer = lm(chain, q0, p_t[bad].repeat(restarts, 1), R_t[bad].repeat(restarts, 1, 1),
                                  restart_iters or iters)
                qr, per, rer, kr = _best(qr, per, rer, restarts, m, tol_mm, tol_deg)
                better = (per * 1000 / tol_mm + rer / DEG / tol_deg) < (pe[bad] * 1000 / tol_mm + re[bad] / DEG / tol_deg)
                idx = bad[better]
                q[idx], pe[idx], re[idx], win[idx] = qr[better], per[better], rer[better], kr[better] + K
        outs["q"].append(q.cpu().numpy())
        outs["pos_err_mm"].append((pe * 1000).cpu().numpy())
        outs["orn_err_deg"].append((re / DEG).cpu().numpy())
        outs["winner"].append(win.cpu().numpy())
    res = {k: np.concatenate(v) if v else np.zeros(0) for k, v in outs.items()}
    res["converged"] = (res["pos_err_mm"] <= tol_mm) & (res["orn_err_deg"] <= tol_deg)
    return res


def random_candidates(chain, pos, quat, n_random=32, iters=50, rng_seed=1):
    """LM from n_random random starts per target, all results kept (no selection):
    q (n_random, N, dof), pos_err_mm, orn_err_deg (n_random, N). Used to look for
    another solution (IK branch) when the best one fails a check the solver cannot
    see, e.g. self-collision."""
    N = len(pos)
    gen = torch.Generator(device="cpu").manual_seed(int(rng_seed))
    p_t = _t(pos, chain)
    R_t = quat_to_matrix(_t(quat, chain))
    u = torch.rand(n_random * N, chain.dof, generator=gen, dtype=torch.float64).to(chain.device, chain.dtype)
    q, pe, re = lm(chain, chain.lo + u * (chain.hi - chain.lo), p_t.repeat(n_random, 1),
                   R_t.repeat(n_random, 1, 1), iters)
    return {"q": q.view(n_random, N, -1).cpu().numpy(),
            "pos_err_mm": (pe * 1000).view(n_random, N).cpu().numpy(),
            "orn_err_deg": (re / DEG).view(n_random, N).cpu().numpy()}
