"""Phase 7: identifiability analysis + staged inverse parameter discovery.

Identifiability: the PDE depends on (g0, T2, alpha) only through
c = g0*T2^2 and ell = (g0-alpha)/2, so those three are NEVER separately
identifiable from a single field. The analysis below confirms this numerically
via the SVD of the residual sensitivity Jacobian, and recovery is therefore
performed in the reduced parameter vector

    theta = (beta2, beta3, beta4, c, ell, gamma, alpha2, Gamma).

Positivity-constrained parameters (c, Gamma) use softplus parameterization;
all parameters are scaled to O(1).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .datasets import BranchSolution, make_forward_data, make_observations, to_torch
from .losses import data_fit_loss
from .networks import PINN
from .trainers import build_model
from .utils import log, set_seed

REDUCED_NAMES = ["beta2", "beta3", "beta4", "c", "ell", "gamma", "alpha2", "Gamma"]
POSITIVE = {"c", "Gamma"}


def reduced_from_physical(p: dict) -> dict:
    out = {k: float(p[k]) for k in ("beta2", "beta3", "beta4", "gamma", "alpha2", "Gamma")}
    out["c"] = float(p["g0"] * p["T2"] ** 2)
    out["ell"] = float((p["g0"] - p["alpha"]) / 2)
    return out


def physical_from_reduced(th: dict) -> dict:
    """Map back for torch_residual (choose g0=1 when c>0; the choice is
    irrelevant to the residual by construction)."""
    c, ell = th["c"], th["ell"]
    g0 = 1.0 if c > 0 else max(2 * ell, 0.0)
    T2 = float(np.sqrt(max(c, 0.0) / g0)) if g0 > 0 else 0.0
    alpha = g0 - 2 * ell
    return {"beta2": th["beta2"], "beta3": th["beta3"], "beta4": th["beta4"],
            "g0": g0, "T2": T2, "alpha": alpha, "gamma": th["gamma"],
            "alpha2": th["alpha2"], "Gamma": th["Gamma"]}


def torch_residual_reduced(u, v, z, t, th: dict):
    """torch residual written directly in the reduced parameters (autograd-safe
    even when th values are trainable tensors)."""
    from .equations import torch_derivative

    b2, b3, b4 = th["beta2"], th["beta3"], th["beta4"]
    c, ell = th["c"], th["ell"]
    a2, gam, Gam = th["alpha2"], th["gamma"], th["Gamma"]
    u_t = torch_derivative(u, t, 1)
    v_t = torch_derivative(v, t, 1)
    u_tt = torch_derivative(u_t, t, 1)
    v_tt = torch_derivative(v_t, t, 1)
    u_ttt = torch_derivative(u_tt, t, 1)
    v_ttt = torch_derivative(v_tt, t, 1)
    u_tttt = torch_derivative(u_ttt, t, 1)
    v_tttt = torch_derivative(v_ttt, t, 1)
    u_z = torch_derivative(u, z, 1)
    v_z = torch_derivative(v, z, 1)
    I2 = u**2 + v**2
    Q = I2 / (1 + Gam * I2)
    Fu = (u_z - c / 2 * u_tt - b2 / 2 * v_tt + b3 / 6 * u_ttt - b4 / 24 * v_tttt
          - a2 / 2 * Q * u - gam * Q * v + ell * u)
    Fv = (v_z - c / 2 * v_tt + b2 / 2 * u_tt + b3 / 6 * v_ttt + b4 / 24 * u_tttt
          - a2 / 2 * Q * v + gam * Q * u + ell * v)
    return Fu, Fv


# ---------------------------------------------------------------------------
# Identifiability
# ---------------------------------------------------------------------------

def sensitivity_analysis(sol: BranchSolution, n_points: int = 400, seed: int = 0,
                         include_g0T2alpha: bool = True) -> dict:
    """SVD of d(residual)/d(theta) evaluated ON the exact solution.

    Also builds the Jacobian in the raw (g0, T2, alpha) coordinates to
    demonstrate the rank deficiency (they enter only via c and ell)."""
    rng = np.random.default_rng(seed)
    t_lo, t_hi = sol.t_domain
    pts = np.stack([rng.uniform(0, sol.z_max, n_points),
                    rng.uniform(t_lo, t_hi, n_points)], axis=1)
    z = to_torch(pts[:, 0:1], "cpu", requires_grad=True)
    t = to_torch(pts[:, 1:2], "cpu", requires_grad=True)
    uv = np.stack(sol.u_v(pts[:, 0], pts[:, 1]), axis=1)

    # exact field evaluated torch-natively (autograd-capable closed form)
    xi = sol.lambda1 * z + sol.lambda2 * t
    u = sol.a * _torch_J(sol.n, sol.m, xi)
    v = torch.zeros_like(z)
    th_true = reduced_from_physical(sol.params)

    J_red = np.stack([_per_point_sens(u, v, z, t, th_true, name)
                      for name in REDUCED_NAMES], axis=1)
    s_red = np.linalg.svd(J_red, compute_uv=False)
    rank_tolerances = (1e-6, 1e-8, 1e-10)
    out = {"reduced_names": REDUCED_NAMES,
           "reduced_singular_values": s_red.tolist(),
           "reduced_condition_number": float(s_red[0] / max(s_red[-1], 1e-300)),
           "reduced_rank_1e-8": int((s_red > 1e-8 * s_red[0]).sum()),
           "reduced_rank_by_tolerance": {
               f"{tol:.0e}": int((s_red > tol * s_red[0]).sum())
               for tol in rank_tolerances}}

    if include_g0T2alpha:
        raw = {k: float(v_) for k, v_ in sol.params.items()}
        J_raw = np.stack([_per_point_sens(u, v, z, t, raw, name, raw_mode=True)
                          for name in ("g0", "T2", "alpha")], axis=1)
        s_raw = np.linalg.svd(J_raw, compute_uv=False)
        _, _, Vt = np.linalg.svd(J_raw)
        out.update({
            "g0T2alpha_singular_values": s_raw.tolist(),
            "g0T2alpha_rank_1e-8": int((s_raw > 1e-8 * max(s_raw[0], 1e-300)).sum()),
            "g0T2alpha_rank_by_tolerance": {
                f"{tol:.0e}": int((s_raw > tol * max(s_raw[0], 1e-300)).sum())
                for tol in rank_tolerances},
            "g0T2alpha_null_direction": Vt[-1].tolist(),
            "statement": "rank < 3 confirms g0, T2, alpha are not separately "
                         "identifiable; only c = g0*T2^2 and ell = (g0-alpha)/2 are.",
        })
    return out


def _per_point_sens(u, v, z, t, params: dict, name: str, raw_mode: bool = False):
    """Per-point dF/d(param) by central differences at the true parameters."""
    def F_at(pdict):
        th = {k: torch.tensor(float(vv), dtype=torch.float64) for k, vv in pdict.items()}
        if raw_mode:
            Fu, Fv = _residual_raw(u, v, z, t, th)
        else:
            Fu, Fv = torch_residual_reduced(u, v, z, t, th)
        return torch.cat([Fu, Fv]).squeeze(-1).detach().numpy()

    eps = 1e-6 * max(1.0, abs(float(params[name])))
    p_hi = dict(params); p_hi[name] = float(params[name]) + eps
    p_lo = dict(params); p_lo[name] = float(params[name]) - eps
    return (F_at(p_hi) - F_at(p_lo)) / (2 * eps)


def _residual_raw(u, v, z, t, th):
    from .equations import torch_residual
    return torch_residual(u, v, z, t, th)


def _torch_J(n, m, xi):
    """Torch-native J_{n,m} for integer-branch profiles (verified catalog)."""
    from fractions import Fraction
    n, m = Fraction(n), Fraction(m)
    assert n.denominator == 1 and m.denominator == 1, "torch J needs integer indices"
    s, cch = torch.sinh(xi), torch.cosh(xi)
    return s ** int(m) / cch ** int(n)


# ---------------------------------------------------------------------------
# Staged inverse recovery
# ---------------------------------------------------------------------------

STAGES = {
    "dispersion": ["beta2", "beta3", "beta4"],
    "nonlinear": ["gamma", "Gamma", "alpha2"],
    "gainloss": ["c", "ell"],
    "joint": ["c", "ell", "alpha2", "Gamma"],
}


def run_inverse(sol: BranchSolution, stage: str, density: float, noise: float,
                seed: int, cfg: dict, device: str = "cpu") -> dict:
    set_seed(seed)
    tr = cfg["training"]
    free_names = STAGES[stage]
    th_true = reduced_from_physical(sol.params)

    data = make_forward_data(sol, tr["n_collocation"], tr["n_initial"],
                             tr["n_boundary"], seed,
                             test_shape=tuple(cfg.get("metrics", {}).get("test_grid", [48, 96])))
    obs = make_observations(sol, density, noise, seed)
    model = build_model(cfg, sol, device)

    # parameter scaling to O(1); softplus for positive parameters
    scale = {k: max(abs(th_true[k]), 0.25) for k in REDUCED_NAMES}
    raw_params = {}
    init_off = cfg.get("init_offset", 0.5)
    rng = np.random.default_rng(seed + 777)
    for k in free_names:
        target = th_true[k] / scale[k]
        start = target + init_off * rng.choice([-1.0, 1.0]) * max(abs(target), 0.4)
        if k in POSITIVE:
            start = max(start, 0.05)
            raw0 = float(np.log(np.expm1(start) + 1e-12))  # inverse softplus
        else:
            raw0 = float(start)
        raw_params[k] = torch.tensor(raw0, dtype=torch.float64, requires_grad=True)

    def current_theta():
        th = {k: torch.tensor(v, dtype=torch.float64) for k, v in th_true.items()}
        for k, raw in raw_params.items():
            val = torch.nn.functional.softplus(raw) if k in POSITIVE else raw
            th[k] = val * scale[k]
        return th

    zt_obs = to_torch(obs["points"], device)
    uv_obs = to_torch(obs["uv"], device)
    zt_col = to_torch(data.interior[: tr["n_collocation"]], device)

    opt = torch.optim.Adam([{"params": model.parameters(), "lr": tr.get("adam_lr", 2e-3)},
                            {"params": list(raw_params.values()),
                             "lr": tr.get("param_lr", 5e-3)}])
    t0 = time.perf_counter()
    history = []
    for step in range(tr["adam_steps"]):
        opt.zero_grad(set_to_none=True)
        th = current_theta()
        z = zt_col[:, 0:1].clone().requires_grad_(True)
        t = zt_col[:, 1:2].clone().requires_grad_(True)
        u, v = model(z, t)
        Fu, Fv = torch_residual_reduced(u, v, z, t, th)
        pde = (Fu**2 + Fv**2).mean()
        fit = data_fit_loss(model, zt_obs, uv_obs)
        loss = cfg.get("w_data", 100.0) * fit + pde
        if not torch.isfinite(loss):
            return {"state": "failed", "error": f"non-finite loss at step {step}",
                    "stage": stage, "density": density, "noise": noise, "seed": seed}
        loss.backward()
        opt.step()
        if step % 100 == 0:
            history.append({"step": step, "fit": float(fit.detach()),
                            "pde": float(pde.detach()),
                            **{k: float(current_theta()[k].detach()) for k in free_names}})
    elapsed = time.perf_counter() - t0

    th_fin = {k: float(v.detach()) for k, v in current_theta().items()}
    rows = []
    for k in free_names:
        true = th_true[k]
        rec = th_fin[k]
        rel = abs(rec - true) / max(abs(true), 1e-12) if abs(true) > 1e-12 else None
        rows.append({"param": k, "true": true, "recovered": rec,
                     "abs_error": abs(rec - true), "rel_error": rel})
    # field error at the end
    from .metrics import all_field_metrics
    Z, Tm = data.test_grid
    pts = np.stack([Z.ravel(), Tm.ravel()], axis=1)
    with torch.no_grad():
        u, v = model(to_torch(pts[:, 0:1], device), to_torch(pts[:, 1:2], device))
        pred = torch.cat([u, v], dim=1).cpu().numpy()
    fm = all_field_metrics(pred, data.test_uv)
    return {"state": "done", "stage": stage, "density": density, "noise": noise,
            "seed": seed, "params": rows, "l2re_field": fm["l2re_complex"],
            "n_obs": len(obs["points"]), "obs_sigma": obs["sigma"],
            "elapsed_s": elapsed, "history": history}
