"""Phase 8: parameter-conditioned PINN  A_theta(z, t; eta).

eta = [n, m, a, lambda1, lambda2, beta2, beta3, beta4, c, ell, gamma, alpha2,
Gamma] scaled to [-1,1] over the task pool (constant entries map to 0).
A plain (z,t) PINN cannot represent multiple solutions at the same
coordinates; the conditioning vector removes that obstruction.

New closure-consistent tasks (for interpolation/extrapolation) are generated
by re-evaluating the verified family's exact constraint expressions at new
values of its free parameters — never by perturbing parameters freely.
"""
from __future__ import annotations

import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as sp
import torch

from .datasets import BranchSolution, latin_hypercube, to_torch
from .inverse_problem import physical_from_reduced
from .networks import PINN
from .utils import log, set_seed

ETA_NAMES = ["n", "m", "a", "lambda1", "lambda2", "beta2", "beta3", "beta4",
             "c", "ell", "gamma", "alpha2", "Gamma"]

_CONSTRAINT_SYMS = {name: sp.Symbol(name, real=True) for name in
                    ["lambda1", "lambda2", "asq", "beta2", "beta3", "beta4",
                     "c", "ell", "gamma", "alpha2", "Gamma"]}


def make_rep_from_constraints(rec: dict, assignments: dict) -> dict:
    """Evaluate a verified family's exact constraints at new free-parameter
    values; returns the physical parameter set (raises if signs violate
    asq>0, c>=0, Gamma>=0)."""
    subs = {_CONSTRAINT_SYMS[k]: sp.sympify(v) for k, v in assignments.items()}
    full = {k: sp.sympify(v) for k, v in assignments.items()}
    for name, expr_s in rec["constraints"].items():
        val = sp.sympify(expr_s, locals=_CONSTRAINT_SYMS).subs(subs)
        if val.free_symbols:
            raise ValueError(f"constraint {name} still has free symbols {val.free_symbols}; "
                             f"assign all of {rec['free_parameters']}")
        full[name] = val
    asq = float(full["asq"])
    c, ell = float(full.get("c", 0)), float(full.get("ell", 0))
    Gam = float(full.get("Gamma", 0))
    if asq <= 0 or c < 0 or Gam < 0:
        raise ValueError(f"unphysical assignment: asq={asq}, c={c}, Gamma={Gam}")
    th = {k: float(full.get(k, 0.0)) for k in
          ("beta2", "beta3", "beta4", "gamma", "alpha2", "Gamma")}
    th.update({"c": c, "ell": ell})
    phys = physical_from_reduced(th)
    return {**phys, "a": float(np.sqrt(asq)),
            "lambda1": float(full.get("lambda1", 0.0)),
            "lambda2": float(full.get("lambda2", 1.0))}


def task_from_rep(rec: dict, rep: dict, z_max: float = 1.0) -> BranchSolution:
    return BranchSolution(
        n=Fraction(rec["n"]), m=Fraction(rec["m"]),
        params={k: float(rep[k]) for k in
                ("beta2", "beta3", "beta4", "g0", "T2", "alpha", "alpha2", "gamma", "Gamma")},
        a=float(rep["a"]), lambda1=float(rep["lambda1"]), lambda2=float(rep["lambda2"]),
        xi_domain=tuple(float(x) for x in rec["xi_domain"]), z_max=z_max)


def eta_vector(sol: BranchSolution) -> np.ndarray:
    p = sol.params
    return np.array([float(sol.n), float(sol.m), sol.a, sol.lambda1, sol.lambda2,
                     p["beta2"], p["beta3"], p["beta4"], p["g0"] * p["T2"] ** 2,
                     (p["g0"] - p["alpha"]) / 2, p["gamma"], p["alpha2"], p["Gamma"]])


class ConditionedTrainer:
    def __init__(self, tasks: list[BranchSolution], cfg: dict, device="cpu", seed=0):
        self.tasks = tasks
        self.cfg = cfg
        self.device = device
        self.seed = seed
        set_seed(seed)
        etas = np.stack([eta_vector(s) for s in tasks])
        # global input box over tasks (z, t, eta)
        t_los = [s.t_domain[0] for s in tasks]
        t_his = [s.t_domain[1] for s in tasks]
        eta_lo, eta_hi = etas.min(axis=0), etas.max(axis=0)
        net = cfg.get("network", {})
        self.model = PINN(hidden_layers=net.get("hidden_layers", 5),
                          width=net.get("width", 64),
                          activation=net.get("activation", "tanh"),
                          input_lo=(0.0, min(t_los), *eta_lo),
                          input_hi=(max(s.z_max for s in tasks), max(t_his), *eta_hi),
                          cond_dim=len(ETA_NAMES)).to(device)
        self._build_data()

    def _build_data(self):
        tr = self.cfg["training"]
        rng = np.random.default_rng(self.seed)
        self.data = []
        for s in self.tasks:
            t_lo, t_hi = s.t_domain
            interior = latin_hypercube(tr["n_collocation_per_task"], [0.0, t_lo],
                                       [s.z_max, t_hi], rng)
            ti = np.linspace(t_lo, t_hi, tr["n_initial_per_task"])
            ic = np.stack([np.zeros_like(ti), ti], axis=1)
            zb = np.linspace(0, s.z_max, tr["n_boundary_per_task"])
            bc = np.concatenate([np.stack([zb, np.full_like(zb, t_lo)], axis=1),
                                 np.stack([zb, np.full_like(zb, t_hi)], axis=1)])
            self.data.append({
                "sol": s,
                "eta": to_torch(eta_vector(s), self.device),
                "interior": to_torch(interior, self.device),
                "ic": to_torch(ic, self.device),
                "ic_uv": to_torch(np.stack(s.u_v(ic[:, 0], ic[:, 1]), axis=1), self.device),
                "bc": to_torch(bc, self.device),
                "bc_uv": to_torch(np.stack(s.u_v(bc[:, 0], bc[:, 1]), axis=1), self.device),
            })

    def train(self, steps: int, lr: float = 2e-3, data_fraction: float = 0.0,
              log_every: int = 200) -> list:
        """data_fraction > 0 adds sparse interior analytic data (fine-tuning
        on a held-out branch); 0 for the pure physics protocols."""
        from .inverse_problem import torch_residual_reduced, reduced_from_physical

        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        history = []
        extra_data = []
        if data_fraction > 0:
            for d in self.data:
                n_extra = max(2, int(data_fraction * len(d["interior"])))
                idx = np.random.default_rng(self.seed + 5).choice(
                    len(d["interior"]), n_extra, replace=False)
                pts = d["interior"][idx].cpu().numpy()
                uv = np.stack(d["sol"].u_v(pts[:, 0], pts[:, 1]), axis=1)
                extra_data.append((to_torch(pts, self.device), to_torch(uv, self.device)))
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            total = 0.0
            parts = {"pde": 0.0, "data": 0.0}
            for ti_, d in enumerate(self.data):
                th = reduced_from_physical(d["sol"].params)
                z = d["interior"][:, 0:1].clone().requires_grad_(True)
                t = d["interior"][:, 1:2].clone().requires_grad_(True)
                u, v = self.model(z, t, d["eta"])
                Fu, Fv = torch_residual_reduced(
                    u, v, z, t, {k: torch.tensor(v_, dtype=torch.float64) for k, v_ in th.items()})
                pde = (Fu**2 + Fv**2).mean()
                u_ic, v_ic = self.model(d["ic"][:, 0:1], d["ic"][:, 1:2], d["eta"])
                fit = ((u_ic - d["ic_uv"][:, 0:1]) ** 2 + (v_ic - d["ic_uv"][:, 1:2]) ** 2).mean()
                u_bc, v_bc = self.model(d["bc"][:, 0:1], d["bc"][:, 1:2], d["eta"])
                fit = fit + ((u_bc - d["bc_uv"][:, 0:1]) ** 2
                             + (v_bc - d["bc_uv"][:, 1:2]) ** 2).mean()
                task_loss = pde + 100.0 * fit
                if extra_data:
                    pts, uv = extra_data[ti_]
                    ue, ve = self.model(pts[:, 0:1], pts[:, 1:2], d["eta"])
                    task_loss = task_loss + 100.0 * ((ue - uv[:, 0:1]) ** 2
                                                     + (ve - uv[:, 1:2]) ** 2).mean()
                total = total + task_loss
                parts["pde"] += float(pde.detach())
            if not torch.isfinite(total):
                raise RuntimeError(f"non-finite conditioned loss at step {step}")
            total.backward()
            opt.step()
            if step % log_every == 0:
                history.append({"step": step, "total": float(total.detach()), **parts})
        return history

    @torch.no_grad()
    def evaluate(self, sol: BranchSolution, grid=(48, 96)) -> dict:
        from .metrics import all_field_metrics
        t_lo, t_hi = sol.t_domain
        zg = np.linspace(0, sol.z_max, grid[0])
        tg = np.linspace(t_lo, t_hi, grid[1])
        Z, Tm = np.meshgrid(zg, tg, indexing="ij")
        pts = np.stack([Z.ravel(), Tm.ravel()], axis=1)
        eta = to_torch(eta_vector(sol), self.device)
        u, v = self.model(to_torch(pts[:, 0:1], self.device),
                          to_torch(pts[:, 1:2], self.device), eta)
        pred = torch.cat([u, v], dim=1).cpu().numpy()
        true_uv = np.stack(sol.u_v(pts[:, 0], pts[:, 1]), axis=1)
        return all_field_metrics(pred, true_uv)
