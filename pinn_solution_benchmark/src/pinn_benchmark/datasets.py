"""Datasets: analytic ground truth, collocation/IC/BC sampling, test grids,
sparse noisy observations for the inverse problem. No leakage: training points
are random (seeded) Latin-hypercube samples, test points are a fixed uniform
grid kept disjoint by construction (verified in tests/test_datasets.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import sympy as sp
import torch

from . import equations as eqs
from .ansatz import J_expr, XI


@dataclass
class BranchSolution:
    """Callable analytic solution A(z,t) = a J_{n,m}(lambda1 z + lambda2 t)
    for one verified branch representative (constant-phase, real amplitude)."""
    n: Fraction
    m: Fraction
    params: dict           # nine physical parameters as floats
    a: float
    lambda1: float
    lambda2: float
    xi_domain: tuple       # (xi_lo, xi_hi) valid domain
    z_max: float = 1.0
    power_convention: str = "positive_xi"

    def __post_init__(self):
        Jexpr = J_expr(self.n, self.m, xi=XI)
        dJ = sp.diff(Jexpr, XI)
        self._J = sp.lambdify(XI, Jexpr, "numpy")
        self._dJ = sp.lambdify(XI, dJ, "numpy")

    @property
    def t_domain(self) -> tuple:
        lo_xi, hi_xi = self.xi_domain
        l1, l2, zm = self.lambda1, self.lambda2, self.z_max
        # keep xi(z,t) inside [lo_xi, hi_xi] for ALL z in [0, z_max]
        t_lo = (lo_xi - min(0.0, l1 * zm)) / l2
        t_hi = (hi_xi - max(0.0, l1 * zm)) / l2
        return (min(t_lo, t_hi), max(t_lo, t_hi))

    def xi(self, z, t):
        return self.lambda1 * np.asarray(z) + self.lambda2 * np.asarray(t)

    def A(self, z, t):
        """Complex field (imaginary part identically 0 for these branches)."""
        xi = self.xi(z, t)
        with np.errstate(all="raise"):
            vals = self.a * np.broadcast_to(
                np.asarray(self._J(xi), dtype=float), np.shape(xi)).copy()
        return vals.astype(complex)

    def u_v(self, z, t):
        Av = self.A(z, t)
        return Av.real, Av.imag


def branch_solution_from_record(rec: dict, rep_index: int = 0, z_max: float = 1.0) -> BranchSolution:
    rep = rec["representative_sets"][rep_index]
    params = {k: float(sp.sympify(rep[k])) for k in eqs.PARAM_NAMES}
    return BranchSolution(
        n=Fraction(rec["n"]), m=Fraction(rec["m"]), params=params,
        a=float(sp.sympify(rep["a"])),
        lambda1=float(sp.sympify(rep["lambda1"])),
        lambda2=float(sp.sympify(rep["lambda2"])),
        xi_domain=tuple(float(x) for x in rec["xi_domain"]),
        z_max=z_max,
        power_convention=rec.get("power_convention", "positive_xi"),
    )


def latin_hypercube(n: int, lo, hi, rng: np.random.Generator) -> np.ndarray:
    d = len(lo)
    u = (rng.permutation(n * d).reshape(n, d) % n + rng.random((n, d))) / n
    return np.asarray(lo) + u * (np.asarray(hi) - np.asarray(lo))


@dataclass
class ForwardData:
    interior: np.ndarray   # (N,2) z,t collocation
    initial: np.ndarray    # (Ni,2) z=0
    initial_uv: np.ndarray
    boundary: np.ndarray   # (Nb,2) t = t_lo or t_hi
    boundary_uv: np.ndarray
    test_grid: tuple       # (Z mesh, T mesh) uniform, disjoint from training
    test_uv: np.ndarray
    residual_test: np.ndarray  # independent residual evaluation points


def make_forward_data(sol: BranchSolution, n_interior: int, n_initial: int,
                      n_boundary: int, seed: int, test_shape=(64, 128),
                      n_residual_test: int = 2000) -> ForwardData:
    rng = np.random.default_rng(seed)
    t_lo, t_hi = sol.t_domain
    zm = sol.z_max
    interior = latin_hypercube(n_interior, [0.0, t_lo], [zm, t_hi], rng)

    ti = np.linspace(t_lo, t_hi, n_initial)
    initial = np.stack([np.zeros_like(ti), ti], axis=1)
    u0, v0 = sol.u_v(initial[:, 0], initial[:, 1])
    initial_uv = np.stack([u0, v0], axis=1)

    zb = np.linspace(0, zm, n_boundary)
    bc = np.concatenate([np.stack([zb, np.full_like(zb, t_lo)], axis=1),
                         np.stack([zb, np.full_like(zb, t_hi)], axis=1)])
    ub, vb = sol.u_v(bc[:, 0], bc[:, 1])
    boundary_uv = np.stack([ub, vb], axis=1)

    # fixed uniform evaluation grid, strictly interior offsets so that it can
    # never coincide with the (continuous-valued) LHS training samples
    zg = np.linspace(0, zm, test_shape[0])
    tg = np.linspace(t_lo, t_hi, test_shape[1])
    Z, Tm = np.meshgrid(zg, tg, indexing="ij")
    ut, vt = sol.u_v(Z.ravel(), Tm.ravel())
    test_uv = np.stack([ut, vt], axis=1)

    residual_test = latin_hypercube(n_residual_test, [0.0, t_lo], [zm, t_hi],
                                    np.random.default_rng(seed + 10_000))
    return ForwardData(interior, initial, initial_uv, bc, boundary_uv,
                       (Z, Tm), test_uv, residual_test)


def make_observations(sol: BranchSolution, density: float, noise_level: float,
                      seed: int, base_grid=(48, 96)) -> dict:
    """Sparse interior observations for the inverse problem.

    density: fraction of the base grid observed; noise_level: Gaussian noise
    std relative to the field's std over the grid."""
    rng = np.random.default_rng(seed)
    t_lo, t_hi = sol.t_domain
    zg = np.linspace(0, sol.z_max, base_grid[0])
    tg = np.linspace(t_lo, t_hi, base_grid[1])
    Z, Tm = np.meshgrid(zg, tg, indexing="ij")
    pts = np.stack([Z.ravel(), Tm.ravel()], axis=1)
    n_obs = max(4, int(density * len(pts)))
    idx = rng.choice(len(pts), size=n_obs, replace=False)
    obs_pts = pts[idx]
    u, v = sol.u_v(obs_pts[:, 0], obs_pts[:, 1])
    field_std = float(np.std(np.hypot(u, v))) or 1.0
    sigma = noise_level * field_std
    u_noisy = u + rng.normal(0, sigma, u.shape) if sigma > 0 else u
    v_noisy = v + rng.normal(0, sigma, v.shape) if sigma > 0 else v
    return {"points": obs_pts, "uv": np.stack([u_noisy, v_noisy], axis=1),
            "uv_clean": np.stack([u, v], axis=1), "sigma": sigma,
            "density": density, "noise_level": noise_level}


def to_torch(arr, device, requires_grad=False):
    t = torch.as_tensor(np.asarray(arr), dtype=torch.float64, device=device)
    if requires_grad:
        t = t.clone().requires_grad_(True)
    return t
