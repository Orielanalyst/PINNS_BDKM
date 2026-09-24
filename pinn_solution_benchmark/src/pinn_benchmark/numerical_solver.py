"""Independent conventional solver for propagation in z (Phase 3).

Two schemes, chosen per branch profile:

* ``fourier_etdrk4``: Fourier pseudospectral in t + ETDRK4 exponential
  integrator (Kassam & Trefethen 2005). Valid only for periodic-compatible
  profiles (decaying/localized or constant). The exact linear propagator
  absorbs the stiff high-order dispersion terms.

* ``fd_mol``: 6th-order central finite differences with three ghost nodes per
  side filled from the EXACT analytic solution (Dirichlet-type boundary strip;
  legitimate for a validation solver with known boundary data), integrated
  with scipy.solve_ivp (DOP853 by default, Radau optionally). Used for kink
  profiles, which are NOT periodic-compatible.

The solver is an independent confirmation of the verified analytical
solutions; it is never used to construct them.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.integrate import solve_ivp

from .datasets import BranchSolution
from .metrics import l2re_complex

# 6th-order central FD stencils
_W2 = np.array([1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90])
_W3 = np.array([1 / 8, -1, 13 / 8, 0, -13 / 8, 1, -1 / 8])  # antisym, O(h^4)
_W4 = np.array([-1 / 6, 2, -13 / 2, 28 / 3, -13 / 2, 2, -1 / 6])


def _nonlinear_rhs(A, p):
    I2 = np.abs(A) ** 2
    Q = I2 / (1 + p["Gamma"] * I2)
    ell = (p["g0"] - p["alpha"]) / 2
    return -(1j * (p["gamma"] + 1j * p["alpha2"] / 2) * Q * A + ell * A)


def _linear_symbol(k, p):
    c = p["g0"] * p["T2"] ** 2
    ell = (p["g0"] - p["alpha"]) / 2
    return (1j / 2 * (p["beta2"] + 1j * c) * k**2 + 1j * p["beta3"] / 6 * k**3
            - 1j * p["beta4"] / 24 * k**4 - ell)


def fourier_etdrk4(sol: BranchSolution, nt: int, nz_steps: int, pad_factor: float = 1.0):
    """Propagate A(0,t) to z_max on a periodic t-grid. Returns (t, z_slices, A[z,t])."""
    p = sol.params
    t_lo, t_hi = sol.t_domain
    span = (t_hi - t_lo) * pad_factor
    mid = 0.5 * (t_lo + t_hi)
    tg = mid - span / 2 + span * np.arange(nt) / nt  # periodic grid, no endpoint dup
    A = sol.A(np.zeros_like(tg), tg)
    k = 2 * np.pi * np.fft.fftfreq(nt, d=span / nt)
    L = _linear_symbol(k, p)
    # ell enters L via the symbol; remove it from the nonlinear part
    ell = (p["g0"] - p["alpha"]) / 2
    h = sol.z_max / nz_steps
    E = np.exp(h * L)
    E2 = np.exp(h * L / 2)
    M = 32  # contour points for phi-functions (full circle: L is complex)
    r = np.exp(2j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = h * L[:, None] + r[None, :]
    Qc = h * np.mean((np.exp(LR / 2) - 1) / LR, axis=1)
    f1 = h * np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=1)
    f2 = h * np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1)
    f3 = h * np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=1)

    def N(Af):
        a_phys = np.fft.ifft(Af)
        nl = _nonlinear_rhs(a_phys, p) + ell * a_phys  # ell already in L
        return np.fft.fft(nl)

    v = np.fft.fft(A)
    z_slices = [0.0]
    fields = [A.copy()]
    for step in range(1, nz_steps + 1):
        Nv = N(v)
        a = E2 * v + Qc * Nv
        Na = N(a)
        b = E2 * v + Qc * Na
        Nb = N(b)
        cc = E2 * a + Qc * (2 * Nb - Nv)
        Nc = N(cc)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        if step % max(1, nz_steps // 8) == 0 or step == nz_steps:
            z_slices.append(step * h)
            fields.append(np.fft.ifft(v))
    return tg, np.array(z_slices), np.array(fields)


def fd_mol(sol: BranchSolution, nt: int, rtol: float, atol: float,
           method: str = "DOP853"):
    """Method-of-lines with 6th-order FD; ghost values from the exact solution."""
    p = sol.params
    c = p["g0"] * p["T2"] ** 2
    t_lo, t_hi = sol.t_domain
    tg = np.linspace(t_lo, t_hi, nt)
    h = tg[1] - tg[0]
    A0 = sol.A(np.zeros_like(tg), tg)
    ghost_t_lo = t_lo + h * np.arange(-3, 0)
    ghost_t_hi = t_hi + h * np.arange(1, 4)

    def rhs(z, y):
        A = y[:nt] + 1j * y[nt:]
        gl = sol.A(np.full(3, z), ghost_t_lo)
        gh = sol.A(np.full(3, z), ghost_t_hi)
        Ae = np.concatenate([gl, A, gh])
        d2 = np.convolve(Ae, _W2[::-1], mode="valid") / h**2
        d3 = np.convolve(Ae, _W3[::-1], mode="valid") / h**3
        d4 = np.convolve(Ae, _W4[::-1], mode="valid") / h**4
        dA = -(1j / 2 * (p["beta2"] + 1j * c) * d2 + p["beta3"] / 6 * d3
               + 1j * p["beta4"] / 24 * d4) + _nonlinear_rhs(A, p)
        return np.concatenate([dA.real, dA.imag])

    y0 = np.concatenate([A0.real, A0.imag])
    z_eval = np.linspace(0, sol.z_max, 9)
    t0 = time.perf_counter()
    res = solve_ivp(rhs, (0, sol.z_max), y0, t_eval=z_eval, method=method,
                    rtol=rtol, atol=atol)
    runtime = time.perf_counter() - t0
    if not res.success:
        raise RuntimeError(f"solve_ivp failed: {res.message}")
    fields = (res.y[:nt] + 1j * res.y[nt:]).T
    return tg, res.t, fields, runtime


def solve_branch(sol: BranchSolution, scheme: str, nt: int, rtol=1e-8, atol=1e-10,
                 nz_steps: int = 400, method: str = "DOP853", pad_factor: float = 2.0):
    t0 = time.perf_counter()
    if scheme == "fourier_etdrk4":
        tg, zs, fields = fourier_etdrk4(sol, nt, nz_steps, pad_factor=pad_factor)
        runtime = time.perf_counter() - t0
    elif scheme == "fd_mol":
        tg, zs, fields, runtime = fd_mol(sol, nt, rtol, atol, method)
    else:
        raise ValueError(f"unknown scheme {scheme}")
    # errors vs analytic at each z-slice
    errs, max_errs = [], []
    for zv, F in zip(zs, fields):
        Aex = sol.A(np.full_like(tg, zv), tg)
        uv_p = np.stack([F.real, F.imag], axis=1)
        uv_t = np.stack([Aex.real, Aex.imag], axis=1)
        errs.append(l2re_complex(uv_p, uv_t))
        max_errs.append(float(np.abs(F - Aex).max()))
    return {"scheme": scheme, "nt": nt, "rtol": rtol, "nz_steps": nz_steps,
            "z_slices": [float(z) for z in zs],
            "l2re_vs_z": [float(e) for e in errs],
            "max_err_vs_z": max_errs,
            "final_l2re": float(errs[-1]), "final_max_err": max_errs[-1],
            "runtime_s": runtime}


def choose_scheme(rec: dict) -> str:
    """Kinks (non-decaying gradient at boundary) get FD; localized/constant get
    Fourier. Decision from catalog properties, not from what 'works better'."""
    props = rec.get("properties", {})
    if props.get("singular_at_xi0") or props.get("fractional_power"):
        return "fd_mol"  # padded periodic window would leave the valid domain
    if props.get("localized") or (float(props.get("feat_gradient_max", 1)) < 1e-12):
        return "fourier_etdrk4"
    return "fd_mol"


def convergence_study(sol: BranchSolution, scheme: str, resolutions, tols_or_steps):
    """>=3 spatial resolutions x >=2 tolerances/step counts."""
    rows = []
    for nt in resolutions:
        for tv in tols_or_steps:
            if scheme == "fourier_etdrk4":
                rows.append(solve_branch(sol, scheme, nt, nz_steps=int(tv)))
            else:
                rows.append(solve_branch(sol, scheme, nt, rtol=float(tv), atol=float(tv) * 1e-2))
    # observed order from the finest tolerance sequence
    finest = [r for r in rows if (r["rtol"] == min(t for t in tols_or_steps) if scheme == "fd_mol"
                                  else r["nz_steps"] == max(int(t) for t in tols_or_steps))]
    orders = []
    finest_sorted = sorted(finest, key=lambda r: r["nt"])
    for a, b in zip(finest_sorted, finest_sorted[1:]):
        if b["final_l2re"] > 0 and a["final_l2re"] > 0:
            ratio = np.log(a["final_l2re"] / b["final_l2re"]) / np.log(b["nt"] / a["nt"])
            orders.append(float(ratio))
    return {"runs": rows, "observed_orders": orders}
