"""The governing PDE, its complex residual, and the real/imaginary split.

Governing equation (paper Eq. (1)):

    A_z + (i/2)(beta2 + i*g0*T2^2) A_tt + (beta3/6) A_ttt + (i*beta4/24) A_tttt
        + i*(gamma + i*alpha2/2) * |A|^2 A / (1 + Gamma*|A|^2)
        + (g0 - alpha)/2 * A = 0.

Shorthand:  c = g0*T2^2  (parabolic gain-bandwidth coefficient),
            ell = (g0 - alpha)/2  (net linear gain/loss).

The real/imaginary split is DERIVED here (see `derive_split_sympy`), not
assumed; `tests/test_residual_split.py` proves F_complex = F_u + i F_v for
random smooth fields. The derived split (which corrects the paper's
Eqs. (37)-(38)) is, with Q = (u^2+v^2)/(1+Gamma*(u^2+v^2)):

  F_u = u_z - (c/2) u_tt - (beta2/2) v_tt + (beta3/6) u_ttt
        - (beta4/24) v_tttt - (alpha2/2) Q u - gamma Q v + ell*u
  F_v = v_z - (c/2) v_tt + (beta2/2) u_tt + (beta3/6) v_ttt
        + (beta4/24) u_tttt - (alpha2/2) Q v + gamma Q u + ell*v
"""
from __future__ import annotations

import sympy as sp

PARAM_NAMES = ["beta2", "beta3", "beta4", "g0", "T2", "alpha", "alpha2", "gamma", "Gamma"]

# Symbolic physical parameters (all real)
beta2, beta3, beta4, g0, T2, alpha, alpha2, gamma, Gamma = sp.symbols(
    "beta2 beta3 beta4 g0 T2 alpha alpha2 gamma Gamma", real=True
)
PARAMS = dict(zip(PARAM_NAMES, [beta2, beta3, beta4, g0, T2, alpha, alpha2, gamma, Gamma]))

z_s, t_s = sp.symbols("z t", real=True)


def derived_coefficients(p: dict) -> tuple:
    """(c, ell) from a symbol/value dict."""
    return p["g0"] * p["T2"] ** 2, (p["g0"] - p["alpha"]) / 2


def complex_residual_sympy(A: sp.Expr, z=z_s, t=t_s, p: dict | None = None) -> sp.Expr:
    """F[A] for a symbolic complex field A(z,t), directly from the paper's Eq. (1)."""
    p = p or PARAMS
    absA2 = A * sp.conjugate(A)
    return (
        sp.diff(A, z)
        + sp.I / 2 * (p["beta2"] + sp.I * p["g0"] * p["T2"] ** 2) * sp.diff(A, t, 2)
        + p["beta3"] / 6 * sp.diff(A, t, 3)
        + sp.I * p["beta4"] / 24 * sp.diff(A, t, 4)
        + sp.I * (p["gamma"] + sp.I * p["alpha2"] / 2) * absA2 * A / (1 + p["Gamma"] * absA2)
        + (p["g0"] - p["alpha"]) / 2 * A
    )


def split_residual_sympy(u: sp.Expr, v: sp.Expr, z=z_s, t=t_s, p: dict | None = None):
    """(F_u, F_v): the independently derived real/imaginary residuals."""
    p = p or PARAMS
    c, ell = derived_coefficients(p)
    Q = (u**2 + v**2) / (1 + p["Gamma"] * (u**2 + v**2))
    b2, b3, b4, a2, gam = p["beta2"], p["beta3"], p["beta4"], p["alpha2"], p["gamma"]
    Fu = (
        sp.diff(u, z) - c / 2 * sp.diff(u, t, 2) - b2 / 2 * sp.diff(v, t, 2)
        + b3 / 6 * sp.diff(u, t, 3) - b4 / 24 * sp.diff(v, t, 4)
        - a2 / 2 * Q * u - gam * Q * v + ell * u
    )
    Fv = (
        sp.diff(v, z) - c / 2 * sp.diff(v, t, 2) + b2 / 2 * sp.diff(u, t, 2)
        + b3 / 6 * sp.diff(v, t, 3) + b4 / 24 * sp.diff(u, t, 4)
        - a2 / 2 * Q * v + gam * Q * u + ell * v
    )
    return Fu, Fv


def derive_split_sympy():
    """Derive (F_u, F_v) from the complex PDE by substituting A = u + i v with
    u, v generic real functions, and return the expanded real and imaginary
    parts. Used by the audit and unit tests to prove the split above."""
    u = sp.Function("u", real=True)(z_s, t_s)
    v = sp.Function("v", real=True)(z_s, t_s)
    F = complex_residual_sympy(u + sp.I * v)
    F = sp.expand(F)
    Fu_derived = sp.simplify(sp.re(F))
    Fv_derived = sp.simplify(sp.im(F))
    return u, v, Fu_derived, Fv_derived


def paper_split_sympy(u, v, z=z_s, t=t_s, p: dict | None = None):
    """The paper's Eqs. (37)-(38) EXACTLY as printed (with its ill-formed
    Re/Im of a complex-coefficient nonlinear term interpreted literally as
    Re/Im of i*(u^2+v^2)(u+iv)/(1+Gamma(u^2+v^2)) scaled by (gamma + i alpha2/2)
    left outside). Kept only to document the discrepancy; NOT used anywhere."""
    p = p or PARAMS
    NL = sp.I * (u**2 + v**2) * (u + sp.I * v) / (1 + p["Gamma"] * (u**2 + v**2))
    coef = p["gamma"] + sp.I * p["alpha2"] / 2  # complex -> the printed form is ill-formed
    Fu = (
        sp.diff(u, z) - sp.Rational(1, 2) * (p["beta2"] * sp.diff(v, t, 2)
                                             + p["g0"] * p["T2"] ** 2 * sp.diff(u, t, 2))
        - p["beta3"] / 6 * sp.diff(v, t, 3) - p["beta4"] / 24 * sp.diff(u, t, 4)
        - coef * sp.re(NL) + (p["g0"] - p["alpha"]) / 2 * u
    )
    Fv = (
        sp.diff(v, z) + sp.Rational(1, 2) * (p["beta2"] * sp.diff(u, t, 2)
                                             - p["g0"] * p["T2"] ** 2 * sp.diff(v, t, 2))
        + p["beta3"] / 6 * sp.diff(u, t, 3) - p["beta4"] / 24 * sp.diff(v, t, 4)
        - coef * sp.im(NL) + (p["g0"] - p["alpha"]) / 2 * v
    )
    return Fu, Fv


# ----------------------------------------------------------------------------
# Torch implementation (used by the PINN and by the residual metrics)
# ----------------------------------------------------------------------------

def torch_derivative(y, x, order=1):
    """Repeated autograd derivative of y w.r.t. x (both requiring grad).

    allow_unused handles fields with no dependence on x (e.g. exact stationary
    solutions with lambda1 = 0): their derivative is identically zero."""
    import torch

    for _ in range(order):
        if not y.requires_grad:
            y = y + 0.0 * x  # constant field: attach to the graph so grad = 0
        (g,) = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                                   create_graph=True, allow_unused=True)
        if g is None:
            g = torch.zeros_like(x)
        if not g.requires_grad:
            # re-attach to the graph so higher derivatives stay well-defined
            g = g + 0.0 * x
        y = g
    return y


def torch_residual(u, v, z, t, params: dict):
    """(F_u, F_v) via automatic differentiation.

    u, v: tensors computed from z, t with requires_grad graphs intact.
    params: dict of python floats or 0-dim tensors for the nine coefficients.
    """
    import torch

    b2, b3, b4 = params["beta2"], params["beta3"], params["beta4"]
    g0v, T2v, av = params["g0"], params["T2"], params["alpha"]
    a2, gam, Gam = params["alpha2"], params["gamma"], params["Gamma"]
    c = g0v * T2v**2
    ell = (g0v - av) / 2

    u_z = torch_derivative(u, z, 1)
    v_z = torch_derivative(v, z, 1)
    u_t = torch_derivative(u, t, 1)
    v_t = torch_derivative(v, t, 1)
    u_tt = torch_derivative(u_t, t, 1)
    v_tt = torch_derivative(v_t, t, 1)
    u_ttt = torch_derivative(u_tt, t, 1)
    v_ttt = torch_derivative(v_tt, t, 1)
    u_tttt = torch_derivative(u_ttt, t, 1)
    v_tttt = torch_derivative(v_ttt, t, 1)

    I2 = u**2 + v**2
    Q = I2 / (1 + Gam * I2)

    Fu = (u_z - c / 2 * u_tt - b2 / 2 * v_tt + b3 / 6 * u_ttt - b4 / 24 * v_tttt
          - a2 / 2 * Q * u - gam * Q * v + ell * u)
    Fv = (v_z - c / 2 * v_tt + b2 / 2 * u_tt + b3 / 6 * v_ttt + b4 / 24 * u_tttt
          - a2 / 2 * Q * v + gam * Q * u + ell * v)
    return Fu, Fv
