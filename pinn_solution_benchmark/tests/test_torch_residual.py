"""Torch autodiff vs symbolic derivatives; 4th-derivative accuracy; torch
residual consistency; analytic-branch residual through torch."""
import numpy as np
import pytest
import sympy as sp
import torch

from pinn_benchmark import equations as eqs
from pinn_benchmark.equations import torch_derivative, torch_residual

torch.set_default_dtype(torch.float64)


def test_torch_derivatives_match_symbolic():
    """d^k/dt^k of a smooth function: autograd vs sympy, k = 1..4."""
    t_s = sp.Symbol("t", real=True)
    expr = sp.exp(-t_s**2 / 2) * sp.sin(2 * t_s)
    tv = np.linspace(-1.5, 1.5, 7)
    t = torch.tensor(tv[:, None], requires_grad=True)
    y = torch.exp(-t**2 / 2) * torch.sin(2 * t)
    for k in range(1, 5):
        dk_sym = sp.lambdify(t_s, sp.diff(expr, t_s, k), "numpy")(tv)
        dk_torch = torch_derivative(y, t, k).detach().numpy().ravel()
        assert np.allclose(dk_torch, dk_sym, rtol=1e-9, atol=1e-9), f"order {k}"


def test_fourth_derivative_accuracy_float64():
    """4th autograd derivative of tanh must agree with the closed form to
    near machine precision in float64."""
    t = torch.linspace(-2, 2, 11, dtype=torch.float64)[:, None].requires_grad_(True)
    y = torch.tanh(t)
    d4 = torch_derivative(y, t, 4).detach().numpy().ravel()
    tv = t.detach().numpy().ravel()
    x = sp.Symbol("x")
    d4_sym = sp.lambdify(x, sp.diff(sp.tanh(x), x, 4), "numpy")(tv)
    assert np.allclose(d4, d4_sym, rtol=1e-10, atol=1e-11)


def test_torch_residual_matches_sympy_residual():
    """torch_residual == lambdified symbolic split residual on a trial field."""
    z_s, t_s = eqs.z_s, eqs.t_s
    u_e = sp.exp(-(t_s**2)) * sp.cos(z_s)
    v_e = sp.Rational(1, 3) * sp.sin(t_s) * sp.exp(-t_s**2 / 2) * sp.sin(z_s)
    pvals = {"beta2": 0.3, "beta3": -0.2, "beta4": 0.15, "g0": 0.8, "T2": 0.5,
             "alpha": 0.4, "alpha2": 0.25, "gamma": 1.1, "Gamma": 0.6}
    Fu_s, Fv_s = eqs.split_residual_sympy(u_e, v_e,
                                          p={k: sp.Float(v) for k, v in pvals.items()})
    fu = sp.lambdify((z_s, t_s), Fu_s, "numpy")
    fv = sp.lambdify((z_s, t_s), Fv_s, "numpy")

    pts = np.array([[0.2, 0.4], [0.7, -0.9], [0.5, 1.3]])
    z = torch.tensor(pts[:, 0:1], requires_grad=True)
    t = torch.tensor(pts[:, 1:2], requires_grad=True)
    u = torch.exp(-(t**2)) * torch.cos(z)
    v = (1.0 / 3.0) * torch.sin(t) * torch.exp(-(t**2) / 2) * torch.sin(z)
    Fu_t, Fv_t = torch_residual(u, v, z, t, pvals)
    assert np.allclose(Fu_t.detach().numpy().ravel(), fu(pts[:, 0], pts[:, 1]), rtol=1e-9)
    assert np.allclose(Fv_t.detach().numpy().ravel(), fv(pts[:, 0], pts[:, 1]), rtol=1e-9)


def test_analytic_branch_residual_torch():
    """The verified kink solution has ~0 torch residual; a wrong parameter set
    does NOT (degenerate/point-fitted rejection guard)."""
    # verified closure: beta2=beta3=beta4=gamma=Gamma=0, ell=-c*l2^2, alpha2=-2c l2^2/a^2
    l2, cval, aval = 0.8, 0.5, 1.0
    good = {"beta2": 0.0, "beta3": 0.0, "beta4": 0.0, "g0": 1.0,
            "T2": float(np.sqrt(cval)), "alpha": 1.0 + 2 * cval * l2**2,
            "alpha2": -2 * cval * l2**2 / aval**2, "gamma": 0.0, "Gamma": 0.0}
    pts = np.random.default_rng(0).uniform(-1, 1, (50, 2))
    z = torch.tensor(pts[:, 0:1], requires_grad=True)
    t = torch.tensor(pts[:, 1:2], requires_grad=True)
    u = aval * torch.tanh(l2 * t) + 0.0 * z  # lambda1 = 0; keep z in graph
    v = torch.zeros_like(u) * z * t
    Fu, Fv = torch_residual(u, v, z, t, good)
    res = np.hypot(Fu.detach().numpy(), Fv.detach().numpy())
    assert res.max() < 1e-12

    bad = dict(good, alpha=0.9)  # violates the closure
    Fu_b, Fv_b = torch_residual(u, v, z, t, bad)
    res_b = np.hypot(Fu_b.detach().numpy(), Fv_b.detach().numpy())
    assert res_b.max() > 1e-3
