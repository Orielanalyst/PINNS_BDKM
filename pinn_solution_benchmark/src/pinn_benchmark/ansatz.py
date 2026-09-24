"""The iB-function (BDKm) ansatz J_{n,m}(xi) = sinh^m(xi)/cosh^n(xi).

Power conventions for fractional exponents of sinh(xi) (which is negative for
xi < 0) are handled explicitly and never silently:

* ``principal``  : complex principal power  sinh(xi)**m  (complex for xi<0)
* ``signed``     : sign(sinh)*|sinh|**m     (real, generally not C^1 at xi=0)
* ``positive_xi``: domain restricted to xi > 0 (sinh > 0, unambiguous)

For integer m all three conventions coincide on their common domain.

The T-representation used by the closure certificate:
    T = tanh(xi) in (0,1) for xi > 0,   W = sech(xi) = sqrt(1-T^2),
    J_{n,m} = T^m * W^(n-m),
    J'/J = m/T - n*T          (log-derivative, rational in T),
    d/dxi = (1-T^2) d/dT.
All k-th log-derivatives J^{(k)}/J are rational functions of T.
"""
from __future__ import annotations

from fractions import Fraction

import mpmath as mp
import numpy as np
import sympy as sp

XI = sp.Symbol("xi", real=True)
T = sp.Symbol("T", positive=True)  # tanh(xi) on xi>0

GRID = [Fraction(-3, 2), Fraction(-1), Fraction(-1, 2), Fraction(0),
        Fraction(1, 2), Fraction(1), Fraction(3, 2)]

ALL_PAIRS = [(n, m) for n in GRID for m in GRID]  # the paper's 49 ordered pairs


def frac_to_sympy(q: Fraction) -> sp.Rational:
    return sp.Rational(q.numerator, q.denominator)


def J_expr(n, m, xi=XI, convention: str = "positive_xi") -> sp.Expr:
    """Symbolic J_{n,m}(xi). For fractional m the convention decides the branch."""
    n = frac_to_sympy(Fraction(n))
    m = frac_to_sympy(Fraction(m))
    s, c = sp.sinh(xi), sp.cosh(xi)
    if m.is_integer or convention in ("principal", "positive_xi"):
        return s**m / c**n
    if convention == "signed":
        return sp.sign(s) * sp.Abs(s) ** m / c**n
    raise ValueError(f"unknown convention {convention!r}")


def is_fractional(n, m) -> bool:
    return Fraction(n).denominator != 1 or Fraction(m).denominator != 1


def log_derivatives_T(n, m, order: int = 4) -> list[sp.Expr]:
    """[phi_1..phi_order] with phi_k = J^{(k)}/J as rational functions of T.

    Derived from phi_1 = m/T - n*T and the recurrence
        J^{(k+1)}/J = d/dxi (J^{(k)}/J) + phi_1 * (J^{(k)}/J),
    with d/dxi = (1-T^2) d/dT.
    """
    n = frac_to_sympy(Fraction(n))
    m = frac_to_sympy(Fraction(m))
    phi1 = m / T - n * T
    phis = [phi1]
    psi = phi1  # psi_k = J^{(k)}/J
    for _ in range(order - 1):
        psi = sp.together((1 - T**2) * sp.diff(psi, T) + phi1 * psi)
        phis.append(sp.cancel(psi))
    return phis


def J_numeric(n, m, xi, convention: str = "positive_xi"):
    """Vectorized numpy evaluation of J_{n,m}. Returns complex for principal
    convention with fractional m and negative sinh; raises on domain violation
    for positive_xi."""
    n = float(Fraction(n))
    m = float(Fraction(m))
    xi = np.asarray(xi, dtype=float)
    s, c = np.sinh(xi), np.cosh(xi)
    frac = (m % 1) != 0
    if not frac:
        with np.errstate(divide="ignore", invalid="ignore"):
            return s**m / c**n
    if convention == "positive_xi":
        if np.any(xi <= 0):
            raise ValueError("positive_xi convention requires xi > 0 everywhere")
        return s**m / c**n
    if convention == "signed":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.sign(s) * np.abs(s) ** m / c**n
    if convention == "principal":
        return np.power(s.astype(complex), m) / np.power(c.astype(complex), n)
    raise ValueError(f"unknown convention {convention!r}")


def J_mp(n, m, xi, convention: str = "positive_xi"):
    """High-precision mpmath evaluation of J_{n,m} at scalar xi."""
    n = mp.mpf(Fraction(n).numerator) / Fraction(n).denominator
    m = mp.mpf(Fraction(m).numerator) / Fraction(m).denominator
    s, c = mp.sinh(xi), mp.cosh(xi)
    if convention == "positive_xi" and xi <= 0:
        raise ValueError("xi must be > 0 for positive_xi convention")
    if convention == "signed":
        return mp.sign(s) * mp.power(abs(s), m) / mp.power(c, n)
    return mp.power(s, m) / mp.power(c, n)  # principal (real if s>0 or m integer)


def singularities(n, m) -> dict:
    """Where and why J_{n,m} or its first four derivatives fail on the real line.

    cosh never vanishes on R, so the only trouble spots are xi=0 (negative or
    fractional powers of sinh) and xi=+-inf (growth). Returns a dict of flags.
    """
    n_f, m_f = Fraction(n), Fraction(m)
    out = {
        "pole_at_0": m_f < 0,
        # J ~ xi^m near 0; k-th derivative ~ xi^(m-k): C^4 at 0 needs m=0 or m>=...
        # integer m>=0 is smooth; fractional m>0 has a derivative blowing up at 0.
        "smooth_C4_at_0": (m_f.denominator == 1 and m_f >= 0),
        "fractional_branch_at_0": (m_f.denominator != 1),
        # growth at infinity: J ~ 2^(n-m) e^((m-n)|xi|) up to sign
        "bounded_at_inf": (m_f - n_f) <= 0,
        "decays_at_inf": (m_f - n_f) < 0,
    }
    out["bounded_on_R"] = bool(out["bounded_at_inf"] and not out["pole_at_0"]
                               and m_f.denominator == 1)
    return out
