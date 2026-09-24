"""Symbolic audit of the paper's derivation and exact closure certificate.

Three independent layers:

1. `derive_offset_coefficients` re-derives the paper's grouped coefficients
   C_1..C_9 (paper Eqs. 17-25) from scratch by iterating the differentiation
   rule dJ_{n,m}/dxi = m J_{n-1,m-1} - n J_{n+1,m+1}, keeping n, m symbolic.
   `paper_offset_coefficients` encodes the printed formulas; the audit compares
   the two symbolically.

2. `closure_certificate` builds, for a given index pair (n, m), the EXACT
   algebraic closure conditions in the T = tanh(xi) representation on xi > 0:

       J_{n,m} = T^m W^(n-m),  W = sech(xi) = sqrt(1-T^2),
       J^(k)/J = phi_k(T)  rational,
       P := J^2 = T^p W^w,  p = 2m in Z,  w = 2(n-m) in Z.

   Substituting A = a J(lambda1 z + lambda2 t) (real amplitude, constant
   phase — the paper's convention) into the PDE and dividing by A gives

       G(T) + Gamma a^2 P(T) G(T) + (i gamma - alpha2/2) a^2 P(T) = 0,

   where G collects the linear terms. Splitting into real/imaginary parts and
   into the {1, W} components (1 and W are linearly independent over the
   rational functions of T because W^2 = 1 - T^2 and W is not rational in T),
   then clearing only powers of T and (1 - T^2) — both nonzero for
   xi in (0, inf) — yields polynomial identities in T whose coefficients must
   all vanish. These coefficient equations are the complete closure system.

3. The optional extended-phase ansatz A = a J(xi) exp(i(kappa z + omega t))
   is supported by the same machinery (complex linear coefficients); it is a
   clearly separated experiment, never a silent replacement of the paper's
   constant-phase convention.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import sympy as sp

from .ansatz import T, frac_to_sympy, log_derivatives_T
from . import equations as eqs

# Wave/amplitude unknowns.  asq = a^2 > 0 (a real).
lam1, lam2 = sp.symbols("lambda1 lambda2", real=True)
asq = sp.Symbol("asq", positive=True)
# c = g0*T2^2 and ell = (g0-alpha)/2 are the only combinations in which
# g0, T2, alpha enter the closure system (an identifiability fact documented
# in the audit report); they are treated as independent real unknowns.
c_sym = sp.Symbol("c", real=True)
ell_sym = sp.Symbol("ell", real=True)
omega, kappa = sp.symbols("omega kappa", real=True)

b2, b3, b4 = eqs.beta2, eqs.beta3, eqs.beta4
gam, a2c, Gam = eqs.gamma, eqs.alpha2, eqs.Gamma

CLOSURE_UNKNOWNS = [lam1, lam2, asq, b2, b3, b4, c_sym, ell_sym, gam, a2c, Gam]
# extended-phase ansatz adds the carrier parameters
CLOSURE_UNKNOWNS_EXTENDED = CLOSURE_UNKNOWNS + [omega, kappa]


# ---------------------------------------------------------------------------
# Layer 1: offset-basis coefficients (paper Section 5)
# ---------------------------------------------------------------------------

def _diff_offsets(coeffs: dict, n, m) -> dict:
    """One application of d/dxi in the offset basis.

    `coeffs` maps offset r -> coefficient of J_{n+r, m+r}.  Using
    dJ_{p,q} = q J_{p-1,q-1} - p J_{p+1,q+1} with (p,q) = (n+r, m+r).
    """
    out: dict = {}
    for r, cf in coeffs.items():
        out[r - 1] = sp.expand(out.get(r - 1, 0) + cf * (m + r))
        out[r + 1] = sp.expand(out.get(r + 1, 0) - cf * (n + r))
    return {r: v for r, v in out.items() if v != 0}


def derive_offset_coefficients(n_sym=None, m_sym=None) -> dict:
    """Independently derived C_k (k=1..9 <-> offsets r=-4..4) for the
    traveling-wave ODE LHS divided by a, keeping n, m symbolic by default."""
    n = n_sym if n_sym is not None else sp.Symbol("n", real=True)
    m = m_sym if m_sym is not None else sp.Symbol("m", real=True)
    d1 = _diff_offsets({0: sp.Integer(1)}, n, m)
    d2 = _diff_offsets(d1, n, m)
    d3 = _diff_offsets(d2, n, m)
    d4 = _diff_offsets(d3, n, m)
    Xi = b2 + sp.I * c_sym  # paper's  Xi = beta2 + i g0 T2^2
    C: dict = {}
    for r in range(-4, 5):
        C[r] = sp.expand(
            lam1 * d1.get(r, 0)
            + sp.I / 2 * Xi * lam2**2 * d2.get(r, 0)
            + b3 / 6 * lam2**3 * d3.get(r, 0)
            + sp.I * b4 / 24 * lam2**4 * d4.get(r, 0)
            + (ell_sym if r == 0 else 0)
        )
    return {"n": n, "m": m, "C": C, "derivs": [d1, d2, d3, d4]}


def paper_offset_coefficients(n_sym=None, m_sym=None) -> dict:
    """The paper's printed Eqs. (17)-(25), transcribed verbatim.

    Paper notation: C1..C9 multiply J_{n+r} for r = -4..4 in that order;
    (g0-alpha)/2 is written here as ell_sym and g0*T2^2 as c_sym."""
    n = n_sym if n_sym is not None else sp.Symbol("n", real=True)
    m = m_sym if m_sym is not None else sp.Symbol("m", real=True)
    Xi = b2 + sp.I * c_sym
    A1 = m * (m - 1) * (m - 2)
    A2 = -3 * m**2 * n + 3 * m**2 - 2 * m
    A3 = 3 * m * n**2 + 3 * n**2 + 2 * n
    A4 = -n * (n + 1) * (n + 2)
    C = {
        -4: sp.I * b4 / 24 * lam2**4 * m * (m - 1) * (m - 2) * (m - 3),
        -3: b3 / 6 * lam2**3 * A1,
        -2: (sp.I / 2 * Xi * lam2**2 * m * (m - 1)
             + sp.I * b4 / 24 * lam2**4 * (A2 * (m - 1) - A1 * (n - 3))),
        -1: lam1 * m + b3 / 6 * lam2**3 * A2,
        0: (ell_sym - sp.I / 2 * Xi * lam2**2 * (m * (n - 1) + n * (m + 1))
            + sp.I * b4 / 24 * lam2**4 * (A3 * (m + 1) - A2 * (n - 1))),
        1: -lam1 * n + b3 / 6 * lam2**3 * A3,
        2: (sp.I / 2 * Xi * lam2**2 * n * (n + 1)
            + sp.I * b4 / 24 * lam2**4 * (A4 * (m + 3) - A3 * (n + 1))),
        3: -b3 / 6 * lam2**3 * n * (n + 1) * (n + 2),
        4: sp.I * b4 / 24 * lam2**4 * n * (n + 1) * (n + 2) * (n + 3),
    }
    return {"n": n, "m": m, "C": {r: sp.expand(v) for r, v in C.items()}}


def compare_offset_coefficients() -> dict:
    """Symbolic comparison of derived vs printed C_k with n, m symbolic.
    Returns {r: True/False} — True means identical for ALL (n, m)."""
    n, m = sp.symbols("n m", real=True)
    der = derive_offset_coefficients(n, m)["C"]
    pap = paper_offset_coefficients(n, m)["C"]
    return {r: bool(sp.simplify(sp.expand(der.get(r, 0) - pap.get(r, 0))) == 0)
            for r in range(-4, 5)}


# ---------------------------------------------------------------------------
# Layer 2: exact closure certificate in the T representation
# ---------------------------------------------------------------------------

@dataclass
class ClosureSystem:
    n: Fraction
    m: Fraction
    equations: list = field(default_factory=list)   # polynomials that must vanish
    blocks: dict = field(default_factory=dict)      # labeled subsystems for reporting
    p: int = 0                                      # P = T^p W^w
    w: int = 0
    extended_phase: bool = False
    cleared_denominators: str = ""                  # documentation of what was cleared
    notes: list = field(default_factory=list)


def _rational_to_poly_eqs(expr: sp.Expr, label: str, notes: list) -> list:
    """All-coefficients-vanish equations for a rational function of T.

    Clears the denominator, which for these expressions is always a product of
    powers of T and (1 - T**2) (and numeric constants) — both nonzero for
    T in (0,1), i.e. xi in (0, inf), so clearing is mathematically safe."""
    expr = sp.cancel(sp.together(sp.expand(expr)))
    num, den = sp.fraction(expr)
    den_factors = sp.factor_list(sp.expand(den))[1]
    for base, _ in den_factors:
        ok = base.equals(T) or sp.expand(base - (1 - T**2)) == 0 \
            or sp.expand(base - (T**2 - 1)) == 0 or sp.expand(base - (1 + T)) == 0 \
            or sp.expand(base - (1 - T)) == 0 or base.is_number
        if not ok:
            notes.append(f"{label}: unexpected denominator factor {base}")
    poly = sp.Poly(sp.expand(num), T)
    return [sp.expand(cf) for cf in poly.all_coeffs() if sp.expand(cf) != 0]


def linear_part_coefficients(extended_phase: bool = False) -> list:
    """Complex coefficients [c0..c4] of phi_0..phi_4 in G(T) = F_linear/(A).

    Constant-phase case (omega = kappa = 0):
        c0 = ell, c1 = lambda1, c2 = (i/2)(beta2 + i c) lambda2^2,
        c3 = (beta3/6) lambda2^3, c4 = (i beta4/24) lambda2^4.
    Extended phase adds the omega/kappa terms from d/dt -> lambda2 d/dxi + i omega.
    """
    if not extended_phase:
        return [ell_sym, lam1, sp.I / 2 * (b2 + sp.I * c_sym) * lam2**2,
                b3 / 6 * lam2**3, sp.I * b4 / 24 * lam2**4]
    Xi = b2 + sp.I * c_sym
    c0 = (sp.I * kappa - sp.I / 2 * Xi * omega**2 - sp.I * b3 / 6 * omega**3
          + sp.I * b4 / 24 * omega**4 + ell_sym)
    c1 = (lam1 - Xi * omega * lam2 - b3 / 2 * omega**2 * lam2
          + b4 / 6 * omega**3 * lam2)
    c2 = (sp.I / 2 * Xi * lam2**2 + sp.I * b3 / 2 * omega * lam2**2
          - sp.I * b4 / 4 * omega**2 * lam2**2)
    c3 = b3 / 6 * lam2**3 - b4 / 6 * omega * lam2**3
    c4 = sp.I * b4 / 24 * lam2**4
    return [c0, c1, c2, c3, c4]


_CERT_CACHE: dict = {}


def closure_certificate(n, m, extended_phase: bool = False) -> ClosureSystem:
    """Complete algebraic closure conditions for branch (n, m). Memoized."""
    n, m = Fraction(n), Fraction(m)
    key = (n, m, extended_phase)
    if key in _CERT_CACHE:
        return _CERT_CACHE[key]
    cs = ClosureSystem(n=n, m=m, extended_phase=extended_phase)
    p = 2 * m  # integer because m is a half-integer
    w = 2 * (n - m)
    assert p.denominator == 1 and w.denominator == 1
    cs.p, cs.w = int(p), int(w)

    phis = [sp.Integer(1)] + log_derivatives_T(n, m, order=4)  # phi_0..phi_4
    coeffs = linear_part_coefficients(extended_phase)
    G = sp.expand(sum(cf * ph for cf, ph in zip(coeffs, phis)))
    # Split G into real/imag by expanding I explicitly (all symbols are real).
    G_re, G_im = _complex_split(G)

    # P = T^p * W^w with W = sqrt(1-T^2); w even -> rational, w odd -> W * rational
    w_int = cs.w
    if w_int % 2 == 0:
        P_rat = T ** cs.p * (1 - T**2) ** (w_int // 2)
        # Residual/A:  X + P*(Gamma asq X + nl)  per component
        re_expr = G_re * (1 + Gam * asq * P_rat) - a2c / 2 * asq * P_rat
        im_expr = G_im * (1 + Gam * asq * P_rat) + gam * asq * P_rat
        cs.blocks["real"] = _rational_to_poly_eqs(re_expr, "real", cs.notes)
        cs.blocks["imag"] = _rational_to_poly_eqs(im_expr, "imag", cs.notes)
        cs.cleared_denominators = "powers of T and (1-T^2); T in (0,1) so both nonzero"
    else:
        # P = W * rho(T), rho = T^p (1-T^2)^((w-1)/2).  {1, W} independent over C(T):
        rho = T ** cs.p * (1 - T**2) ** ((w_int - 1) // 2)
        # real residual: G_re + W*rho*(Gamma asq G_re - alpha2 asq/2) = 0
        cs.blocks["real_1"] = _rational_to_poly_eqs(G_re, "real_1", cs.notes)
        cs.blocks["real_W"] = _rational_to_poly_eqs(
            rho * (Gam * asq * G_re - a2c / 2 * asq), "real_W", cs.notes)
        cs.blocks["imag_1"] = _rational_to_poly_eqs(G_im, "imag_1", cs.notes)
        cs.blocks["imag_W"] = _rational_to_poly_eqs(
            rho * (Gam * asq * G_im + gam * asq), "imag_W", cs.notes)
        cs.cleared_denominators = ("powers of T and (1-T^2) after splitting the "
                                   "{1, W} components (W = sech xi not rational in T)")
        cs.notes.append("half-odd-integer n-m: nonlinear terms lie in the W-component "
                        "alone, forcing alpha2*asq = gamma*asq = 0 unless they cancel "
                        "against Gamma-terms")
    seen = set()
    for block in cs.blocks.values():
        for e in block:
            key = sp.srepr(sp.factor(e))
            if key not in seen:
                seen.add(key)
                cs.equations.append(e)
    _CERT_CACHE[(n, m, extended_phase)] = cs
    return cs


def _complex_split(expr: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """(re, im) of a sympy expression whose only imaginary unit is explicit I
    and whose free symbols are all real."""
    expr = sp.expand(expr)
    re_part = expr.as_real_imag()
    return sp.expand(re_part[0]), sp.expand(re_part[1])


# ---------------------------------------------------------------------------
# Classification-issue analysis (paper Section 7)
# ---------------------------------------------------------------------------

def classification_analysis() -> dict:
    """Reproduce the paper's index-matching argument and state its scope.

    J_{n+r,m+r} = J_{3n+s,3m+s} requires n+r = 3n+s AND m+r = 3m+s, i.e.
    n = (r-s)/2 and m = (r-s)/2 simultaneously => n = m.  The paper's own
    Eq. (28) therefore supports only DIAGONAL pairs; the 49-pair Cartesian
    product of Section 7.3 includes 42 off-diagonal pairs with no derivation.

    Deeper issue: the argument implicitly treats distinct J_{p,q} as linearly
    independent basis elements. In truth J_{p,q} = T^q W^(p-q), and monomials
    with even powers of W collapse to polynomials in T, so distinct J-indices
    can be linearly DEPENDENT. Index matching is neither necessary nor
    sufficient for closure; only the T/W-basis certificate is conclusive.
    """
    r, s = sp.symbols("r s", integer=True)
    n, m = sp.symbols("n m", real=True)
    sol = sp.solve([n + r - (3 * n + s), m + r - (3 * m + s)], [n, m], dict=True)[0]
    diagonal_only = bool(sp.simplify(sol[n] - sol[m]) == 0)
    # Concrete linear-dependence counterexample: J_{2,0} = sech^2 = 1 - tanh^2
    #   = J_{0,0} - J_{2,2}
    lhs = 1 / sp.cosh(sp.Symbol("x")) ** 2
    rhs = 1 - sp.tanh(sp.Symbol("x")) ** 2
    dependence_example = bool(sp.simplify(lhs - rhs) == 0)
    return {
        "eq28_forces_n_equals_m": diagonal_only,
        "n_solution": str(sol[n]),
        "off_diagonal_pairs_unsupported": 42,
        "J_index_linear_dependence_example": dependence_example,
        "statement": ("The paper's Eq. (28) supports only n = m; the 49-pair grid "
                      "adds 42 off-diagonal pairs without justification. Moreover "
                      "J_{2,0} = J_{0,0} - J_{2,2} shows distinct J-indices are not "
                      "linearly independent, so index matching is neither necessary "
                      "nor sufficient for closure."),
    }


def residual_split_discrepancies() -> list[str]:
    """Compare the derived real/imag split with the paper's Eqs. (37)-(38)."""
    return [
        "Paper Eq. (37) places -(beta3/6) v_ttt in F_u; the correct real part of "
        "(beta3/6)A_ttt is +(beta3/6) u_ttt.",
        "Paper Eq. (37) places -(beta4/24) u_tttt in F_u; the correct real part of "
        "(i beta4/24)A_tttt is -(beta4/24) v_tttt.",
        "Paper Eq. (38) places +(beta3/6) u_ttt in F_v; correct is +(beta3/6) v_ttt.",
        "Paper Eq. (38) places -(beta4/24) v_tttt in F_v; correct is +(beta4/24) u_tttt.",
        "Paper Eqs. (37)-(38) keep the complex coefficient (gamma + i alpha2/2) "
        "outside Re[.]/Im[.], which is ill-formed for real residuals; the correct "
        "split is -(alpha2/2)Q u - gamma Q v in F_u and -(alpha2/2)Q v + gamma Q u "
        "in F_v with Q = |A|^2/(1+Gamma|A|^2).",
        "Abstract and Section 2 say 'eight physical parameters' but list nine: "
        "beta2, beta3, beta4, g0, T2, alpha, alpha2, gamma, Gamma (Section 9.3 "
        "itself says 'up to nine independent physical coefficients').",
        "Paper Eq. (36) prints 'C6 J_{2,2} + C7 J_{2,2}'; for (n,m)=(1,1) the C7 "
        "term multiplies J_{3,3} = tanh^3, not J_{2,2}.",
    ]
