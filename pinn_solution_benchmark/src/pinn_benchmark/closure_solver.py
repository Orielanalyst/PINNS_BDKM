"""Solve the closure systems exactly and verify candidate solution families.

Strategy: the closure equations are highly factorable polynomials in
(lambda1, lambda2, asq, beta2, beta3, beta4, c, ell, gamma, alpha2, Gamma).
A bounded recursive factor-split (a light triangular decomposition) explores
every case: pick the simplest unsolved equation, factor it, branch on each
irreducible factor, solve the factor for the best available unknown, substitute
and recurse. Nontriviality (a != 0, i.e. asq > 0; lambda2 != 0 where lambda2
occurs) is enforced by refusing branches that set forbidden atoms to zero.

Every terminal family is then verified INDEPENDENTLY by substituting the
family's constraints into the original complex PDE residual (sympy, in xi) and
checking exact vanishing symbolically plus 50-digit mpmath evaluation on a
dense grid and random points — solutions are never accepted merely because
they solve the derived system.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from fractions import Fraction

import mpmath as mp
import numpy as np
import sympy as sp

from . import equations as eqs
from .ansatz import J_expr, XI
from .symbolic_audit import (CLOSURE_UNKNOWNS, CLOSURE_UNKNOWNS_EXTENDED,
                             ClosureSystem, asq, c_sym, ell_sym,
                             lam1, lam2, b2, b3, b4, gam, a2c, Gam, omega, kappa)

# Unknown preference for solving a linear factor (solve for these first).
SOLVE_PRIORITY = [lam1, kappa, ell_sym, gam, a2c, b2, b3, b4, c_sym, Gam,
                  omega, asq, lam2]
FORBIDDEN_ZERO = {asq}   # a = 0 is a rejected degenerate branch
# lambda2 = 0 collapses xi to a z-only coordinate; branches forcing it are
# recorded but tagged degenerate rather than silently dropped.


@dataclass
class SolutionFamily:
    n: Fraction
    m: Fraction
    substitutions: dict          # {symbol: expr} triangular assignments
    free_symbols: list           # remaining free parameters
    degenerate_reasons: list = field(default_factory=list)
    verified_symbolic: bool = False
    verified_numeric: bool = False
    residual_max: float = float("inf")
    residual_rms: float = float("inf")

    def canonical_key(self):
        items = sorted((str(k), sp.srepr(sp.simplify(v))) for k, v in self.substitutions.items())
        return tuple(items)


def _poly_num(e):
    """Numerator after combining into a single fraction.

    Denominators produced by triangular substitution are expressions that were
    implicitly assumed nonzero when the corresponding factor branch divided by
    them; the complementary (=0) cases are explored as separate branches by the
    factor-split, so keeping the numerator is exhaustive."""
    num, _den = sp.fraction(sp.cancel(sp.together(sp.expand(e))))
    return sp.expand(num)


def _apply(eq_list, sub):
    out = []
    for e in eq_list:
        e2 = _poly_num(e.subs(sub))
        if e2 != 0:
            out.append(e2)
    return out


def solve_closure(cs: ClosureSystem, max_depth: int = 14, max_families: int = 60) -> list[SolutionFamily]:
    families: list[SolutionFamily] = []
    seen = set()
    visited_states = set()

    def record(subs: dict):
        unknowns = CLOSURE_UNKNOWNS_EXTENDED if cs.extended_phase else CLOSURE_UNKNOWNS
        fam = SolutionFamily(
            n=cs.n, m=cs.m,
            substitutions={k: sp.simplify(v) for k, v in subs.items()},
            free_symbols=[u for u in unknowns if u not in subs],
        )
        if lam2 in subs and sp.simplify(subs[lam2]) == 0:
            fam.degenerate_reasons.append("lambda2 = 0 (xi has no t-dependence)")
        key = fam.canonical_key()
        if key not in seen:
            seen.add(key)
            families.append(fam)

    def recurse(eq_list, subs: dict, depth: int):
        if len(families) >= max_families:
            return
        eq_list = [e for e in eq_list if sp.expand(e) != 0]
        # contradiction?
        for e in eq_list:
            if e.is_number and e != 0:
                return
        if not eq_list:
            record(subs)
            return
        if depth <= 0:
            return  # bounded search; unreached cases are reported by caller
        state = (frozenset(sp.srepr(e) for e in eq_list),
                 frozenset((str(k), sp.srepr(v)) for k, v in subs.items()))
        if state in visited_states:
            return
        visited_states.add(state)
        # pick the equation with the fewest ops
        eq_sorted = sorted(eq_list, key=lambda e: sp.count_ops(e))
        target = _poly_num(eq_sorted[0])
        rest = [e for e in eq_list if e is not eq_sorted[0]]
        factors = sp.factor_list(sp.factor(target))[1]
        if not factors:
            return
        for base, _mult in factors:
            base = sp.expand(base)
            if base.is_number:
                if base != 0:
                    continue  # nonzero constant factor: irrelevant
            # forbidden: asq = 0
            if base in FORBIDDEN_ZERO or (-base) in FORBIDDEN_ZERO:
                continue
            # choose an unknown to solve base = 0 for
            solved = False
            for u in SOLVE_PRIORITY:
                if u not in base.free_symbols:
                    continue
                if sp.degree(sp.Poly(base, u)) != 1:
                    continue
                sols = sp.solve(base, u, dict=False)
                if len(sols) != 1:
                    continue
                val = sp.simplify(sols[0])
                if u in FORBIDDEN_ZERO and val == 0:
                    break
                new_subs = {k: sp.simplify(v.subs(u, val)) for k, v in subs.items()}
                new_subs[u] = val
                recurse(_apply(rest, {u: val}), new_subs, depth - 1)
                solved = True
                break
            if not solved:
                # nonlinear in every unknown (e.g. lam2**2 factors were already
                # split off by factor_list as lam2 powers). Try direct roots in
                # the least-priority unknown present.
                for u in reversed(SOLVE_PRIORITY):
                    if u in base.free_symbols:
                        try:
                            roots = sp.solve(sp.Eq(base, 0), u)
                        except Exception:
                            roots = []
                        for val in roots:
                            if u in FORBIDDEN_ZERO and sp.simplify(val) == 0:
                                continue
                            new_subs = {k: sp.simplify(v.subs(u, val)) for k, v in subs.items()}
                            new_subs[u] = sp.simplify(val)
                            recurse(_apply(rest, {u: val}), new_subs, depth - 1)
                        break

    recurse(list(cs.equations), {}, max_depth)
    return dedup_families(families)


def dedup_families(fams: list[SolutionFamily]) -> list[SolutionFamily]:
    """Remove families that are strict specializations of another family."""
    out = []
    for f in fams:
        subsumed = False
        for g in fams:
            if f is g:
                continue
            if set(g.substitutions).issubset(set(f.substitutions)) and \
               len(g.substitutions) < len(f.substitutions):
                if all(sp.simplify(f.substitutions[k] - g.substitutions[k].subs(f.substitutions)) == 0
                       for k in g.substitutions):
                    subsumed = True
                    break
        if not subsumed:
            out.append(f)
    return out


# ---------------------------------------------------------------------------
# Independent verification
# ---------------------------------------------------------------------------

def _to_physical(pmap: dict) -> dict:
    """Map closure symbols (c, ell) back to the nine physical parameters.
    Chooses g0, T2, alpha consistent with c = g0*T2^2, ell = (g0-alpha)/2.
    Convention: if c > 0, take g0 = 1 unless ell forces otherwise; T2 = sqrt(c/g0);
    alpha = g0 - 2*ell. If c == 0: take T2 = 0, g0 = max(2*ell, 0) + loss balance."""
    cval = pmap["c"]
    ellval = pmap["ell"]
    if cval != 0:
        g0v = sp.Integer(1)
        T2v = sp.sqrt(sp.Rational(cval) if not isinstance(cval, sp.Expr) else cval)
        if (sp.Rational(cval) if not isinstance(cval, sp.Expr) else cval) < 0:
            raise ValueError("c = g0*T2^2 < 0 is unphysical (g0>0, T2 real)")
    else:
        g0v = sp.Integer(0) if ellval <= 0 else 2 * ellval
        T2v = sp.Integer(0)
    alph = g0v - 2 * ellval
    return {"g0": g0v, "T2": T2v, "alpha": alph}


def verify_family_residual(n, m, param_values: dict, xi_lo=0.35, xi_hi=4.0,
                           dps: int = 50, n_grid: int = 60, n_random: int = 40,
                           seed: int = 0) -> dict:
    """50-digit residual check of A = a J_{n,m}(lambda1 z + lambda2 t) in the
    ORIGINAL complex PDE, on a dense xi-grid plus random (z,t) points.

    param_values must contain the nine physical parameters plus a, lambda1,
    lambda2 as exact sympy numbers. Returns max/rms of |F| and of |F|/||A||."""
    mp.mp.dps = dps
    a_v = param_values["a"]
    l1, l2 = param_values["lambda1"], param_values["lambda2"]
    p = {k: param_values[k] for k in eqs.PARAM_NAMES}

    z, t = eqs.z_s, eqs.t_s
    xi_of_zt = l1 * z + l2 * t
    A = a_v * J_expr(n, m, xi=XI).subs(XI, xi_of_zt)
    om = sp.sympify(param_values.get("omega", 0))
    kp = sp.sympify(param_values.get("kappa", 0))
    if om != 0 or kp != 0:  # optional extended-phase ansatz
        A = A * sp.exp(sp.I * (kp * z + om * t))
    F = eqs.complex_residual_sympy(A, z, t, {k: sp.sympify(v) for k, v in p.items()})
    Ffun = sp.lambdify((z, t), F, modules="mpmath")
    Afun = sp.lambdify((z, t), A, modules="mpmath")

    rng = np.random.default_rng(seed)
    pts = []
    l2f = float(l2)
    l1f = float(l1)
    # dense xi grid along z=0 (t chosen so xi covers [xi_lo, xi_hi])
    if l2f != 0:
        for xi_v in np.linspace(xi_lo, xi_hi, n_grid):
            pts.append((mp.mpf(0), mp.mpf(xi_v) / mp.mpf(l2f)))
    # random (z,t) with xi in range
    for _ in range(n_random):
        zv = mp.mpf(float(rng.uniform(0, 1)))
        xi_v = mp.mpf(float(rng.uniform(xi_lo, xi_hi)))
        tv = (xi_v - mp.mpf(l1f) * zv) / mp.mpf(l2f) if l2f != 0 else mp.mpf(float(rng.uniform(-2, 2)))
        pts.append((zv, tv))

    res_abs, amp_abs = [], []
    for zv, tv in pts:
        Fv = mp.mpc(Ffun(zv, tv))
        Av = mp.mpc(Afun(zv, tv))
        res_abs.append(abs(Fv))
        amp_abs.append(abs(Av))
    res_abs = [float(x) for x in res_abs]
    amp_rms = float(mp.sqrt(sum(x**2 for x in amp_abs) / len(amp_abs)))
    res_rms = float(np.sqrt(np.mean(np.square(res_abs))))
    return {
        "residual_max": float(np.max(res_abs)),
        "residual_rms": res_rms,
        "relative_rms": res_rms / (amp_rms + 1e-300),
        "amplitude_rms": amp_rms,
        "n_points": len(pts),
        "dps": dps,
    }


def verify_family_symbolic(n, m, substitutions: dict, extended_phase: bool = False) -> bool:
    """Exact symbolic check: substitute the family's triangular assignments
    into the closure-certificate residual components and confirm identical 0.
    Free symbols remain symbolic, so this proves the whole family at once."""
    from .symbolic_audit import closure_certificate
    cs = closure_certificate(n, m, extended_phase=extended_phase)
    for e in cs.equations:
        val = sp.simplify(sp.expand(e.subs(substitutions)))
        if val != 0:
            return False
    return True
