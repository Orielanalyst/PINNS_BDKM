"""Classify solved closure families into tiers, build representative parameter
sets, compute difficulty features, and write the verified-branch catalog.

Tiers (from the study protocol):
  A: nontrivial full-model solutions (4th-order dispersion, saturation and
     meaningful gain/loss all simultaneously active),
  B: nontrivial reduced-model solutions (some physical terms forced to vanish
     by the closure, but at least two PDE terms in genuine balance),
  C: trivial / degenerate / vacuous / unsupported candidates (excluded from
     PINN benchmarking).

Benchmark groups:
  main:   smooth, finite, bounded, unambiguously defined on their domain,
  stress: singular, unbounded, or fractional-power (branch-cut-dependent),
  none:   Tier C.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field, asdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as sp

from . import equations as eqs
from .ansatz import J_expr, XI, is_fractional, singularities
from .closure_solver import SolutionFamily, verify_family_residual, _to_physical
from .symbolic_audit import (asq, c_sym, ell_sym, lam1, lam2, b2, b3, b4, gam,
                             a2c, Gam, omega, kappa)

SYM_TO_NAME = {b2: "beta2", b3: "beta3", b4: "beta4", gam: "gamma", a2c: "alpha2",
               Gam: "Gamma", c_sym: "c", ell_sym: "ell", lam1: "lambda1",
               lam2: "lambda2", asq: "asq", omega: "omega", kappa: "kappa"}

# Exact preference values for free symbols (two sets -> two representatives).
FREE_VALUE_SETS = [
    {lam2: sp.Rational(4, 5), asq: sp.Integer(1), c_sym: sp.Rational(1, 2),
     Gam: sp.Rational(1, 2), b2: sp.Rational(1, 5), b3: sp.Rational(1, 5),
     b4: sp.Rational(1, 10), gam: sp.Integer(1), a2c: sp.Rational(1, 5),
     ell_sym: sp.Rational(1, 10), lam1: sp.Rational(1, 5),
     omega: sp.Rational(1, 2), kappa: sp.Rational(1, 4)},
    {lam2: sp.Rational(13, 10), asq: sp.Rational(36, 25), c_sym: sp.Rational(3, 4),
     Gam: sp.Integer(1), b2: sp.Rational(-1, 5), b3: sp.Rational(1, 10),
     b4: sp.Rational(1, 20), gam: sp.Rational(1, 2), a2c: sp.Rational(3, 10),
     ell_sym: sp.Rational(-1, 10), lam1: sp.Rational(1, 10),
     omega: sp.Rational(3, 4), kappa: sp.Rational(1, 2)},
]

TERM_NAMES = ["A_z", "beta2_GVD", "gain_bandwidth_c", "beta3_TOD", "beta4_FOD",
              "kerr_gamma", "two_photon_alpha2", "linear_gain_loss", "saturation_Gamma"]


@dataclass
class BranchRecord:
    branch_id: str
    n: str
    m: str
    family_index: int
    tier: str = "C"
    group: str = "none"
    constraints: dict = field(default_factory=dict)
    free_parameters: list = field(default_factory=list)
    representative_sets: list = field(default_factory=list)
    xi_domain: list = field(default_factory=lambda: [-4.0, 4.0])
    power_convention: str = "integer_exact"
    properties: dict = field(default_factory=dict)
    active_terms: list = field(default_factory=list)
    residual_max: float = float("nan")
    residual_rms_rel: float = float("nan")
    verified_symbolic: bool = False
    verified_numeric: bool = False
    rejection_reason: str = ""
    notes: list = field(default_factory=list)


def resolve_representative(fam: SolutionFamily, value_set: dict):
    """Assign exact values to free symbols, propagate through the family's
    triangular substitutions, and map to the nine physical parameters.
    Returns dict or None if signs are unphysical (c<0, Gamma<0, asq<=0)."""
    assign = {u: value_set[u] for u in fam.free_symbols if u in value_set}
    full = dict(assign)
    for k, v in fam.substitutions.items():
        full[k] = sp.simplify(sp.sympify(v).subs(assign))
    # residual free symbols inside substituted expressions?
    for k, v in full.items():
        if getattr(v, "free_symbols", set()):
            return None
    vals = {SYM_TO_NAME[s]: full.get(s, value_set.get(s)) for s in SYM_TO_NAME}
    if any(v is None for v in vals.values()):
        return None
    if float(sp.N(vals["asq"])) <= 0:
        return None
    if float(sp.N(vals["Gamma"])) < 0:
        return None
    if float(sp.N(vals["c"])) < 0:
        return None  # c = g0*T2^2 >= 0 required
    try:
        phys3 = _to_physical({"c": vals["c"], "ell": vals["ell"]})
    except ValueError:
        return None
    out = {k: vals[k] for k in ("beta2", "beta3", "beta4", "gamma", "alpha2", "Gamma")}
    out.update({k: sp.nsimplify(v) for k, v in phys3.items()})
    out["lambda1"] = vals["lambda1"]
    out["lambda2"] = vals["lambda2"]
    out["a"] = sp.sqrt(sp.Rational(vals["asq"]))
    out["omega"] = vals.get("omega", 0) or 0
    out["kappa"] = vals.get("kappa", 0) or 0
    return out


def term_activity(n, m, rep: dict, xi_grid) -> dict:
    """Max |term| of each PDE term evaluated on the exact solution."""
    z, t = eqs.z_s, eqs.t_s
    l1, l2 = rep["lambda1"], rep["lambda2"]
    A = rep["a"] * J_expr(n, m, xi=XI).subs(XI, l1 * z + l2 * t)
    om, kp = sp.sympify(rep.get("omega", 0)), sp.sympify(rep.get("kappa", 0))
    if om != 0 or kp != 0:
        A = A * sp.exp(sp.I * (kp * z + om * t))
    p = {k: sp.sympify(rep[k]) for k in eqs.PARAM_NAMES}
    cc = p["g0"] * p["T2"] ** 2
    absA2 = A * sp.conjugate(A)
    Qden = 1 + p["Gamma"] * absA2
    terms = {
        "A_z": sp.diff(A, z),
        "beta2_GVD": sp.I / 2 * p["beta2"] * sp.diff(A, t, 2),
        "gain_bandwidth_c": -cc / 2 * sp.diff(A, t, 2),
        "beta3_TOD": p["beta3"] / 6 * sp.diff(A, t, 3),
        "beta4_FOD": sp.I * p["beta4"] / 24 * sp.diff(A, t, 4),
        "kerr_gamma": sp.I * p["gamma"] * absA2 * A / Qden,
        "two_photon_alpha2": -p["alpha2"] / 2 * absA2 * A / Qden,
        "linear_gain_loss": (p["g0"] - p["alpha"]) / 2 * A,
        # saturation counted active if Gamma*|A|^2 materially deforms Q
        "saturation_Gamma": p["Gamma"] * absA2 * (p["gamma"] ** 2 + p["alpha2"] ** 2) ** sp.Rational(1, 2) * absA2 * A / Qden,
    }
    l2f = float(l2)
    out = {}
    for name, expr in terms.items():
        f = sp.lambdify((z, t), expr, modules=["numpy", {"conjugate": np.conjugate}])
        tv = np.array([x / l2f for x in xi_grid]) if l2f != 0 else np.linspace(-2, 2, len(xi_grid))
        with np.errstate(all="ignore"):
            vals = np.abs(np.asarray(f(np.zeros_like(tv), tv), dtype=complex))
        out[name] = float(np.nanmax(vals)) if np.size(vals) else 0.0
    return out


def difficulty_features(n, m, rep: dict, xi_grid) -> dict:
    """Profile-shape features used in the branch-difficulty analysis."""
    z, t = eqs.z_s, eqs.t_s
    l1, l2 = float(rep["lambda1"]), float(rep["lambda2"])
    Aexpr = rep["a"] * J_expr(n, m, xi=XI)
    d = {k: sp.lambdify(XI, sp.diff(Aexpr, XI, k), "numpy") for k in range(5)}
    xg = np.asarray(xi_grid, dtype=float)
    with np.errstate(all="ignore"):
        prof = {k: np.abs(np.asarray(d[k](xg), dtype=complex)) for k in d}
    # scale xi-derivatives to t-derivatives with lambda2
    feats = {
        "max_amplitude": float(np.nanmax(prof[0])),
        "max_curvature_tt": float(np.nanmax(prof[2]) * l2**2),
        "max_fourth_deriv_tttt": float(np.nanmax(prof[4]) * l2**4),
        "gradient_max": float(np.nanmax(prof[1]) * abs(l2)),
    }
    # localization width: xi-extent where |dA/dxi| > 10% of its max (kink/pulse edge)
    g = prof[1]
    if np.nanmax(g) > 0:
        mask = g > 0.1 * np.nanmax(g)
        width_xi = float(xg[mask][-1] - xg[mask][0]) if mask.any() else float("inf")
    else:
        width_xi = float("inf")
    feats["localization_width_t"] = width_xi / abs(l2) if l2 != 0 else float("inf")
    sing = singularities(n, m)
    feats["distance_to_singularity_xi"] = (float(np.min(np.abs(xg)))
                                           if (sing["pole_at_0"] or sing["fractional_branch_at_0"])
                                           else float("inf"))
    return feats


def classify_family(fam: SolutionFamily, family_index: int,
                    residual_tol_max: float = 1e-10, residual_tol_rel: float = 1e-10,
                    verbose_notes: bool = True, extended: bool = False) -> BranchRecord:
    n, m = fam.n, fam.m
    rec = BranchRecord(
        branch_id=f"n{n}_m{m}_f{family_index}".replace("/", "o").replace("-", "neg"),
        n=str(n), m=str(m), family_index=family_index,
        constraints={str(SYM_TO_NAME.get(k, k)): str(v) for k, v in fam.substitutions.items()},
        free_parameters=[str(SYM_TO_NAME.get(s, s)) for s in fam.free_symbols],
    )
    frac = is_fractional(n, m)
    sing = singularities(n, m)
    rec.power_convention = "positive_xi" if frac else "integer_exact"

    if fam.degenerate_reasons:
        rec.rejection_reason = "; ".join(fam.degenerate_reasons)
        return rec

    # domain: symmetric if smooth at 0 and no pole; else xi > 0 strictly
    if sing["smooth_C4_at_0"]:
        rec.xi_domain = [-4.0, 4.0]
    else:
        rec.xi_domain = [0.35, 4.35]
    xi_grid = np.linspace(rec.xi_domain[0] + 1e-9, rec.xi_domain[1], 121)
    if sing["smooth_C4_at_0"]:
        xi_grid = xi_grid[np.abs(xi_grid) > 1e-6]

    # representatives
    reps = []
    for vs in FREE_VALUE_SETS:
        r = resolve_representative(fam, vs)
        if r is not None:
            reps.append(r)
    if not reps:
        rec.rejection_reason = ("no physically admissible representative "
                                "(needs asq>0, c=g0*T2^2>=0, Gamma>=0)")
        return rec

    # verification on the first representative (numbers exact -> mpmath 50 dps)
    ver = verify_family_residual(n, m, reps[0],
                                 xi_lo=max(rec.xi_domain[0], 0.35) if not sing["smooth_C4_at_0"] else 0.35,
                                 xi_hi=rec.xi_domain[1])
    rec.residual_max = ver["residual_max"]
    rec.residual_rms_rel = ver["relative_rms"]
    rec.verified_numeric = (ver["residual_max"] < residual_tol_max
                            and ver["relative_rms"] < residual_tol_rel)
    from .closure_solver import verify_family_symbolic
    rec.verified_symbolic = verify_family_symbolic(n, m, fam.substitutions,
                                                   extended_phase=extended)
    if not rec.verified_numeric:
        rec.rejection_reason = (f"independent residual check failed: max|F|={ver['residual_max']:.2e}, "
                                f"rel RMS={ver['relative_rms']:.2e}")
        return rec

    # term activity / vacuousness
    act = term_activity(n, m, reps[0], xi_grid)
    scale = max(act.values()) if act else 0.0
    active = [k for k, v in act.items() if v > 1e-12 and v > 1e-9 * scale]
    rec.active_terms = active
    if len(active) <= 1:
        rec.rejection_reason = ("vacuous solution: at most one PDE term is active "
                                "(governing equation trivializes on this family)")
        rec.tier = "C"
        return rec

    rec.representative_sets = [{k: str(v) for k, v in r.items()} for r in reps]
    rec.properties = {
        "real_profile": not extended or all(
            sp.sympify(r.get("omega", 0)) == 0 and sp.sympify(r.get("kappa", 0)) == 0
            for r in reps),
        "bounded_on_domain": bool(np.isfinite(difficulty_features(n, m, reps[0], xi_grid)["max_amplitude"])),
        "bounded_on_R": sing["bounded_on_R"],
        "localized": sing["decays_at_inf"],
        "smooth_C4_on_domain": True,
        "smooth_C4_on_R": sing["smooth_C4_at_0"] and not sing["pole_at_0"],
        "fractional_power": frac,
        "singular_at_xi0": sing["pole_at_0"],
        **{f"feat_{k}": v for k, v in difficulty_features(n, m, reps[0], xi_grid).items()},
        **{f"activity_{k}": v for k, v in act.items()},
    }

    # tiering
    fod = "beta4_FOD" in active
    sat = "saturation_Gamma" in active
    gl = ("linear_gain_loss" in active) or ("two_photon_alpha2" in active)
    if fod and sat and gl:
        rec.tier = "A"
    else:
        rec.tier = "B"

    # benchmark group
    if sing["pole_at_0"] or not sing["bounded_at_inf"]:
        rec.group = "stress"
        if not sing["bounded_at_inf"]:
            rec.notes.append("profile unbounded as |xi|->inf; finite only on truncated domain")
    elif frac:
        rec.group = "stress"
        rec.notes.append("fractional power: domain restricted to xi>0 (positive_xi convention)")
    else:
        rec.group = "main"
    return rec


def write_catalog(records: list[BranchRecord], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    js = [asdict(r) for r in records]
    import json
    with open(out_dir / "branch_catalog.json", "w") as f:
        json.dump(js, f, indent=2, default=str)
    cols = ["branch_id", "n", "m", "family_index", "tier", "group", "verified_symbolic",
            "verified_numeric", "residual_max", "residual_rms_rel", "power_convention",
            "xi_domain", "free_parameters", "active_terms", "rejection_reason"]
    with open(out_dir / "branch_catalog.csv", "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(cols)
        for r in js:
            wtr.writerow([r[c] for c in cols])
