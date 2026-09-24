"""Closure-equation verification and degenerate-solution rejection."""
from fractions import Fraction

import numpy as np
import pytest
import sympy as sp

from pinn_benchmark.symbolic_audit import (closure_certificate, asq, c_sym, ell_sym,
                                           lam1, lam2, b2, b3, b4, gam, a2c, Gam)
from pinn_benchmark.closure_solver import (solve_closure, verify_family_residual,
                                           verify_family_symbolic)


KINK_SUBS = {Gam: 0, b4: 0, b2: 0, gam: 0, b3: 0, lam1: 0,
             ell_sym: -c_sym * lam2**2, a2c: -2 * c_sym * lam2**2 / asq}


class TestClosure:
    def test_kink_closure_satisfies_certificate(self):
        assert verify_family_symbolic(1, 1, KINK_SUBS)

    def test_wrong_closure_fails_certificate(self):
        bad = dict(KINK_SUBS)
        bad[ell_sym] = -c_sym * lam2**2 / 2  # wrong balance
        assert not verify_family_symbolic(1, 1, bad)

    def test_high_precision_residual_kink(self):
        params = {"beta2": 0, "beta3": 0, "beta4": 0, "g0": 1,
                  "T2": sp.sqrt(sp.Rational(1, 2)), "alpha": sp.Rational(41, 25),
                  "alpha2": sp.Rational(-16, 25), "gamma": 0, "Gamma": 0,
                  "a": 1, "lambda1": 0, "lambda2": sp.Rational(4, 5)}
        v = verify_family_residual(1, 1, params, dps=50)
        assert v["residual_max"] < 1e-40  # exact rational closure -> ~0 at 50 dps

    def test_point_fitted_rejected_by_dense_grid(self):
        """Parameters tuned to kill the residual at ONE point must fail the
        dense-grid test (guards against point-fitted 'solutions')."""
        params = {"beta2": 0, "beta3": 0, "beta4": 0, "g0": 1,
                  "T2": sp.sqrt(sp.Rational(1, 2)), "alpha": sp.Rational(3, 2),
                  "alpha2": sp.Rational(-16, 25), "gamma": 0, "Gamma": 0,
                  "a": 1, "lambda1": 0, "lambda2": sp.Rational(4, 5)}
        v = verify_family_residual(1, 1, params, dps=30)
        assert v["residual_max"] > 1e-6

    def test_degenerate_families_flagged(self):
        cs = closure_certificate(Fraction(1), Fraction(1))
        fams = solve_closure(cs)
        lam2_zero = [f for f in fams if f.degenerate_reasons]
        assert lam2_zero, "the lambda2=0 degenerate family must be flagged, not dropped"
        good = [f for f in fams if not f.degenerate_reasons]
        assert any(ell_sym in f.substitutions for f in good)

    def test_a_zero_never_accepted(self):
        cs = closure_certificate(Fraction(1), Fraction(1))
        fams = solve_closure(cs)
        for f in fams:
            assert f.substitutions.get(asq, 1) != 0

    def test_fractional_pair_only_vacuous(self):
        """(1/2,1/2): every family kills every physical coefficient."""
        cs = closure_certificate(Fraction(1, 2), Fraction(1, 2))
        fams = solve_closure(cs)
        for f in fams:
            subs = {str(k): sp.simplify(v) for k, v in f.substitutions.items()}
            nontrivial = [k for k, v in subs.items()
                          if k in ("beta2", "beta3", "beta4", "gamma", "alpha2", "c", "ell")
                          and v != 0]
            assert not nontrivial, f"unexpected nontrivial closure {subs}"
