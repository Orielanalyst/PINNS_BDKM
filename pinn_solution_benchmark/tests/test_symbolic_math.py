"""Unit tests for the symbolic layers: J derivatives, residual split,
offset coefficients vs paper, closure certificate sanity."""
import numpy as np
import sympy as sp
import pytest

from pinn_benchmark import equations as eqs
from pinn_benchmark.ansatz import J_expr, XI, T, log_derivatives_T, J_numeric, singularities
from pinn_benchmark import symbolic_audit as sa


class TestJDerivatives:
    @pytest.mark.parametrize("n,m", [(1, 1), (1, 0), (-1, -1), ("3/2", "1/2"), ("-1/2", "3/2")])
    def test_log_derivatives_match_direct(self, n, m):
        """phi_k(T) from the recurrence == direct d^k J/dxi^k / J at sample points."""
        from fractions import Fraction
        J = J_expr(Fraction(n), Fraction(m))
        phis = log_derivatives_T(Fraction(n), Fraction(m), order=4)
        for xi_v in [0.3, 0.9, 1.7]:
            Tv = float(np.tanh(xi_v))
            for k, phi in enumerate(phis, start=1):
                direct = float(sp.diff(J, XI, k).subs(XI, xi_v) / J.subs(XI, xi_v))
                viaT = float(phi.subs(T, Tv))
                assert direct == pytest.approx(viaT, rel=1e-10), (n, m, k)

    def test_paper_differentiation_rule(self):
        """dJ_{n,m}/dxi = m J_{n-1,m-1} - n J_{n+1,m+1} (paper Eq. 7)."""
        n, m = sp.Rational(3, 2), sp.Rational(-1, 2)
        J = sp.sinh(XI) ** m / sp.cosh(XI) ** n
        rhs = (m * sp.sinh(XI) ** (m - 1) / sp.cosh(XI) ** (n - 1)
               - n * sp.sinh(XI) ** (m + 1) / sp.cosh(XI) ** (n + 1))
        assert sp.simplify(sp.diff(J, XI) - rhs) == 0


class TestResidualSplit:
    @staticmethod
    def _eval(expr, pvals, zv, tv):
        f = sp.lambdify((eqs.z_s, eqs.t_s), expr.subs(pvals), "numpy")
        return complex(f(zv, tv))

    def test_complex_equals_split_random_fields(self):
        """F_complex == F_u + i F_v for random smooth trial fields/params
        (numeric identity check at random points, no symbolic expansion)."""
        rng = np.random.default_rng(7)
        for trial in range(3):
            t = eqs.t_s
            z = eqs.z_s
            c1, c2, c3, c4 = [sp.Rational(int(x), 10) for x in rng.integers(-9, 9, 4)]
            u = c1 * sp.exp(-((t - c2) ** 2)) * sp.cos(z + c3 * t)
            v = (c4 + sp.Rational(1, 20)) * sp.tanh(t + c3) * sp.sin(2 * z - t)
            F = eqs.complex_residual_sympy(u + sp.I * v)
            Fu, Fv = eqs.split_residual_sympy(u, v)
            pvals = {s: sp.Rational(int(x), 7) for s, x in
                     zip(eqs.PARAMS.values(), rng.integers(1, 9, 9))}
            for zv, tv in [(0.2, 0.5), (0.8, -0.7), (0.4, 1.1)]:
                lhs = self._eval(F, pvals, zv, tv)
                rhs = (self._eval(Fu, pvals, zv, tv)
                       + 1j * self._eval(Fv, pvals, zv, tv))
                scale = max(abs(lhs), abs(rhs), 1.0)
                assert abs(lhs - rhs) / scale < 1e-12

    def test_paper_split_is_wrong(self):
        """The paper's printed Eqs. (37)-(38) do NOT reproduce the complex
        residual (documented discrepancy)."""
        z, t = eqs.z_s, eqs.t_s
        u = sp.exp(-t**2) * sp.cos(z)
        v = sp.sin(t) * sp.cos(2 * z)
        F = eqs.complex_residual_sympy(u + sp.I * v)
        Fu_p, Fv_p = eqs.paper_split_sympy(u, v)
        pvals = {s: sp.Rational(k + 1, 3) for k, s in enumerate(eqs.PARAMS.values())}
        lhs = self._eval(F, pvals, 0.3, 0.9)
        rhs = self._eval(Fu_p, pvals, 0.3, 0.9) + 1j * self._eval(Fv_p, pvals, 0.3, 0.9)
        assert abs(lhs - rhs) > 1e-6


class TestOffsetCoefficients:
    def test_derived_matches_paper_general_formulas(self):
        """Independent derivation reproduces paper Eqs. (17)-(25) for symbolic n,m."""
        cmp = sa.compare_offset_coefficients()
        assert all(cmp.values()), cmp

    def test_appendix_spot_checks(self):
        """Spot-check printed appendix entries: (1,1) column and n=3/2 C8, C9."""
        n, m = sp.Integer(1), sp.Integer(1)
        C = sa.derive_offset_coefficients(n, m)["C"]
        lam1, lam2 = sa.lam1, sa.lam2
        b3, b4 = sa.b3, sa.b4
        assert sp.simplify(C[-4]) == 0 and sp.simplify(C[-3]) == 0 and sp.simplify(C[-2]) == 0
        assert sp.simplify(C[-1] - (lam1 - b3 / 3 * lam2**3)) == 0
        assert sp.simplify(C[3] - (-b3 * lam2**3)) == 0
        assert sp.simplify(C[4] - sp.I * b4 * lam2**4) == 0
        n32 = sa.derive_offset_coefficients(sp.Rational(3, 2), sp.Rational(1, 2))["C"]
        assert sp.simplify(n32[3] - (-sp.Rational(35, 16) * b3 * lam2**3)) == 0
        assert sp.simplify(n32[4] - sp.I * sp.Rational(315, 128) * b4 * lam2**4) == 0


class TestClassification:
    def test_eq28_forces_diagonal(self):
        res = sa.classification_analysis()
        assert res["eq28_forces_n_equals_m"]
        assert res["J_index_linear_dependence_example"]


class TestFractionalConventions:
    def test_conventions_differ_for_negative_xi(self):
        vals_signed = J_numeric("1/2", "1/2", [-1.0], convention="signed")
        vals_princ = J_numeric("1/2", "1/2", [-1.0], convention="principal")
        assert np.isreal(vals_signed[0])
        assert np.iscomplex(vals_princ[0]) and abs(vals_princ[0].imag) > 0.1
        with pytest.raises(ValueError):
            J_numeric("1/2", "1/2", [-1.0], convention="positive_xi")

    def test_singularity_detection(self):
        assert singularities(-1, -1)["pole_at_0"]
        assert not singularities(1, 1)["pole_at_0"]
        assert singularities(1, 1)["smooth_C4_at_0"]
        assert not singularities("3/2", "3/2")["smooth_C4_at_0"]
        assert singularities(1, 0)["decays_at_inf"]
        assert not singularities(-1, 0)["bounded_at_inf"]
