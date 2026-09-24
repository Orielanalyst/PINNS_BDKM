"""Dataset split leakage, deterministic seeding, observation noise, metrics."""
import json
from pathlib import Path

import numpy as np
import pytest

from pinn_benchmark.datasets import (BranchSolution, make_forward_data,
                                     make_observations)
from pinn_benchmark import metrics

ROOT = Path(__file__).resolve().parents[1]


def kink_solution():
    return BranchSolution(
        n=1, m=1,
        params={"beta2": 0, "beta3": 0, "beta4": 0, "g0": 1.0,
                "T2": float(np.sqrt(0.5)), "alpha": 1.64, "alpha2": -0.64,
                "gamma": 0, "Gamma": 0},
        a=1.0, lambda1=0.0, lambda2=0.8, xi_domain=(-4.0, 4.0))


class TestDatasets:
    def test_no_leakage_train_test(self):
        sol = kink_solution()
        d = make_forward_data(sol, 500, 64, 64, seed=3)
        Z, Tm = d.test_grid
        test_pts = set(map(tuple, np.round(np.stack([Z.ravel(), Tm.ravel()], 1), 12)))
        train_pts = set(map(tuple, np.round(d.interior, 12)))
        assert not (test_pts & train_pts), "test grid leaked into training interior"
        res_pts = set(map(tuple, np.round(d.residual_test, 12)))
        assert not (res_pts & train_pts), "residual test points leaked into training"

    def test_deterministic_seeding(self):
        sol = kink_solution()
        d1 = make_forward_data(sol, 200, 32, 32, seed=7)
        d2 = make_forward_data(sol, 200, 32, 32, seed=7)
        d3 = make_forward_data(sol, 200, 32, 32, seed=8)
        assert np.array_equal(d1.interior, d2.interior)
        assert not np.array_equal(d1.interior, d3.interior)

    def test_ground_truth_is_exact_solution(self):
        sol = kink_solution()
        d = make_forward_data(sol, 50, 16, 16, seed=0)
        u = d.initial_uv[:, 0]
        expected = sol.a * np.tanh(sol.lambda2 * d.initial[:, 1])
        assert np.allclose(u, expected)
        assert np.allclose(d.initial_uv[:, 1], 0)

    def test_observation_noise_scaling(self):
        sol = kink_solution()
        clean = make_observations(sol, 0.1, 0.0, seed=1)
        noisy = make_observations(sol, 0.1, 0.05, seed=1)
        assert clean["sigma"] == 0
        assert noisy["sigma"] > 0
        assert np.allclose(clean["uv"], clean["uv_clean"])
        diff = noisy["uv"] - noisy["uv_clean"]
        assert 0.2 * noisy["sigma"] < diff.std() < 3 * noisy["sigma"]

    def test_domain_respects_xi_window(self):
        sol = kink_solution()
        t_lo, t_hi = sol.t_domain
        assert t_lo == pytest.approx(-5.0)
        assert t_hi == pytest.approx(5.0)


class TestMetrics:
    def test_l2re_zero_for_exact(self):
        x = np.random.default_rng(0).normal(size=(50, 2))
        assert metrics.l2re_complex(x, x) == 0

    def test_l2re_known_value(self):
        true = np.zeros((4, 2)); true[:, 0] = 1.0
        pred = true.copy(); pred[:, 0] = 1.1
        assert metrics.l2re_complex(pred, true) == pytest.approx(0.1, rel=1e-9)

    def test_intensity_and_component_errors(self):
        true = np.ones((10, 2))
        pred = true * 1.05
        m = metrics.all_field_metrics(pred, true)
        assert m["rel_err_real"] == pytest.approx(0.05, rel=1e-9)
        assert m["max_pointwise_error"] == pytest.approx(np.hypot(0.05, 0.05), rel=1e-9)
        assert m["intensity_error"] == pytest.approx(abs(1.05**2 * 2 - 2) / 2, rel=1e-9)

    def test_error_vs_z_shapes(self):
        true = np.random.default_rng(1).normal(size=(6 * 4, 2))
        out = metrics.error_vs_z(true, true, np.arange(6), (6, 4))
        assert len(out["l2re"]) == 6 and max(out["l2re"]) == 0


class TestCatalogIntegrity:
    def test_catalog_exists_and_verified(self):
        path = ROOT / "data" / "verified_branches" / "branch_catalog.json"
        assert path.exists(), "run scripts/run_audit.py first"
        cat = json.loads(path.read_text())
        assert len(cat) >= 3
        for rec in cat:
            assert rec["verified_numeric"], rec["branch_id"]
            assert rec["verified_symbolic"], rec["branch_id"]
            assert float(rec["residual_max"]) < 1e-10
            assert len(rec["representative_sets"]) >= 1

    def test_no_offdiagonal_without_verification(self):
        """Every accepted record passed the independent residual test; the
        49-branch claim itself must NOT be reflected in the catalog."""
        path = ROOT / "data" / "verified_branches" / "branch_catalog.json"
        cat = json.loads(path.read_text())
        assert len(cat) < 49
