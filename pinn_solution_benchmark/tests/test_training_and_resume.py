"""Checkpoint/resume, deterministic seeding of training, end-to-end smoke,
and the numerical solver's basic correctness."""
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from pinn_benchmark.datasets import BranchSolution
from pinn_benchmark.trainers import train_forward

TMP = Path(__file__).resolve().parent / "_tmp_runs"


def tiny_cfg(adam=60, lbfgs=0):
    return {"training": {"adam_steps": adam, "adam_lr": 2e-3, "lbfgs_steps": lbfgs,
                         "n_collocation": 96, "n_initial": 24, "n_boundary": 24,
                         "checkpoint_every": 20},
            "network": {"hidden_layers": 2, "width": 12},
            "metrics": {"test_grid": [12, 24], "n_residual_test": 100}}


def kink():
    return BranchSolution(
        n=1, m=1,
        params={"beta2": 0, "beta3": 0, "beta4": 0, "g0": 1.0,
                "T2": float(np.sqrt(0.5)), "alpha": 1.64, "alpha2": -0.64,
                "gamma": 0, "Gamma": 0},
        a=1.0, lambda1=0.0, lambda2=0.8, xi_domain=(-4.0, 4.0))


@pytest.fixture(autouse=True)
def _clean_tmp():
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True)
    yield
    shutil.rmtree(TMP, ignore_errors=True)


class TestTraining:
    def test_deterministic_same_seed(self):
        m1 = train_forward(kink(), tiny_cfg(), "vanilla", 5, TMP / "a", "cpu")
        m2 = train_forward(kink(), tiny_cfg(), "vanilla", 5, TMP / "b", "cpu")
        assert m1["l2re_complex"] == pytest.approx(m2["l2re_complex"], rel=1e-10)

    def test_different_seed_differs(self):
        m1 = train_forward(kink(), tiny_cfg(), "vanilla", 1, TMP / "a", "cpu")
        m2 = train_forward(kink(), tiny_cfg(), "vanilla", 2, TMP / "b", "cpu")
        assert m1["l2re_complex"] != m2["l2re_complex"]

    def test_checkpoint_resume_continues(self):
        """Interrupt after 30 steps, resume to 60; history must be continuous
        and the checkpoint step advanced."""
        cfg_half = tiny_cfg(adam=30)
        train_forward(kink(), cfg_half, "vanilla", 3, TMP / "r", "cpu")
        ck = torch.load(TMP / "r" / "checkpoint.pt", weights_only=False)
        assert ck["step"] == 30
        cfg_full = tiny_cfg(adam=60)
        m = train_forward(kink(), cfg_full, "vanilla", 3, TMP / "r", "cpu", resume=True)
        ck2 = torch.load(TMP / "r" / "checkpoint.pt", weights_only=False)
        assert ck2["step"] == 60
        steps = [h["step"] for h in ck2["history"] if h["phase"] == "adam"]
        assert min(steps) == 0 and max(steps) == 59
        assert np.isfinite(m["l2re_complex"])

    def test_end_to_end_smoke_learns(self):
        """A 200-step run must fit IC/BC to <1e-1 and reduce L2RE well below
        the trivial zero-prediction error (which is 1.0)."""
        m = train_forward(kink(), tiny_cfg(adam=200, lbfgs=20), "vanilla", 0,
                          TMP / "s", "cpu")
        assert m["l2re_complex"] < 0.5
        assert m["initial_rmse"] < 0.1
        assert np.isfinite(m["pde_residual_rms_independent"])

    def test_all_loss_components_recorded(self):
        m = train_forward(kink(), tiny_cfg(), "vanilla", 0, TMP / "c", "cpu")
        last = [h for h in m["loss_history"] if h["phase"] == "adam"][-1]
        for key in ("initial", "boundary", "pde", "total"):
            assert key in last and np.isfinite(last[key])


class TestNumericalSolver:
    def test_kink_fd_accuracy(self):
        from pinn_benchmark.numerical_solver import solve_branch
        r = solve_branch(kink(), "fd_mol", nt=97, rtol=1e-9, atol=1e-11)
        assert r["final_l2re"] < 1e-6

    def test_periodic_not_applied_to_kink(self):
        """choose_scheme must not give a kink profile periodic BCs."""
        from pinn_benchmark.numerical_solver import choose_scheme
        rec = {"properties": {"localized": False, "feat_gradient_max": 0.8,
                              "singular_at_xi0": False, "fractional_power": False}}
        assert choose_scheme(rec) == "fd_mol"
