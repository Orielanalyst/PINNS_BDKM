"""Forward-PINN trainer: Adam warm-up + L-BFGS, checkpointing/resume, NaN
guards, gradient monitoring, three method variants:

  vanilla           fixed loss weights, fixed collocation points
  adaptive_weights  gradient-norm loss balancing (losses.GradNormBalancer)
  adaptive_sampling residual-adaptive collocation resampling (RAD)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .adaptive_sampling import residual_adaptive_resample
from .datasets import BranchSolution, ForwardData, make_forward_data, to_torch
from .losses import FixedWeights, GradNormBalancer, data_fit_loss, pde_residual_loss
from .metrics import all_field_metrics, error_vs_z, residual_metrics, spectral_error
from .networks import PINN, count_parameters
from .utils import log, set_seed


class NaNLossError(RuntimeError):
    pass


def build_model(cfg: dict, sol: BranchSolution, device: str, cond_dim: int = 0,
                extra_lo=(), extra_hi=()) -> PINN:
    t_lo, t_hi = sol.t_domain
    net = cfg.get("network", {})
    model = PINN(
        hidden_layers=net.get("hidden_layers", 6),
        width=net.get("width", 64),
        activation=net.get("activation", "tanh"),
        fourier_features=net.get("fourier_features", 0),
        fourier_sigma=net.get("fourier_sigma", 2.0),
        input_lo=(0.0, t_lo, *extra_lo),
        input_hi=(sol.z_max, t_hi, *extra_hi),
        cond_dim=cond_dim,
    ).to(device)
    return model


def train_forward(sol: BranchSolution, cfg: dict, method: str, seed: int,
                  run_dir: Path, device: str = "cpu", resume: bool = True) -> dict:
    """Train one forward PINN; returns the metrics dict (also saved by caller)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "checkpoint.pt"
    set_seed(seed)
    tr = cfg["training"]

    data = make_forward_data(
        sol, tr["n_collocation"], tr["n_initial"], tr["n_boundary"], seed,
        test_shape=tuple(cfg.get("metrics", {}).get("test_grid", [64, 128])),
        n_residual_test=cfg.get("metrics", {}).get("n_residual_test", 2000),
    )
    model = build_model(cfg, sol, device)
    params = {k: float(v) for k, v in sol.params.items()}

    if method == "adaptive_weights":
        balancer = GradNormBalancer(["initial", "boundary", "pde"],
                                    alpha=tr.get("balancer_alpha", 0.1),
                                    update_every=tr.get("balancer_every", 100))
    else:
        w = cfg.get("loss_weights", {"initial": 100.0, "boundary": 100.0, "pde": 1.0})
        balancer = FixedWeights(w)

    adam = torch.optim.Adam(model.parameters(), lr=tr.get("adam_lr", 1e-3))
    sched = torch.optim.lr_scheduler.ExponentialLR(
        adam, gamma=tr.get("adam_lr_decay", 1.0) ** (1.0 / max(tr["adam_steps"], 1)))

    start_step = 0
    history: list[dict] = []
    t_start = time.perf_counter()
    prev_elapsed = 0.0
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        adam.load_state_dict(ck["adam"])
        sched.load_state_dict(ck["sched"])
        start_step = ck["step"]
        history = ck["history"]
        balancer.weights = ck.get("balancer_weights", balancer.weights)
        torch.set_rng_state(ck["torch_rng"].cpu())
        np.random.set_state(ck["numpy_rng"])
        prev_elapsed = ck.get("elapsed", 0.0)
        log.info("resumed %s at step %d", run_dir.name, start_step)

    zt_int = to_torch(data.interior, device)
    zt_ic = to_torch(data.initial, device)
    uv_ic = to_torch(data.initial_uv, device)
    zt_bc = to_torch(data.boundary, device)
    uv_bc = to_torch(data.boundary_uv, device)
    t_lo, t_hi = sol.t_domain

    def compute_losses():
        pde, per_point = pde_residual_loss(model, zt_int, params)
        return {"initial": data_fit_loss(model, zt_ic, uv_ic),
                "boundary": data_fit_loss(model, zt_bc, uv_bc),
                "pde": pde}

    def guard(loss):
        if not torch.isfinite(loss):
            diag = {"step": step, "history_tail": history[-5:],
                    "grad_norm": grad_norm(), "method": method, "seed": seed}
            import json
            (run_dir / "failure_diagnostics.json").write_text(json.dumps(diag, default=str))
            raise NaNLossError(f"non-finite loss at step {step}")

    def grad_norm():
        tot = 0.0
        for p in model.parameters():
            if p.grad is not None:
                tot += float(p.grad.detach().norm() ** 2)
        return float(np.sqrt(tot))

    # ---- Adam phase ----
    for step in range(start_step, tr["adam_steps"]):
        if method == "adaptive_sampling" and step > 0 and \
                step % tr.get("resample_every", 500) == 0:
            new_pts = residual_adaptive_resample(
                model, params, [0.0, t_lo], [sol.z_max, t_hi],
                tr["n_collocation"], seed + step, device)
            zt_int = to_torch(new_pts, device)
        adam.zero_grad(set_to_none=True)
        losses = compute_losses()
        balancer.maybe_update(step, model, losses)
        total = balancer.total(losses)
        guard(total)
        total.backward()
        adam.step()
        sched.step()
        if step % 50 == 0 or step == tr["adam_steps"] - 1:
            history.append({"step": step, "phase": "adam",
                            **{k: float(v.detach()) for k, v in losses.items()},
                            "total": float(total.detach()), "grad_norm": grad_norm(),
                            "param_norm": float(sum(p.detach().norm() ** 2
                                                    for p in model.parameters()) ** 0.5),
                            "weights": dict(balancer.weights) if hasattr(balancer, "weights") else {}})
        if step % tr.get("checkpoint_every", 500) == 0 and step > start_step:
            _save_ckpt(ckpt_path, model, adam, sched, step, history, balancer,
                       prev_elapsed + time.perf_counter() - t_start)

    # ---- L-BFGS phase ----
    if tr.get("lbfgs_steps", 0) > 0 and start_step <= tr["adam_steps"] + tr["lbfgs_steps"]:
        lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=tr["lbfgs_steps"],
                                  history_size=tr.get("lbfgs_history", 50),
                                  tolerance_grad=1e-12, tolerance_change=1e-14,
                                  line_search_fn="strong_wolfe")
        n_evals = [0]

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            losses = compute_losses()
            total = balancer.total(losses)
            if not torch.isfinite(total):
                # L-BFGS line search can probe bad regions; return large finite value
                return torch.tensor(1e10, dtype=torch.float64, device=device, requires_grad=True)
            total.backward()
            n_evals[0] += 1
            if n_evals[0] % 20 == 0:
                history.append({"step": tr["adam_steps"] + n_evals[0], "phase": "lbfgs",
                                **{k: float(v.detach()) for k, v in losses.items()},
                                "total": float(total.detach()), "grad_norm": grad_norm()})
            return total
        try:
            lbfgs.step(closure)
        except NaNLossError:
            log.warning("L-BFGS aborted on non-finite loss; keeping Adam solution")

    elapsed = prev_elapsed + time.perf_counter() - t_start
    step = tr["adam_steps"] + tr.get("lbfgs_steps", 0)
    _save_ckpt(ckpt_path, model, adam, sched, step, history, balancer, elapsed)

    return evaluate_forward(model, sol, data, params, device, history, elapsed, cfg)


def _save_ckpt(path, model, adam, sched, step, history, balancer, elapsed):
    torch.save({"model": model.state_dict(), "adam": adam.state_dict(),
                "sched": sched.state_dict(), "step": step, "history": history,
                "balancer_weights": getattr(balancer, "weights", {}),
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
                "elapsed": elapsed}, path)


@torch.no_grad()
def _predict(model, pts, device):
    u, v = model(to_torch(pts[:, 0:1], device), to_torch(pts[:, 1:2], device))
    return torch.cat([u, v], dim=1).cpu().numpy()


def evaluate_forward(model, sol: BranchSolution, data: ForwardData, params,
                     device, history, elapsed, cfg) -> dict:
    Z, Tm = data.test_grid
    pts = np.stack([Z.ravel(), Tm.ravel()], axis=1)
    t_eval0 = time.perf_counter()
    pred = _predict(model, pts, device)
    infer_time = time.perf_counter() - t_eval0
    field = all_field_metrics(pred, data.test_uv)
    evz = error_vs_z(pred, data.test_uv, Z[:, 0], Z.shape)
    spec = spectral_error(pred, data.test_uv, Z.shape)

    # independent residual points (never used in training)
    zt_res = to_torch(data.residual_test, device)
    _, per_point = pde_residual_loss(model, zt_res, params)
    res = residual_metrics(np.stack([np.sqrt(per_point.squeeze(-1).cpu().numpy()),
                                     np.zeros(len(data.residual_test))], axis=1))

    ic_pred = _predict(model, data.initial, device)
    bc_pred = _predict(model, data.boundary, device)
    ic_err = float(np.sqrt(((ic_pred - data.initial_uv) ** 2).mean()))
    bc_err = float(np.sqrt(((bc_pred - data.boundary_uv) ** 2).mean()))

    mem_mb = None
    if device.startswith("cuda"):
        mem_mb = torch.cuda.max_memory_allocated() / 2**20
    thresh = cfg.get("metrics", {}).get("success_threshold_l2re", 1e-2)
    return {
        **field,
        "spectral_error": spec,
        "error_vs_z": evz,
        "pde_residual_rms_independent": res["residual_rms"],
        "pde_residual_max_independent": res["residual_max"],
        "initial_rmse": ic_err,
        "boundary_rmse": bc_err,
        "train_time_s": elapsed,
        "inference_time_s": infer_time,
        "peak_gpu_mem_mb": mem_mb,
        "n_parameters": count_parameters(model),
        "final_losses": history[-1] if history else {},
        "success": bool(field["l2re_complex"] < thresh),
        "success_threshold": thresh,
        "loss_history": history,
    }
