"""Loss terms and adaptive loss balancing for the forward/inverse PINN.

Loss components are always recorded separately (initial, boundary, PDE,
observation); training loss is never the sole correctness measure.

Adaptive weighting: gradient-norm balancing (Wang, Teng & Perdikaris 2021,
"Understanding and mitigating gradient flow pathologies in PINNs"): weights
of the data terms are updated so each term's gradient magnitude matches the
PDE term's, via an exponential moving average.
"""
from __future__ import annotations

import torch

from .equations import torch_residual


def pde_residual_loss(model, zt, params, eta=None):
    z = zt[:, 0:1].clone().requires_grad_(True)
    t = zt[:, 1:2].clone().requires_grad_(True)
    u, v = model(z, t, eta)
    Fu, Fv = torch_residual(u, v, z, t, params)
    per_point = Fu**2 + Fv**2
    return per_point.mean(), per_point.detach()


def data_fit_loss(model, zt, uv_target, eta=None):
    u, v = model(zt[:, 0:1], zt[:, 1:2], eta)
    return ((u - uv_target[:, 0:1]) ** 2 + (v - uv_target[:, 1:2]) ** 2).mean()


class GradNormBalancer:
    """lambda_k <- (1-alpha)*lambda_k + alpha * max_grad(PDE)/mean_grad(term_k)."""

    def __init__(self, term_names, alpha: float = 0.1, update_every: int = 100):
        self.weights = {k: 1.0 for k in term_names}
        self.alpha = alpha
        self.update_every = update_every

    def maybe_update(self, step, model, losses: dict, pde_key: str = "pde"):
        if step % self.update_every != 0 or step == 0:
            return
        grads = {}
        for k, loss in losses.items():
            if not loss.requires_grad:
                continue
            g = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad],
                                    retain_graph=True, allow_unused=True)
            flat = torch.cat([gi.reshape(-1) for gi in g if gi is not None])
            grads[k] = flat.abs()
        if pde_key not in grads:
            return
        ref = grads[pde_key].max()
        for k in self.weights:
            if k == pde_key or k not in grads:
                continue
            mean_k = grads[k].mean()
            if mean_k > 0 and torch.isfinite(mean_k) and torch.isfinite(ref):
                target = (ref / mean_k).item()
                self.weights[k] = (1 - self.alpha) * self.weights[k] + self.alpha * target

    def total(self, losses: dict, pde_key: str = "pde"):
        tot = losses[pde_key]
        for k, w in self.weights.items():
            if k != pde_key and k in losses:
                tot = tot + w * losses[k]
        return tot


class FixedWeights:
    def __init__(self, weights: dict):
        self.weights = dict(weights)

    def maybe_update(self, *a, **k):
        pass

    def total(self, losses: dict, pde_key: str = "pde"):
        tot = self.weights.get(pde_key, 1.0) * losses[pde_key]
        for k, loss in losses.items():
            if k != pde_key:
                tot = tot + self.weights.get(k, 1.0) * loss
        return tot
