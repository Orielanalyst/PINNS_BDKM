"""Residual-adaptive collocation resampling (RAD; Wu et al. 2023,
"A comprehensive study of non-adaptive and residual-based adaptive sampling
for PINNs"): sample new collocation points with probability

    p(x) ~ eps(x)^k / mean(eps^k) + c,

where eps is the pointwise PDE residual on a dense candidate pool.
"""
from __future__ import annotations

import numpy as np
import torch

from .datasets import latin_hypercube, to_torch
from .losses import pde_residual_loss


def residual_adaptive_resample(model, params, lo, hi, n_points: int, seed: int,
                               device, k: float = 1.0, c: float = 1.0,
                               pool_factor: int = 4, eta=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pool = latin_hypercube(pool_factor * n_points, lo, hi, rng)
    zt = to_torch(pool, device)
    with torch.enable_grad():
        _, per_point = pde_residual_loss(model, zt, params, eta)
    eps = per_point.squeeze(-1).cpu().numpy()
    eps = np.nan_to_num(eps, nan=np.nanmax(eps[np.isfinite(eps)]) if np.isfinite(eps).any() else 1.0)
    w = eps**k
    mean_w = w.mean() if w.mean() > 0 else 1.0
    p = w / mean_w + c
    p = p / p.sum()
    idx = rng.choice(len(pool), size=n_points, replace=False, p=p)
    return pool[idx]
