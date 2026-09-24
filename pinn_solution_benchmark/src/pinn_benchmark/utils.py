"""Shared utilities: config loading, seeding, device resolution, IO, logging."""
from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

log = logging.getLogger("pinn_benchmark")


def setup_logging(level: str = "INFO") -> None:
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S"))
        log.addHandler(h)
    log.setLevel(getattr(logging, level.upper()))


def load_config(path: str | Path) -> dict:
    """Load a YAML config; supports a single-level `inherit: other.yaml` key."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "inherit" in cfg:
        base = load_config(path.parent / cfg.pop("inherit"))
        base = deep_update(base, cfg)
        cfg = base
    return cfg


def deep_update(base: dict, upd: dict) -> dict:
    out = dict(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def set_seed(seed: int) -> None:
    """Deterministic seeding for python, numpy and (if available) torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(device: str = "auto") -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def environment_info() -> dict:
    """Hardware + package versions, saved with every run for reproducibility."""
    info = {
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for pkg in ("numpy", "scipy", "sympy", "mpmath", "torch", "matplotlib", "pandas"):
        try:
            info[pkg] = __import__(pkg).__version__
        except ImportError:
            info[pkg] = "not installed"
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def save_json(obj: Any, path: str | Path, overwrite: bool = True) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, complex):
        return {"re": o.real, "im": o.imag}
    if isinstance(o, Path):
        return str(o)
    return str(o)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Bootstrap CI for the mean. Returns (lo, hi)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def aggregate_stats(values) -> dict:
    v = np.asarray(values, dtype=float)
    finite = v[np.isfinite(v)]
    if len(finite) == 0:
        return {"n": 0, "n_finite": 0, "mean": None, "std": None, "median": None,
                "iqr": None, "min": None, "max": None, "ci95": [None, None]}
    lo, hi = bootstrap_ci(finite)
    return {
        "n": int(len(v)),
        "n_finite": int(len(finite)),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
        "median": float(np.median(finite)),
        "iqr": float(np.quantile(finite, 0.75) - np.quantile(finite, 0.25)),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "ci95": [lo, hi],
    }


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.elapsed = time.perf_counter() - self.t0
