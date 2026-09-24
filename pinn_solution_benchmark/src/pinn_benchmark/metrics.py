"""Evaluation metrics for the complex field and residuals."""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def l2re_complex(pred_uv: np.ndarray, true_uv: np.ndarray) -> float:
    dif = (pred_uv[:, 0] - true_uv[:, 0]) ** 2 + (pred_uv[:, 1] - true_uv[:, 1]) ** 2
    den = true_uv[:, 0] ** 2 + true_uv[:, 1] ** 2
    return float(np.sqrt(dif.sum() / (den.sum() + EPS)))


def component_relative_error(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(((pred - true) ** 2).sum() / ((true**2).sum() + EPS)))


def intensity_error(pred_uv, true_uv) -> float:
    ip = pred_uv[:, 0] ** 2 + pred_uv[:, 1] ** 2
    it = true_uv[:, 0] ** 2 + true_uv[:, 1] ** 2
    return float(np.sqrt(((ip - it) ** 2).sum() / ((it**2).sum() + EPS)))


def all_field_metrics(pred_uv: np.ndarray, true_uv: np.ndarray) -> dict:
    err = np.hypot(pred_uv[:, 0] - true_uv[:, 0], pred_uv[:, 1] - true_uv[:, 1])
    return {
        "l2re_complex": l2re_complex(pred_uv, true_uv),
        "rel_err_real": component_relative_error(pred_uv[:, 0], true_uv[:, 0]),
        "rel_err_imag": component_relative_error(pred_uv[:, 1], true_uv[:, 1]),
        "intensity_error": intensity_error(pred_uv, true_uv),
        "max_pointwise_error": float(err.max()),
        "mse": float((err**2).mean()),
    }


def error_vs_z(pred_uv, true_uv, z_values, grid_shape) -> dict:
    """Per-z-slice complex L2RE on a (nz, nt) grid flattened in C order."""
    nz, nt = grid_shape
    p = pred_uv.reshape(nz, nt, 2)
    tr = true_uv.reshape(nz, nt, 2)
    out = []
    for i in range(nz):
        out.append(l2re_complex(p[i], tr[i]))
    return {"z": list(map(float, z_values)), "l2re": out}


def spectral_error(pred_uv, true_uv, grid_shape) -> float:
    """Relative L2 error of the t-spectrum, averaged over z-slices."""
    nz, nt = grid_shape
    p = (pred_uv[:, 0] + 1j * pred_uv[:, 1]).reshape(nz, nt)
    tr = (true_uv[:, 0] + 1j * true_uv[:, 1]).reshape(nz, nt)
    Fp, Ft = np.fft.fft(p, axis=1), np.fft.fft(tr, axis=1)
    return float(np.sqrt((np.abs(Fp - Ft) ** 2).sum() / ((np.abs(Ft) ** 2).sum() + EPS)))


def residual_metrics(res_uv: np.ndarray) -> dict:
    mag = np.hypot(res_uv[:, 0], res_uv[:, 1])
    return {"residual_rms": float(np.sqrt((mag**2).mean())),
            "residual_max": float(mag.max())}
