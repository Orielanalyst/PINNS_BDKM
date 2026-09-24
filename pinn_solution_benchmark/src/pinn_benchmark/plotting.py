"""Publication-quality figures with consistent labels/colormaps/filenames.

Conventions: intensity |A|^2 -> viridis; absolute error -> magma (log);
residual -> cividis (log); analytic/PINN/error triptychs share color scales.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9, "figure.constrained_layout.use": True,
})


def field_triptych(Z, Tm, true_uv, pred_uv, path: Path, title: str = ""):
    shape = Z.shape
    It = (true_uv[:, 0] ** 2 + true_uv[:, 1] ** 2).reshape(shape)
    Ip = (pred_uv[:, 0] ** 2 + pred_uv[:, 1] ** 2).reshape(shape)
    err = np.hypot(pred_uv[:, 0] - true_uv[:, 0],
                   pred_uv[:, 1] - true_uv[:, 1]).reshape(shape)
    vmax = max(It.max(), Ip.max())
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.0))
    for a, F, ttl, cmap, norm in [
        (ax[0], It, r"analytic $|A|^2$", "viridis", None),
        (ax[1], Ip, r"PINN $|\hat A|^2$", "viridis", None),
        (ax[2], err, r"$|\hat A - A|$", "magma",
         matplotlib.colors.LogNorm(vmin=max(err.min(), 1e-12), vmax=max(err.max(), 1e-11))),
    ]:
        kw = {"vmin": 0, "vmax": vmax} if norm is None else {"norm": norm}
        im = a.pcolormesh(Tm, Z, F, cmap=cmap, shading="auto", **kw)
        a.set_xlabel("t"); a.set_ylabel("z"); a.set_title(ttl)
        fig.colorbar(im, ax=a)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.savefig(path); plt.close(fig)


def components_and_sections(Z, Tm, true_uv, pred_uv, path: Path, title: str = ""):
    shape = Z.shape
    nz = shape[0]
    idxs = [0, nz // 2, nz - 1]
    labels = ["z=0", "z=Z/2", "z=Z"]
    tg = Tm[0]
    fig, ax = plt.subplots(2, 3, figsize=(11, 5.5))
    for j, (i, lab) in enumerate(zip(idxs, labels)):
        tr = true_uv.reshape(nz, -1, 2)[i]
        pr = pred_uv.reshape(nz, -1, 2)[i]
        ax[0, j].plot(tg, tr[:, 0], "k-", lw=1.4, label="Re A (exact)")
        ax[0, j].plot(tg, pr[:, 0], "C0--", label=r"Re $\hat A$")
        ax[0, j].set_title(lab); ax[0, j].set_xlabel("t"); ax[0, j].legend(fontsize=7)
        ax[1, j].plot(tg, tr[:, 1], "k-", lw=1.4, label="Im A (exact)")
        ax[1, j].plot(tg, pr[:, 1], "C3--", label=r"Im $\hat A$")
        ax[1, j].set_xlabel("t"); ax[1, j].legend(fontsize=7)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.savefig(path); plt.close(fig)


def error_vs_z_plot(curves: dict, path: Path, title: str = ""):
    """curves: {label: {"z": [...], "l2re": [...]}}"""
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for lab, c in curves.items():
        ax.semilogy(c["z"], np.maximum(c["l2re"], 1e-16), label=lab)
    ax.set_xlabel("z"); ax.set_ylabel("complex L2RE"); ax.legend(fontsize=7)
    if title:
        ax.set_title(title)
    fig.savefig(path); plt.close(fig)


def loss_curves(history: list, path: Path, title: str = ""):
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    steps = [h["step"] for h in history]
    for key, color in [("initial", "C0"), ("boundary", "C1"), ("pde", "C2"), ("total", "k")]:
        vals = [h.get(key) for h in history]
        if any(v is not None for v in vals):
            ax.semilogy(steps, [max(v, 1e-18) if v is not None else np.nan for v in vals],
                        color=color, label=key, lw=1)
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.legend(fontsize=7)
    if title:
        ax.set_title(title)
    fig.savefig(path); plt.close(fig)


def scatter_difficulty(rows: list[dict], xkey: str, ykey: str, path: Path,
                       xlabel: str = "", ylabel: str = "", logx=True, logy=True):
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    groups = sorted({r.get("branch", "?") for r in rows})
    for gi, g in enumerate(groups):
        xs = [r[xkey] for r in rows if r.get("branch") == g and np.isfinite(r.get(xkey, np.nan))]
        ys = [r[ykey] for r in rows if r.get("branch") == g and np.isfinite(r.get(xkey, np.nan))]
        ax.scatter(xs, ys, s=18, label=g, color=f"C{gi % 10}", alpha=0.75)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel or xkey); ax.set_ylabel(ylabel or ykey)
    ax.legend(fontsize=6)
    fig.savefig(path); plt.close(fig)


def bar_comparison(labels, means, stds, path: Path, ylabel: str, title: str = "", log=True):
    fig, ax = plt.subplots(figsize=(max(4, 0.9 * len(labels)), 3.4))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=3, color=[f"C{i % 10}" for i in range(len(labels))])
    if log:
        ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    fig.savefig(path); plt.close(fig)


def recovered_params_plot(rows: list[dict], param_names: list[str], path: Path, title=""):
    """rows: [{param, noise, rel_error, ...}]"""
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    noises = sorted({r["noise"] for r in rows})
    for pi, pn in enumerate(param_names):
        med = []
        for nv in noises:
            errs = [abs(r["rel_error"]) for r in rows if r["param"] == pn and r["noise"] == nv]
            med.append(np.median(errs) if errs else np.nan)
        ax.semilogy(noises, med, "o-", color=f"C{pi}", label=pn)
    ax.set_xlabel("relative noise level"); ax.set_ylabel("median |rel. error|")
    ax.legend(fontsize=7)
    if title:
        ax.set_title(title)
    fig.savefig(path); plt.close(fig)
