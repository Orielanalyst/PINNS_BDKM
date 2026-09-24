#!/usr/bin/env python
"""Generate the cross-experiment summary figures for the final report:

- branch-wise error distributions (all seeds, all methods)
- method comparison with bootstrap 95% CIs per branch
- error vs curvature / fourth-derivative / localization width
- error vs z overlays for median seeds
- forward success-rate matrix
- inverse recovery vs noise AND vs observation density
- conditioned-PINN zero-shot / fine-tune / scratch comparison
- stress vs main group comparison

Idempotent: reads everything from results/, writes reports/figures/summary/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pinn_benchmark.utils import bootstrap_ci, setup_logging, log  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 9,
    "axes.titlesize": 9.5, "axes.labelsize": 9,
    "figure.constrained_layout.use": True, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5,
})

FIG = ROOT / "reports" / "figures" / "summary"
FIG.mkdir(parents=True, exist_ok=True)

METHOD_LABEL = {"vanilla": "vanilla", "adaptive_weights": "adaptive weights",
                "adaptive_sampling": "adaptive sampling"}
BRANCH_LABEL = {"(0,0)": "constant (0,0)", "(1,0)": "bright sech (1,0)",
                "(1,1)": "dark kink (1,1)"}


def load_runs(exp: str) -> list[dict]:
    d = ROOT / "results" / exp
    if not d.exists():
        return []
    out = []
    for f in sorted(d.glob("*/result.json")):
        with open(f) as fh:
            out.append(json.load(fh))
    return out


def catalog() -> dict:
    with open(ROOT / "data/verified_branches/branch_catalog.json") as f:
        return {r["branch_id"]: r for r in json.load(f)}


def fig_branch_distributions(pilot, hifi):
    done = [r for r in pilot if r.get("state") == "done"]
    if not done:
        return
    branches = sorted({r["branch"] for r in done})
    methods = ["vanilla", "adaptive_weights", "adaptive_sampling"]
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    xpos, xticklab = [], []
    x = 0
    for b in branches:
        for m in methods:
            vals = [r["l2re_complex"] for r in done
                    if r["branch"] == b and r["method"] == m]
            if not vals:
                continue
            jitter = (np.random.default_rng(1).random(len(vals)) - 0.5) * 0.25
            ax.scatter(np.full(len(vals), x) + jitter, vals, s=14,
                       color=f"C{methods.index(m)}", alpha=0.75, zorder=3)
            ax.plot([x - 0.2, x + 0.2], [np.median(vals)] * 2, "k-", lw=1.6, zorder=4)
            xpos.append(x)
            xticklab.append(f"{BRANCH_LABEL.get(b, b).split()[0]}\n{METHOD_LABEL[m].replace(' ', chr(10))}")
            x += 1
        x += 0.6
    # hi-fi sech overlay
    hin = [r for r in hifi if r.get("state") == "done"]
    if hin:
        vals = [r["l2re_complex"] for r in hin]
        ax.scatter(np.full(len(vals), x), vals, s=16, color="C3", marker="D",
                   alpha=0.8, zorder=3, label="sech, high-fidelity budget")
        ax.plot([x - 0.2, x + 0.2], [np.median(vals)] * 2, "k-", lw=1.6, zorder=4)
        xpos.append(x)
        xticklab.append("sech\nhi-fi")
    ax.axhline(1e-2, color="0.4", ls="--", lw=1, label="success threshold $10^{-2}$")
    ax.set_yscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels(xticklab, fontsize=6.5)
    ax.set_ylabel("complex L2RE")
    ax.set_title("Forward PINN error distributions — every seed shown, no cherry-picking")
    ax.legend(fontsize=7, loc="upper left")
    fig.savefig(FIG / "forward_error_distributions.png")
    plt.close(fig)


def fig_method_ci(pilot):
    done = [r for r in pilot if r.get("state") == "done"]
    if not done:
        return
    branches = sorted({r["branch"] for r in done})
    methods = ["vanilla", "adaptive_weights", "adaptive_sampling"]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    w = 0.25
    for mi, m in enumerate(methods):
        means, los, his = [], [], []
        for b in branches:
            vals = [r["l2re_complex"] for r in done if r["branch"] == b and r["method"] == m]
            means.append(np.mean(vals) if vals else np.nan)
            lo, hi = bootstrap_ci(vals) if vals else (np.nan, np.nan)
            los.append(lo)
            his.append(hi)
        xs = np.arange(len(branches)) + (mi - 1) * w
        means = np.array(means)
        yerr = np.abs(np.array([means - los, np.array(his) - means]))
        ax.bar(xs, means, width=w, yerr=yerr, capsize=3,
               label=METHOD_LABEL[m], color=f"C{mi}")
    ax.axhline(1e-2, color="0.4", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(branches)))
    ax.set_xticklabels([BRANCH_LABEL.get(b, b) for b in branches], fontsize=8)
    ax.set_ylabel("complex L2RE (mean, bootstrap 95% CI)")
    ax.set_title("Method comparison per branch (5 seeds x 2 parameter sets)")
    ax.legend(fontsize=7)
    fig.savefig(FIG / "method_comparison_ci.png")
    plt.close(fig)


def fig_difficulty(pilot, stress, cat):
    rows = []
    for r in pilot + stress:
        if r.get("state") != "done":
            continue
        rec = cat.get(r["branch_id"], {})
        p = rec.get("properties", {})
        rows.append({"branch": r["branch"], "group": rec.get("group", "?"),
                     "l2re": r["l2re_complex"],
                     "c2": float(p.get("feat_max_curvature_tt", np.nan)),
                     "c4": float(p.get("feat_max_fourth_deriv_tttt", np.nan)),
                     "amp": float(p.get("feat_max_amplitude", np.nan))})
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for ax, key, lab in [(axes[0], "c2", r"max $|A_{tt}|$"),
                         (axes[1], "c4", r"max $|A_{tttt}|$"),
                         (axes[2], "amp", r"max $|A|$")]:
        for gi, grp in enumerate(["main", "stress"]):
            xs = [r[key] for r in rows if r["group"] == grp and np.isfinite(r[key]) and r[key] > 0]
            ys = [r["l2re"] for r in rows if r["group"] == grp and np.isfinite(r[key]) and r[key] > 0]
            if xs:
                ax.scatter(xs, ys, s=16, alpha=0.7, label=grp, color=["C0", "C3"][gi])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(lab)
        ax.set_ylabel("complex L2RE")
        ax.legend(fontsize=7)
    fig.suptitle("Error vs profile-shape features (all runs)", fontsize=10)
    fig.savefig(FIG / "difficulty_features.png")
    plt.close(fig)


def fig_error_vs_z(pilot):
    done = [r for r in pilot if r.get("state") == "done" and "error_vs_z" in r]
    if not done:
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    cells = {}
    for r in done:
        cells.setdefault((r["branch"], r["method"]), []).append(r)
    for (b, m), rs in sorted(cells.items()):
        if m != "vanilla":
            continue
        med = sorted(rs, key=lambda r: r["l2re_complex"])[len(rs) // 2]
        c = med["error_vs_z"]
        ax.semilogy(c["z"], np.maximum(c["l2re"], 1e-12),
                    label=BRANCH_LABEL.get(b, b))
    ax.set_xlabel("propagation distance z")
    ax.set_ylabel("complex L2RE per z-slice")
    ax.set_title("Error growth along propagation (vanilla, median seed)")
    ax.legend(fontsize=7)
    fig.savefig(FIG / "error_vs_z_branches.png")
    plt.close(fig)


def fig_inverse(rows):
    if not rows:
        return
    # vs noise (d = 0.05) and vs density (gainloss)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axes[0]
    params = sorted({r["param"] for r in rows})
    for pi, pn in enumerate(params):
        noises = sorted({r["noise"] for r in rows if r["param"] == pn and r["density"] == 0.05})
        med = []
        for nv in noises:
            v = [r["rel_error"] if (r["rel_error"] is not None and np.isfinite(r["rel_error"]))
                 else r["abs_error"] for r in rows
                 if r["param"] == pn and r["noise"] == nv and r["density"] == 0.05]
            med.append(np.median(v) if v else np.nan)
        if noises:
            ax.semilogy(noises, med, "o-", label=pn, color=f"C{pi}")
    ax.set_xlabel("relative noise level")
    ax.set_ylabel("median |relative error| (abs. if true = 0)")
    ax.set_title("Recovery vs noise (5% observation density)")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    for pi, pn in enumerate(["c", "ell"]):
        dens = sorted({r["density"] for r in rows if r["param"] == pn})
        med = []
        for dv in dens:
            v = [r["rel_error"] for r in rows
                 if r["param"] == pn and r["density"] == dv
                 and r["rel_error"] is not None and np.isfinite(r["rel_error"])]
            med.append(np.median(v) if v else np.nan)
        if dens:
            ax.loglog(dens, med, "s-", label=pn, color=f"C{pi}")
    ax.set_xlabel("observation density (fraction of grid)")
    ax.set_ylabel("median |relative error|")
    ax.set_title("Recovery vs observation density (gain/loss stage)")
    ax.legend(fontsize=8)
    fig.savefig(FIG / "inverse_recovery.png")
    plt.close(fig)


def fig_generalization(gen):
    if not gen:
        return
    branches = ["n0_m0_f0", "n1_m0_f0", "n1_m1_f0"]
    labels = ["constant", "sech", "kink"]
    modes = [("zero_shot", "zero-shot"), ("fine_tune_0.01", "fine-tune 1%"),
             ("fine_tune_0.05", "fine-tune 5%"), ("scratch_same_budget", "scratch (same budget)")]
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    w = 0.2
    for mi, (key, lab) in enumerate(modes):
        means, spreads = [], []
        for b in branches:
            vals = []
            for res in gen:
                e = res.get("lobo", {}).get(b, {}).get(key, {})
                vals += [v for v in e.values()] if isinstance(e, dict) else []
            means.append(np.mean(vals) if vals else np.nan)
            spreads.append(np.std(vals) if len(vals) > 1 else 0)
        xs = np.arange(len(branches)) + (mi - 1.5) * w
        ax.bar(xs, means, width=w, yerr=spreads, capsize=2, label=lab, color=f"C{mi}")
    ax.axhline(1e-2, color="0.4", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(branches)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("complex L2RE on held-out branch")
    ax.set_title("Conditioned PINN: leave-one-branch-out transfer")
    ax.legend(fontsize=7)
    fig.savefig(FIG / "generalization_lobo.png")
    plt.close(fig)

    # interpolation / extrapolation
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    for mi, key in enumerate(["interp", "extrap"]):
        means = []
        for b in branches:
            vals = [res[key][b] for res in gen
                    if key in res and isinstance(res[key].get(b), (int, float))]
            means.append(np.mean(vals) if vals else np.nan)
        ax.bar(np.arange(len(branches)) + (mi - 0.5) * 0.32, means, width=0.32,
               label={"interp": "interpolation (new in-range params)",
                      "extrap": "extrapolation (out-of-range params)"}[key],
               color=["C0", "C3"][mi])
    ax.axhline(1e-2, color="0.4", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(branches)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("complex L2RE")
    ax.set_title("Conditioned PINN: closure-consistent parameter transfer")
    ax.legend(fontsize=7)
    fig.savefig(FIG / "generalization_interp_extrap.png")
    plt.close(fig)


def fig_success_matrix(pilot, stress, hifi):
    rows = {}
    for r in pilot + stress + hifi:
        if r.get("state") != "done":
            continue
        exp = r.get("experiment", "?").replace("forward_", "")
        key = (f"{r['branch']} [{exp}]", r["method"])
        rows.setdefault(key, []).append(r.get("success", False))
    if not rows:
        return
    labels = sorted({k[0] for k in rows})
    methods = ["vanilla", "adaptive_weights", "adaptive_sampling"]
    M = np.full((len(labels), len(methods)), np.nan)
    for (bl, m), v in rows.items():
        if m in methods:
            M[labels.index(bl), methods.index(m)] = np.mean(v)
    fig, ax = plt.subplots(figsize=(5.4, 0.5 + 0.42 * len(labels)))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(len(labels)):
        for j in range(len(methods)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([METHOD_LABEL[m] for m in methods], fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Success rate across seeds (L2RE < 1e-2)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(FIG / "success_matrix.png")
    plt.close(fig)


def main():
    setup_logging()
    cat = catalog()
    pilot = load_runs("forward_pilot")
    hifi = load_runs("forward_sech_hifi")
    stress = load_runs("forward_stress")
    inv_rows = []
    p = ROOT / "results/inverse_pilot/param_rows.json"
    if p.exists():
        with open(p) as f:
            inv_rows = json.load(f)
    gen = []
    g = ROOT / "results/generalization_pilot/all_results.json"
    if g.exists():
        with open(g) as f:
            gen = json.load(f)

    fig_branch_distributions(pilot, hifi)
    fig_method_ci(pilot)
    fig_difficulty(pilot, stress, cat)
    fig_error_vs_z(pilot)
    fig_inverse(inv_rows)
    fig_generalization(gen)
    fig_success_matrix(pilot, stress, hifi)
    log.info("summary figures -> %s", FIG)


if __name__ == "__main__":
    main()
