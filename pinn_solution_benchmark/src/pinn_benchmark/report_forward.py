"""Figures + difficulty analysis for forward experiments."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .datasets import branch_solution_from_record, make_forward_data, to_torch
from .experiment_runner import load_catalog
from .utils import log, save_json


def write_forward_figures(root: Path, cfg: dict, results: list[dict]) -> None:
    from . import plotting

    exp = cfg.get("experiment_name", "forward")
    fig_dir = root / "reports" / "figures" / exp
    fig_dir.mkdir(parents=True, exist_ok=True)
    records = {r["branch_id"]: r for r in load_catalog(root)}
    done = [r for r in results if r.get("state") == "done"]

    # field plots: best seed (median l2re) per (branch, rep, method)
    cells = {}
    for r in done:
        cells.setdefault((r["branch_id"], r["rep"], r["method"]), []).append(r)
    for (bid, rep, method), rows in cells.items():
        rows.sort(key=lambda r: r["l2re_complex"])
        best = rows[len(rows) // 2]  # median seed, not the prettiest
        try:
            _replot_field(root, cfg, records[bid], best, fig_dir)
        except Exception as e:  # noqa: BLE001
            log.warning("field plot failed for %s: %s", best["job"], e)

    # error-vs-z overlay per method
    curves = {}
    for (bid, rep, method), rows in cells.items():
        med = sorted(rows, key=lambda r: r["l2re_complex"])[len(rows) // 2]
        curves[f"{bid}-r{rep}-{method}"] = med["error_vs_z"]
    if curves:
        plotting.error_vs_z_plot(curves, fig_dir / "error_vs_z_all.png",
                                 title=f"{exp}: complex L2RE vs z (median seeds)")

    # branch-difficulty scatter: error vs curvature / 4th derivative features
    rows = []
    for r in done:
        rec = records.get(r["branch_id"], {})
        props = rec.get("properties", {})
        rows.append({
            "branch": r["branch"], "l2re": r["l2re_complex"],
            "max_curvature": float(props.get("feat_max_curvature_tt", np.nan)),
            "max_fourth": float(props.get("feat_max_fourth_deriv_tttt", np.nan)),
            "width": float(props.get("feat_localization_width_t", np.nan)),
        })
    if rows:
        plotting.scatter_difficulty(rows, "max_curvature", "l2re",
                                    fig_dir / "difficulty_error_vs_curvature.png",
                                    xlabel=r"max $|A_{tt}|$", ylabel="complex L2RE")
        plotting.scatter_difficulty(rows, "max_fourth", "l2re",
                                    fig_dir / "difficulty_error_vs_fourth_deriv.png",
                                    xlabel=r"max $|A_{tttt}|$", ylabel="complex L2RE")
        save_json(rows, fig_dir / "difficulty_rows.json")

    # method comparison bars
    methods = sorted({r["method"] for r in done})
    labels, means, stds = [], [], []
    for m in methods:
        vals = [r["l2re_complex"] for r in done if r["method"] == m]
        if vals:
            labels.append(m)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
    if labels:
        plotting.bar_comparison(labels, means, stds, fig_dir / "method_comparison.png",
                                ylabel="complex L2RE (mean over runs)")
    log.info("forward figures written to %s", fig_dir)


def _replot_field(root, cfg, rec, result, fig_dir):
    from . import plotting
    from .trainers import build_model

    run_dir = root / "results" / result["experiment"] / result["job"]
    ckpt = run_dir / "checkpoint.pt"
    if not ckpt.exists():
        return
    sol = branch_solution_from_record(rec, result["rep"],
                                      z_max=cfg.get("domain", {}).get("z_max", 1.0))
    model = build_model(cfg, sol, "cpu")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    data = make_forward_data(sol, 16, 64,
                             64, result["seed"],
                             test_shape=tuple(cfg.get("metrics", {}).get("test_grid", [64, 128])))
    Z, Tm = data.test_grid
    pts = np.stack([Z.ravel(), Tm.ravel()], axis=1)
    with torch.no_grad():
        u, v = model(to_torch(pts[:, 0:1], "cpu"), to_torch(pts[:, 1:2], "cpu"))
        pred = torch.cat([u, v], dim=1).numpy()
    ttl = f"{result['branch']} rep{result['rep']} {result['method']} seed{result['seed']}"
    plotting.field_triptych(Z, Tm, data.test_uv, pred,
                            fig_dir / f"field_{result['job']}.png", title=ttl)
    plotting.components_and_sections(Z, Tm, data.test_uv, pred,
                                     fig_dir / f"sections_{result['job']}.png", title=ttl)
    plotting.loss_curves(result.get("loss_history", []),
                         fig_dir / f"losses_{result['job']}.png", title=ttl)
