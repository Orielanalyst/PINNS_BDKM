#!/usr/bin/env python
"""Phase 7: identifiability analysis + staged inverse recovery (resumable).

Usage:
    python scripts/run_inverse.py --config configs/inverse_pilot.yaml
        [--stages gainloss,nonlinear] [--densities 0.05] [--noises 0.0,0.01]
        [--seeds 0,1,2] [--force]
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pinn_benchmark import utils  # noqa: E402
from pinn_benchmark.utils import log, save_json  # noqa: E402


def _worker(task):
    cfg, rec, stage, density, noise, seed, device, force = task
    import torch
    torch.set_num_threads(1)
    from pinn_benchmark.datasets import branch_solution_from_record
    from pinn_benchmark.inverse_problem import run_inverse
    jid = f"{rec['branch_id']}__{stage}__d{density}__n{noise}__seed{seed}"
    out_path = ROOT / "results" / cfg["experiment_name"] / f"{jid}.json"
    if out_path.exists() and not force:
        with open(out_path) as f:
            return json.load(f)
    sol = branch_solution_from_record(rec, 0)
    try:
        res = run_inverse(sol, stage, density, noise, seed, cfg, device)
    except Exception as e:  # noqa: BLE001
        import traceback
        res = {"state": "failed", "error": str(e), "traceback": traceback.format_exc()[-2000:],
               "stage": stage, "density": density, "noise": noise, "seed": seed}
    res["branch_id"] = rec["branch_id"]
    res["job"] = jid
    res.pop("history", None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(res, out_path)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "inverse_pilot.yaml"))
    ap.add_argument("--stages", default="")
    ap.add_argument("--densities", default="")
    ap.add_argument("--noises", default="")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--branches", default="")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-identifiability", action="store_true")
    args = ap.parse_args()
    utils.setup_logging()
    cfg = utils.load_config(args.config)
    for key, arg, cast in [("stages", args.stages, str), ("densities", args.densities, float),
                           ("noises", args.noises, float), ("seeds", args.seeds, int),
                           ("branches", args.branches, str)]:
        if arg:
            cfg[key] = [cast(x) for x in arg.split(",")]
    device = utils.resolve_device(cfg.get("device", "auto"))
    workers = args.workers or cfg.get("workers", 1)

    from pinn_benchmark.experiment_runner import load_catalog
    records = {r["branch_id"]: r for r in load_catalog(ROOT)}
    chosen = [records[b] for b in cfg["branches"]]
    exp_dir = ROOT / "results" / cfg["experiment_name"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_json(utils.environment_info(), exp_dir / "environment.json")

    # --- identifiability first ---
    if not args.skip_identifiability:
        from pinn_benchmark.datasets import branch_solution_from_record
        from pinn_benchmark.inverse_problem import sensitivity_analysis
        ident = {}
        for rec in chosen:
            sol = branch_solution_from_record(rec, 0)
            ident[rec["branch_id"]] = sensitivity_analysis(sol)
            log.info("identifiability %s: reduced rank=%d cond=%.2e | (g0,T2,alpha) rank=%d",
                     rec["branch_id"], ident[rec["branch_id"]]["reduced_rank_1e-8"],
                     ident[rec["branch_id"]]["reduced_condition_number"],
                     ident[rec["branch_id"]]["g0T2alpha_rank_1e-8"])
        save_json(ident, exp_dir / "identifiability.json")

    tasks = [(cfg, rec, st, d, nv, sd, device, args.force)
             for rec in chosen
             for st in cfg["stages"]
             for d in cfg["densities"]
             for nv in cfg["noises"]
             for sd in cfg["seeds"]]
    log.info("inverse experiment: %d runs (%d workers)", len(tasks), workers)
    if workers <= 1:
        results = [_worker(t) for t in tasks]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            results = pool.map(_worker, tasks)

    # aggregate over EVERY completed run on disk, not just this invocation's
    # subset (restricted invocations would otherwise clobber the aggregate)
    special = {"all_results.json", "aggregate.json", "param_rows.json",
               "identifiability.json", "environment.json"}
    all_disk = []
    for f in sorted(exp_dir.glob("*.json")):
        if f.name in special:
            continue
        with open(f) as fh:
            all_disk.append(json.load(fh))
    save_json(all_disk, exp_dir / "all_results.json")
    _aggregate_and_plot(exp_dir, all_disk)


def _aggregate_and_plot(exp_dir: Path, results: list):
    import numpy as np
    from pinn_benchmark.utils import aggregate_stats
    from pinn_benchmark import plotting

    done = [r for r in results if r.get("state") == "done"]
    rows = []
    for r in done:
        for p in r["params"]:
            rows.append({"branch": r["branch_id"], "stage": r["stage"],
                         "param": p["param"], "noise": r["noise"],
                         "density": r["density"], "seed": r["seed"],
                         "true": p["true"], "recovered": p["recovered"],
                         "rel_error": p["rel_error"] if p["rel_error"] is not None else np.nan,
                         "abs_error": p["abs_error"], "l2re_field": r["l2re_field"]})
    save_json(rows, exp_dir / "param_rows.json")
    agg = {}
    for r in rows:
        key = f"{r['branch']}|{r['stage']}|{r['param']}|d{r['density']}|n{r['noise']}"
        agg.setdefault(key, []).append(r["rel_error"] if np.isfinite(r["rel_error"])
                                       else r["abs_error"])
    save_json({k: aggregate_stats(v) for k, v in agg.items()}, exp_dir / "aggregate.json")
    fig_dir = exp_dir.parents[1] / "reports" / "figures" / exp_dir.name
    fig_dir.mkdir(parents=True, exist_ok=True)
    for stage in sorted({r["stage"] for r in rows}):
        srows = [r for r in rows if r["stage"] == stage and np.isfinite(r["rel_error"])]
        if srows:
            pnames = sorted({r["param"] for r in srows})
            plotting.recovered_params_plot(srows, pnames,
                                           fig_dir / f"recovery_vs_noise_{stage}.png",
                                           title=f"stage: {stage}")
    log.info("inverse aggregate + figures written")


if __name__ == "__main__":
    main()
