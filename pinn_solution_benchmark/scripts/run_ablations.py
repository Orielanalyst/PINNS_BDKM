#!/usr/bin/env python
"""Phase 6: matched, closure-respecting physical ablations.

1. Gamma = 0 vs Gamma > 0        (only the constant branch admits both; the
                                  kink/sech closures FORCE Gamma = 0 -> the
                                  incompatibility is documented, not forced)
2. beta4 = 0 vs beta4 != 0       (again only the constant branch: beta4 is a
                                  free parameter there; other branches force 0)
3. conservative vs gain-loss     (NO verified branch admits a conservative
                                  closure: setting ell = 0 collapses every
                                  family to a = 0 or the vacuous case ->
                                  documented as incompatible)
4. fixed vs adaptive weights     (from the pilot matrix, no extra runs)
5. uniform vs residual-adaptive  (from the pilot matrix, no extra runs)

Each matched pair changes EXACTLY ONE closure-consistent quantity.

Usage: python scripts/run_ablations.py [--seeds 0,1,2] [--force]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pinn_benchmark import utils  # noqa: E402
from pinn_benchmark.utils import log, save_json, aggregate_stats  # noqa: E402

BASE_ASSIGN = {"lambda1": 0.2, "lambda2": 0.8, "asq": 1.0, "beta2": 0.2,
               "beta3": 0.2, "beta4": 0.1, "c": 0.5, "alpha2": 0.2, "Gamma": 0.5}

ABLATIONS = {
    "saturation": {"arms": {"Gamma0": {"Gamma": 0.0}, "GammaPos": {"Gamma": 1.0}},
                   "branch": "n0_m0_f0"},
    "fourth_order": {"arms": {"beta4_0": {"beta4": 0.0}, "beta4_pos": {"beta4": 0.4}},
                     "branch": "n0_m0_f0"},
}

INCOMPATIBLE = [
    ("saturation on n1_m0_f0 / n1_m1_f0", "closure forces Gamma = 0: any Gamma > 0 "
     "collapses the family to gamma = alpha2 = 0 and a vacuous equation"),
    ("fourth_order on n1_m0_f0 / n1_m1_f0", "closure forces beta4 = 0"),
    ("conservative (ell = 0, alpha2 = 0)", "no verified branch admits it: ell = 0 "
     "forces c*lambda2^2 = 0 hence a fully vacuous equation; the surviving "
     "solution set is intrinsically dissipative"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    args = ap.parse_args()
    utils.setup_logging()
    seeds = [int(s) for s in args.seeds.split(",")]
    cfg = utils.load_config(args.config)
    cfg["training"] = dict(cfg["training"], adam_steps=800, lbfgs_steps=150)
    device = utils.resolve_device(cfg.get("device", "auto"))

    import torch
    torch.set_num_threads(2)
    from pinn_benchmark.experiment_runner import load_catalog
    from pinn_benchmark.conditioned_pinn import make_rep_from_constraints, task_from_rep
    from pinn_benchmark.trainers import train_forward

    records = {r["branch_id"]: r for r in load_catalog(ROOT)}
    exp_dir = ROOT / "results" / "ablations"
    exp_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for abl_name, spec in ABLATIONS.items():
        rec = records[spec["branch"]]
        for arm_name, override in spec["arms"].items():
            assign = {k: v for k, v in {**BASE_ASSIGN, **override}.items()
                      if k in rec["free_parameters"]}
            rep = make_rep_from_constraints(rec, assign)
            sol = task_from_rep(rec, rep)
            for seed in seeds:
                jid = f"{abl_name}__{arm_name}__seed{seed}"
                rpath = exp_dir / f"{jid}.json"
                if rpath.exists() and not args.force:
                    with open(rpath) as f:
                        rows.append(json.load(f))
                    continue
                log.info("ablation %s ...", jid)
                try:
                    m = train_forward(sol, cfg, "vanilla", seed,
                                      run_dir=exp_dir / jid, device=device)
                    out = {"job": jid, "ablation": abl_name, "arm": arm_name,
                           "seed": seed, "state": "done",
                           "l2re_complex": m["l2re_complex"],
                           "residual_rms": m["pde_residual_rms_independent"],
                           "train_time_s": m["train_time_s"], "success": m["success"]}
                except Exception as e:  # noqa: BLE001
                    out = {"job": jid, "ablation": abl_name, "arm": arm_name,
                           "seed": seed, "state": "failed", "error": str(e)}
                save_json(out, rpath)
                rows.append(out)

    # aggregate + pull method/sampling comparisons from the pilot
    agg = {}
    for r in rows:
        if r.get("state") == "done":
            agg.setdefault(f"{r['ablation']}::{r['arm']}", []).append(r["l2re_complex"])
    summary = {k: aggregate_stats(v) for k, v in agg.items()}

    pilot_agg_path = ROOT / "results" / "forward_pilot" / "aggregate.json"
    pilot_part = {}
    if pilot_agg_path.exists():
        with open(pilot_agg_path) as f:
            pilot_part = json.load(f)
    save_json({"matched_runs": rows, "summary": summary,
               "incompatible": INCOMPATIBLE,
               "method_comparison_from_pilot": pilot_part},
              exp_dir / "ablation_summary.json")

    lines = ["# Physical and methodological ablations", ""]
    lines.append("Matched closure-respecting pairs (constant branch, one free "
                 "parameter changed per pair):")
    lines.append("")
    lines.append("| ablation | arm | mean L2RE | std | n |")
    lines.append("|---|---|---|---|---|")
    for k, st in summary.items():
        lines.append(f"| {k.split('::')[0]} | {k.split('::')[1]} | "
                     f"{st['mean']:.2e} | {st['std']:.1e} | {st['n_finite']} |")
    lines.append("")
    lines.append("## Requested ablations that are incompatible with the verified closures")
    lines.append("")
    for name, why in INCOMPATIBLE:
        lines.append(f"- **{name}**: {why}")
    lines.append("")
    lines.append("Fixed-vs-adaptive weighting and uniform-vs-residual-adaptive "
                 "collocation are covered by the pilot matrix "
                 "(results/forward_pilot/aggregate.json; see final report).")
    (ROOT / "reports" / "ablations.md").write_text("\n".join(lines))
    log.info("wrote reports/ablations.md")


if __name__ == "__main__":
    main()
