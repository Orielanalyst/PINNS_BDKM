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


def statistical_comparisons(summary_rows: list[dict]) -> dict:
    """Report paired-seed arm comparisons without hiding small-sample limits."""
    from scipy import stats

    arms = {}
    for row in summary_rows:
        if row.get("state") == "done":
            arms.setdefault(row["ablation"], {}).setdefault(row["arm"], []).append(
                float(row["l2re_complex"]))

    comparisons = {}
    for ablation, values in arms.items():
        arm_names = sorted(values)
        if len(arm_names) != 2:
            continue
        left, right = (values[name] for name in arm_names)
        shapiro = {
            name: {"statistic": float(stats.shapiro(values[name]).statistic),
                   "p_value": float(stats.shapiro(values[name]).pvalue),
                   "n": len(values[name])}
            for name in arm_names
        }
        welch = stats.ttest_ind(left, right, equal_var=False)
        mann = stats.mannwhitneyu(left, right, alternative="two-sided", method="exact")
        comparisons[ablation] = {
            "arms": arm_names,
            "n_per_arm": [len(left), len(right)],
            "normality": {"test": "Shapiro-Wilk", "results": shapiro,
                           "interpretation": "low power at n=3; diagnostic only"},
            "welch_t_test": {"statistic": float(welch.statistic),
                              "p_value": float(welch.pvalue)},
            "mann_whitney_u": {"statistic": float(mann.statistic),
                                "p_value": float(mann.pvalue)},
        }
    return comparisons


def minimum_detectable_effect(n_per_arm: int, alpha: float = 0.05,
                              target_power: float = 0.80) -> dict:
    """Normal-approximation standardized MDE for a two-arm comparison."""
    from scipy.stats import norm

    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(target_power)
    d = (z_alpha + z_power) * (2.0 / n_per_arm) ** 0.5
    return {"n_per_arm": n_per_arm, "alpha": alpha,
            "target_power": target_power, "standardized_mde": float(d),
            "method": "two-sided normal approximation"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="analyze completed JSON results without importing PyTorch")
    ap.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    args = ap.parse_args()
    utils.setup_logging()
    seeds = [int(s) for s in args.seeds.split(",")]
    cfg = utils.load_config(args.config)
    cfg["training"] = dict(cfg["training"], adam_steps=800, lbfgs_steps=150)
    exp_dir = ROOT / "results" / "ablations"
    exp_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    if args.report_only:
        for rpath in sorted(exp_dir.glob("*.json")):
            if rpath.name == "ablation_summary.json":
                continue
            with open(rpath) as f:
                rows.append(json.load(f))
    else:
        device = utils.resolve_device(cfg.get("device", "auto"))
        import torch
        torch.set_num_threads(2)
        from pinn_benchmark.experiment_runner import load_catalog
        from pinn_benchmark.conditioned_pinn import make_rep_from_constraints, task_from_rep
        from pinn_benchmark.trainers import train_forward

        records = {r["branch_id"]: r for r in load_catalog(ROOT)}
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
    comparisons = statistical_comparisons(rows)
    power = minimum_detectable_effect(3)

    pilot_agg_path = ROOT / "results" / "forward_pilot" / "aggregate.json"
    pilot_part = {}
    if pilot_agg_path.exists():
        with open(pilot_agg_path) as f:
            pilot_part = json.load(f)
    save_json({"matched_runs": rows, "summary": summary,
               "statistical_comparisons": comparisons,
               "power_analysis": power,
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
    lines.extend(["", "## Statistical comparisons", "",
                  "Welch two-sample t-tests and exact two-sided Mann-Whitney U "
                  "tests are reported as sensitivity analyses. Shapiro-Wilk "
                  "values are diagnostics only because n=3 per arm.", ""])
    for name, result in comparisons.items():
        lines.append(f"- **{name}** ({' vs '.join(result['arms'])}): "
                     f"Welch p={result['welch_t_test']['p_value']:.4g}; "
                     f"Mann-Whitney p={result['mann_whitney_u']['p_value']:.4g}.")
    lines.extend(["", "## Power analysis", "",
                  "At n=3 per arm, alpha=0.05, and target power=0.80, the "
                  f"two-sided normal-approximation standardized minimum "
                  f"detectable effect is d={power['standardized_mde']:.3f}.",
                  "This is a planning calculation, not evidence that the "
                  "observed ablation has that effect size."])
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
