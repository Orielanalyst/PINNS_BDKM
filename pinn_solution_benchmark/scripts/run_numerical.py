#!/usr/bin/env python
"""Phase 3: independent numerical solver validation + convergence study.

Usage:
    python scripts/run_numerical.py [--branches n1_m1_f0,n1_m0_f0] [--quick]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pinn_benchmark import utils  # noqa: E402
from pinn_benchmark.utils import log, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branches", default="", help="comma list of branch_ids (default: all main)")
    ap.add_argument("--quick", action="store_true", help="smoke: coarse resolutions only")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    utils.setup_logging()

    from pinn_benchmark.experiment_runner import load_catalog
    from pinn_benchmark.datasets import branch_solution_from_record
    from pinn_benchmark.numerical_solver import convergence_study, solve_branch, choose_scheme

    records = load_catalog(ROOT)
    if args.branches:
        sel = [r for r in records if r["branch_id"] in args.branches.split(",")]
    else:
        sel = [r for r in records if r.get("group") == "main"]

    out_dir = ROOT / "results" / "numerical"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_out = {}
    for rec in sel:
        bid = rec["branch_id"]
        fpath = out_dir / f"{bid}.json"
        if fpath.exists() and not args.force:
            log.info("%s cached, skipping", bid)
            all_out[bid] = utils.load_json(fpath)
            continue
        sol = branch_solution_from_record(rec, 0)
        res_list = [65, 129, 257] if not args.quick else [65, 129]
        tols = [1e-8, 1e-10] if not args.quick else [1e-8]
        log.info("convergence study %s (fd_mol primary) ...", bid)
        study = convergence_study(sol, "fd_mol", res_list, tols)
        entry = {"branch_id": bid, "primary": study}
        # independent cross-check scheme for periodic-compatible profiles
        props = rec.get("properties", {})
        if (props.get("localized") or float(props.get("feat_gradient_max", 1)) < 1e-12) \
                and not props.get("singular_at_xi0") and not props.get("fractional_power"):
            entry["cross_check_fourier"] = solve_branch(
                sol, "fourier_etdrk4", nt=256, nz_steps=400, pad_factor=3.0)
        save_json(entry, fpath)
        all_out[bid] = entry
        best = min(r["final_l2re"] for r in study["runs"])
        log.info("  best L2RE %.2e | observed orders %s", best,
                 [f"{o:.1f}" for o in study["observed_orders"]])

    save_json(all_out, out_dir / "summary.json")

    # markdown report
    lines = ["# Independent numerical solver validation", "",
             "Primary scheme: 6th-order finite differences in t with three ghost",
             "nodes per side filled from the exact analytic solution (Dirichlet-type",
             "boundary strip - NOT periodic; kinks are never given periodic BCs),",
             "integrated in z with scipy DOP853. Cross-check scheme for",
             "periodic-compatible (localized/constant) profiles: Fourier",
             "pseudospectral + ETDRK4 on a 3x padded window.", ""]
    for bid, entry in all_out.items():
        lines.append(f"## {bid}")
        lines.append("")
        lines.append("| scheme | nt | rtol/steps | final L2RE | final max err | runtime (s) |")
        lines.append("|---|---|---|---|---|---|")
        for r in entry["primary"]["runs"]:
            lines.append(f"| fd_mol | {r['nt']} | {r['rtol']:g} | {r['final_l2re']:.2e} | "
                         f"{r['final_max_err']:.2e} | {r['runtime_s']:.2f} |")
        if "cross_check_fourier" in entry:
            r = entry["cross_check_fourier"]
            lines.append(f"| fourier_etdrk4 | {r['nt']} | {r['nz_steps']} steps | "
                         f"{r['final_l2re']:.2e} | {r['final_max_err']:.2e} | {r['runtime_s']:.2f} |")
        lines.append("")
        oo = entry["primary"]["observed_orders"]
        lines.append(f"Observed spatial convergence order: {[f'{o:.2f}' for o in oo]} "
                     "(6th-order stencils).")
        lines.append("")
    (ROOT / "reports" / "numerical_solver_validation.md").write_text("\n".join(lines))
    log.info("wrote reports/numerical_solver_validation.md")


if __name__ == "__main__":
    main()
