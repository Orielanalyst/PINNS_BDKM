#!/usr/bin/env python
"""Phase 1-2: symbolic audit of all 49 proposed (n,m) branches.

Resumable: per-pair results are cached under results/audit/pairs/ and skipped
on re-run unless --force is given.

Usage:
    python scripts/run_audit.py --config configs/audit.yaml [--force]
        [--pairs "1,1;1,0"] [--extended-phase]
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pinn_benchmark import utils  # noqa: E402
from pinn_benchmark.utils import log, save_json, load_json  # noqa: E402


def audit_pair(n: Fraction, m: Fraction, cfg: dict, extended: bool = False) -> dict:
    from pinn_benchmark.symbolic_audit import closure_certificate
    from pinn_benchmark.closure_solver import solve_closure
    from pinn_benchmark.branch_classifier import classify_family
    from dataclasses import asdict

    t0 = time.perf_counter()
    cs = closure_certificate(n, m, extended_phase=extended)
    fams = solve_closure(cs, max_depth=cfg.get("max_depth", 14),
                         max_families=cfg.get("max_families", 60))
    records = []
    for i, fam in enumerate(fams):
        try:
            rec = classify_family(fam, i,
                                  residual_tol_max=cfg.get("residual_tol_max", 1e-10),
                                  residual_tol_rel=cfg.get("residual_tol_rel", 1e-10),
                                  extended=extended)
            records.append(asdict(rec))
        except Exception as e:  # noqa: BLE001 - report, never silently drop
            records.append({
                "branch_id": f"n{n}_m{m}_f{i}", "n": str(n), "m": str(m),
                "family_index": i, "tier": "C", "group": "none",
                "rejection_reason": f"classification error: {e}",
                "traceback": traceback.format_exc()[-2000:],
            })
    return {
        "n": str(n), "m": str(m),
        "n_equations": len(cs.equations),
        "certificate_notes": cs.notes,
        "cleared_denominators": cs.cleared_denominators,
        "n_families_found": len(fams),
        "families": records,
        "elapsed_s": time.perf_counter() - t0,
        "extended_phase": extended,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "audit.yaml"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--pairs", default="", help='subset like "1,1;1,0;3/2,-1/2"')
    ap.add_argument("--extended-phase", action="store_true",
                    help="OPTIONAL separate experiment: carrier-phase ansatz "
                         "A = a J(xi) exp(i(kappa z + omega t))")
    args = ap.parse_args()

    utils.setup_logging()
    cfg = utils.load_config(args.config)
    from pinn_benchmark.ansatz import ALL_PAIRS

    tag = "extended" if args.extended_phase else "standard"
    out_dir = ROOT / "results" / "audit" / tag
    pair_dir = out_dir / "pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    save_json(utils.environment_info(), out_dir / "environment.json")

    if args.pairs:
        pairs = []
        for chunk in args.pairs.split(";"):
            a, b = chunk.split(",")
            pairs.append((Fraction(a), Fraction(b)))
    else:
        pairs = ALL_PAIRS

    # global symbolic checks (fast, always rerun)
    from pinn_benchmark.symbolic_audit import (compare_offset_coefficients,
                                               classification_analysis,
                                               residual_split_discrepancies)
    log.info("comparing derived C_k with paper Eqs. (17)-(25), n,m symbolic ...")
    offs = compare_offset_coefficients()
    cls = classification_analysis()
    global_checks = {
        "offset_coefficients_match_paper": offs,
        "all_offsets_match": all(offs.values()),
        "classification": cls,
        "residual_split_discrepancies": residual_split_discrepancies(),
    }
    save_json(global_checks, out_dir / "global_checks.json")
    log.info("offset coefficients all match paper: %s", all(offs.values()))
    log.info("Eq.(28) forces n=m: %s", cls["eq28_forces_n_equals_m"])

    all_results = []
    for n, m in pairs:
        fname = pair_dir / f"n{n}_m{m}.json".replace("/", "o")
        if fname.exists() and not args.force:
            log.info("(%s,%s) cached, skipping", n, m)
            all_results.append(load_json(fname))
            continue
        log.info("auditing (n,m) = (%s,%s) ...", n, m)
        try:
            res = audit_pair(n, m, cfg, extended=args.extended_phase)
        except Exception as e:  # noqa: BLE001
            res = {"n": str(n), "m": str(m), "error": str(e),
                   "traceback": traceback.format_exc()[-3000:], "families": []}
            log.error("pair (%s,%s) failed: %s", n, m, e)
        save_json(res, fname)
        all_results.append(res)
        nfam = res.get("n_families_found", 0)
        acc = sum(1 for f in res.get("families", []) if not f.get("rejection_reason"))
        log.info("  -> %d families, %d accepted, %.1fs", nfam, acc, res.get("elapsed_s", 0))

    save_json(all_results, out_dir / "all_pairs.json")

    # catalog + reports
    from pinn_benchmark.report_audit import write_catalog_and_reports
    write_catalog_and_reports(ROOT, all_results, global_checks, cfg, tag=tag)
    log.info("audit complete: results in %s", out_dir)


if __name__ == "__main__":
    main()
