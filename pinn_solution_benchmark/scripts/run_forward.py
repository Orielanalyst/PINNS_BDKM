#!/usr/bin/env python
"""Phase 4: forward PINN experiments on verified branches.

Usage:
    python scripts/run_forward.py --config configs/pilot.yaml [--force]
        [--seeds 0,1] [--methods vanilla] [--workers 3] [--jobs-list]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pinn_benchmark import utils  # noqa: E402
from pinn_benchmark.utils import log  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true", help="rerun completed jobs")
    ap.add_argument("--seeds", default="", help="comma list to restrict seeds")
    ap.add_argument("--methods", default="", help="comma list to restrict methods")
    ap.add_argument("--workers", type=int, default=0, help="override config workers")
    ap.add_argument("--jobs-list", action="store_true", help="print jobs and exit")
    args = ap.parse_args()

    utils.setup_logging()
    cfg = utils.load_config(args.config)
    if args.seeds:
        cfg["seeds"] = [int(s) for s in args.seeds.split(",")]
    if args.methods:
        cfg["methods"] = args.methods.split(",")
    workers = args.workers or cfg.get("workers", 1)
    device = utils.resolve_device(cfg.get("device", "auto"))
    log.info("device: %s", device)

    from pinn_benchmark.experiment_runner import (expand_forward_jobs, load_catalog,
                                                  run_forward_experiment)
    if args.jobs_list:
        jobs = expand_forward_jobs(cfg, load_catalog(ROOT))
        for j in jobs:
            print(j["record"]["branch_id"], j["rep"], j["method"], j["seed"])
        print(f"total: {len(jobs)}")
        return

    results = run_forward_experiment(cfg, ROOT, device, force=args.force, workers=workers)
    done = [r for r in results if r.get("state") == "done"]
    log.info("done %d/%d jobs", len(done), len(results))
    from pinn_benchmark.report_forward import write_forward_figures
    write_forward_figures(ROOT, cfg, results)


if __name__ == "__main__":
    main()
