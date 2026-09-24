#!/usr/bin/env python
"""Run the complete benchmark pipeline in order, resumably.

    python scripts/run_full_benchmark.py --mode smoke|pilot|full [--force]

Order: audit -> unit tests -> numerical validation -> forward PINN ->
inverse -> ablations -> generalization -> report. Each stage skips completed
work unless --force. Expensive PINN stages never start unless the audit
produced a verified catalog.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd: list, allow_fail=False):
    print(f"\n=== {' '.join(str(c) for c in cmd)} ===", flush=True)
    r = subprocess.run([PY] + [str(c) for c in cmd], cwd=ROOT)
    if r.returncode != 0 and not allow_fail:
        sys.exit(f"stage failed: {' '.join(map(str, cmd))}")
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "pilot", "full"], default="pilot")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()
    force = ["--force"] if args.force else []

    # 1. mathematical audit (gates everything else)
    run(["scripts/run_audit.py", "--config", "configs/audit.yaml"] + force)
    catalog = ROOT / "data" / "verified_branches" / "branch_catalog.json"
    if not catalog.exists():
        sys.exit("audit produced no catalog; aborting before PINN training")

    # 2. unit tests
    if not args.skip_tests:
        rc = subprocess.run([PY, "-m", "pytest", "tests/", "-q"], cwd=ROOT).returncode
        if rc != 0:
            sys.exit("unit tests failed; aborting before PINN training")

    # 3. numerical solver validation
    quick = ["--quick"] if args.mode == "smoke" else []
    run(["scripts/run_numerical.py"] + quick + force)

    # 4. forward PINN
    cfgmap = {"smoke": "configs/smoke.yaml", "pilot": "configs/pilot.yaml",
              "full": "configs/full.yaml"}
    run(["scripts/run_forward.py", "--config", cfgmap[args.mode]] + force)

    # 5-7. inverse, ablations, generalization (pilot/full only)
    if args.mode != "smoke":
        inv_extra = ([] if args.mode == "full" else
                     ["--densities", "0.05", "--noises", "0.0,0.01,0.05",
                      "--seeds", "0,1,2", "--stages", "gainloss,nonlinear"])
        run(["scripts/run_inverse.py", "--config", "configs/inverse_pilot.yaml"]
            + inv_extra + force, allow_fail=True)
        run(["scripts/run_ablations.py"] + force, allow_fail=True)
        gen_extra = [] if args.mode == "full" else ["--seeds", "0"]
        run(["scripts/run_generalization.py", "--config",
             "configs/generalization_pilot.yaml"] + gen_extra + force, allow_fail=True)

    # 8. final report
    run(["scripts/make_report.py"])


if __name__ == "__main__":
    main()
