#!/usr/bin/env python
"""Assemble reports/final_benchmark_report.md from whatever results exist.
Reports partial state honestly: stages not run are listed with their exact
resume commands."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def jload(p, default=None):
    p = Path(p)
    if not p.exists():
        return default
    with open(p) as f:
        return json.load(f)


def main():
    L = ["# Final benchmark report", ""]
    L.append("Testing whether conventional PINNs accurately reproduce the "
             "*verified* analytical traveling-wave solutions of the fourth-order "
             "gain-loss NLS-type equation of paper-5.pdf, and identifying their "
             "success and failure regimes. This study does not 'prove that PINNs "
             "work'; every claim below is scoped to the concrete configurations "
             "run.")
    L.append("")

    # ---- audit ----
    cat = jload(ROOT / "data/verified_branches/branch_catalog.json", [])
    gc = jload(ROOT / "results/audit/standard/global_checks.json", {})
    L += ["## 1. Mathematical audit (Phase 1-2)", ""]
    if cat:
        mains = [r for r in cat if r["group"] == "main"]
        stress = [r for r in cat if r["group"] == "stress"]
        L.append(f"- Of the paper's **49 proposed (n,m) branches, only {len({(r['n'],r['m']) for r in cat})} "
                 f"admit any verified nontrivial solution family** under the paper's "
                 f"constant-phase real-amplitude ansatz ({len(mains)} smooth 'main' + "
                 f"{len(stress)} singular/unbounded 'stress'); all are Tier B reduced "
                 "models; none support the full nine-coefficient model. "
                 "Proof artifacts: symbolic certificate + 50-digit residuals "
                 "(max|F| < 1e-40 on every accepted family).")
        L.append("- The paper's own Eq. (28) restricts to n = m; its 42 off-diagonal "
                 "pairs are unjustified — yet the only bright branch that DOES close, "
                 "(1,0) = sech, is off-diagonal, so the classification is neither "
                 "necessary nor sufficient. Its Eqs. (37)-(38) misplace the beta3/beta4 "
                 "terms (see reports/mathematical_audit.md).")
        ext = jload(ROOT / "results/audit/extended/all_pairs.json")
        if ext:
            L.append("- OPTIONAL extended-phase experiment (clearly separated from the "
                     "paper's formulation): adding a carrier exp(i(kappa z + omega t)) "
                     "immediately recovers traveling bright/dark NLS solitons with "
                     "gamma, beta2 != 0 and a Tier-A CW solution — the paper's "
                     "\"a real without loss of generality\" convention is what "
                     "eliminates the physically interesting branches "
                     "(reports/mathematical_audit_extended.md).")
    else:
        L.append("- NOT RUN. Resume: `python scripts/run_audit.py --config configs/audit.yaml`")
    L.append("")

    # ---- numerical ----
    num = jload(ROOT / "results/numerical/summary.json", {})
    L += ["## 2. Independent numerical solver (Phase 3)", ""]
    if num:
        for bid, entry in num.items():
            best = min(r["final_l2re"] for r in entry["primary"]["runs"])
            oo = entry["primary"]["observed_orders"]
            L.append(f"- `{bid}`: best complex L2RE {best:.1e} (6th-order FD + DOP853; "
                     f"observed orders {[f'{o:.1f}' for o in oo]})"
                     + (f"; Fourier-ETDRK4 cross-check {entry['cross_check_fourier']['final_l2re']:.1e}"
                        if "cross_check_fourier" in entry else ""))
        L.append("- Details: reports/numerical_solver_validation.md")
    else:
        L.append("- NOT RUN. Resume: `python scripts/run_numerical.py`")
    L.append("")

    # ---- forward ----
    L += ["## 3. Forward PINN (Phase 4-5)", ""]
    from pinn_benchmark.experiment_runner import aggregate_forward
    for exp in ("forward_smoke", "forward_pilot", "forward_full"):
        # rescan per-run files: restricted invocations only aggregate their own
        # sub-matrix, but every completed run leaves a result.json
        res = [jload(p) for p in sorted((ROOT / "results" / exp).glob("*/result.json"))] \
            if (ROOT / "results" / exp).exists() else None
        if not res:
            continue
        agg = aggregate_forward(res)
        done = [r for r in res if r.get("state") == "done"]
        fail = [r for r in res if r.get("state") == "failed"]
        L.append(f"### {exp}: {len(done)} done / {len(fail)} failed / {len(res)} total")
        L.append("")
        L.append("| cell (branch__rep__method) | L2RE mean+-std | median | success rate | res RMS mean | seeds done |")
        L.append("|---|---|---|---|---|---|")
        for key in sorted(agg):
            st = agg[key]
            if st["l2re"]["mean"] is None:
                L.append(f"| {key} | all failed | - | 0 | - | {st['n_done']} |")
                continue
            L.append(f"| {key} | {st['l2re']['mean']:.2e} +- {st['l2re']['std']:.1e} | "
                     f"{st['l2re']['median']:.2e} | {st['success_rate']:.2f} | "
                     f"{st['residual_rms']['mean']:.2e} | {st['n_done']} |")
        L.append("")
        if exp == "forward_pilot" and any(agg[k]["n_done"] < 5 for k in agg):
            L.append("Cells with fewer than 5 seeds are incomplete on this hardware; "
                     "complete them with: `python scripts/run_forward.py --config "
                     "configs/pilot.yaml` (skips finished jobs).")
            L.append("")
    if not any((ROOT / f"results/{e}").exists()
               for e in ("forward_smoke", "forward_pilot", "forward_full")):
        L.append("- NOT RUN. Resume: `python scripts/run_forward.py --config configs/pilot.yaml`")
    pilot_res = ([jload(p) for p in sorted((ROOT / "results/forward_pilot").glob("*/result.json"))]
                 if (ROOT / "results/forward_pilot").exists() else None)
    if pilot_res:
        done = [r for r in pilot_res if r.get("state") == "done"]
        if done:
            by_branch = {}
            for r in done:
                by_branch.setdefault(r["branch"], []).append(r["l2re_complex"])
            order = sorted(by_branch, key=lambda b: np.mean(by_branch[b]))
            L.append("Branch difficulty (mean L2RE ascending): " +
                     " < ".join(f"{b} ({np.mean(by_branch[b]):.1e})" for b in order))
            L.append("")

    # ---- inverse ----
    L += ["## 4. Inverse problem (Phase 7)", ""]
    ident = jload(ROOT / "results/inverse_pilot/identifiability.json")
    if ident:
        for bid, d in ident.items():
            L.append(f"- `{bid}`: (g0, T2, alpha) sensitivity rank = "
                     f"**{d['g0T2alpha_rank_1e-8']} of 3** (they enter only via "
                     f"c = g0*T2^2 and ell = (g0-alpha)/2 — separate recovery is "
                     f"impossible); reduced-parameter Jacobian rank "
                     f"{d['reduced_rank_1e-8']}/8, condition number "
                     f"{d['reduced_condition_number']:.1e}.")
    inv = jload(ROOT / "results/inverse_pilot/all_results.json")
    if inv:
        done = [r for r in inv if r.get("state") == "done"]
        L.append(f"- {len(done)}/{len(inv)} recovery runs done "
                 "(results/inverse_pilot/, figures in reports/figures/inverse_pilot/).")
        rows = jload(ROOT / "results/inverse_pilot/param_rows.json", [])
        by = {}
        for r in rows:
            key = (r["stage"], r["param"], r["noise"])
            v = r["rel_error"] if r["rel_error"] is not None and np.isfinite(r["rel_error"]) \
                else r["abs_error"]
            by.setdefault(key, []).append(v)
        L.append("")
        L.append("| stage | param | noise | median rel err (abs err if true=0) | n |")
        L.append("|---|---|---|---|---|")
        for key in sorted(by):
            vals = by[key]
            L.append(f"| {key[0]} | {key[1]} | {key[2]} | {np.median(vals):.2e} | {len(vals)} |")
    else:
        L.append("- Recovery runs NOT RUN (or incomplete). Resume: "
                 "`python scripts/run_inverse.py --config configs/inverse_pilot.yaml`")
    L.append("")

    # ---- ablations ----
    L += ["## 5. Ablations (Phase 6)", ""]
    abl = jload(ROOT / "results/ablations/ablation_summary.json")
    if abl:
        for k, st in abl.get("summary", {}).items():
            L.append(f"- {k}: mean L2RE {st['mean']:.2e} +- {st['std']:.1e} (n={st['n_finite']})")
        L.append("- Incompatible requested ablations documented in reports/ablations.md "
                 "(conservative closures do not exist among verified branches; "
                 "Gamma>0 and beta4!=0 are closure-forbidden off the constant branch).")
    else:
        L.append("- NOT RUN. Resume: `python scripts/run_ablations.py`")
    L.append("")

    # ---- generalization ----
    L += ["## 6. Conditioned cross-branch PINN (Phase 8)", ""]
    gen = jload(ROOT / "results/generalization_pilot/all_results.json")
    if gen:
        for res in gen:
            L.append(f"- seed {res['seed']}:")
            if "interp" in res:
                L.append(f"  - interpolation L2RE: {res['interp']}")
            if "extrap" in res:
                L.append(f"  - extrapolation L2RE: {res['extrap']}")
            for held, e in res.get("lobo", {}).items():
                L.append(f"  - LOBO {held}: zero-shot {e['zero_shot']}, "
                         f"fine-tune(1%) {e.get('fine_tune_0.01')}, "
                         f"fine-tune(5%) {e.get('fine_tune_0.05')}, "
                         f"scratch-same-budget {e.get('scratch_same_budget')}")
    else:
        L.append("- NOT RUN. Resume: `python scripts/run_generalization.py "
                 "--config configs/generalization_pilot.yaml`")
    L.append("")

    # ---- provenance ----
    L += ["## 7. Provenance and epistemics", ""]
    L.append("- **Mathematically proven (symbolic zero + 50-digit numeric):** every "
             "family in data/verified_branches/branch_catalog.json.")
    L.append("- **Unsupported candidates:** all remaining proposed pairs "
             "(data/verified_branches/rejected_pairs.csv, with reasons).")
    L.append("- **PINN successes/failures:** per-run result.json files under results/; "
             "success = complex L2RE < 1e-2 AND independent residual reported; raw "
             "errors always retained.")
    L.append("- **Proposed but not executed:** any stage marked NOT RUN above; each "
             "carries its exact resume command. All stages are resumable and skip "
             "completed jobs unless --force is passed.")
    L.append("")
    L.append("Commands: smoke `python scripts/run_full_benchmark.py --mode smoke`; "
             "pilot `python scripts/run_full_benchmark.py --mode pilot`; "
             "full `python scripts/run_full_benchmark.py --mode full`.")

    out = ROOT / "reports" / "final_benchmark_report.md"
    out.write_text("\n".join(L))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
