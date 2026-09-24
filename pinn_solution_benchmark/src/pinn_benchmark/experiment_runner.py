"""Experiment runner: expands the job matrix from a config, runs jobs
resumably (skip completed unless --force), saves raw per-run results before
aggregation, and never overwrites without the explicit flag.

Job identity: (experiment, branch_id, rep_index, method, seed) -> stable
directory name under results/<experiment>/.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import traceback
from fractions import Fraction
from pathlib import Path

import numpy as np

from .datasets import branch_solution_from_record
from .utils import environment_info, log, save_json, aggregate_stats


def load_catalog(root: Path, catalog: str = "branch_catalog.json") -> list[dict]:
    path = root / "data" / "verified_branches" / catalog
    with open(path) as f:
        return json.load(f)


def select_pilot_branches(records: list[dict], k: int = 3) -> list[dict]:
    """Choose k main-group branches spanning easy -> difficult, ranked by the
    max-fourth-derivative feature (the PDE's hardest term), preferring
    physically distinct profiles. Deterministic; no result-based cherry-picking."""
    main = [r for r in records if r.get("group") == "main" and not r.get("rejection_reason")]
    main.sort(key=lambda r: float(r["properties"].get("feat_max_fourth_deriv_tttt", 0.0)))
    if len(main) <= k:
        return main
    idx = np.linspace(0, len(main) - 1, k).round().astype(int)
    return [main[i] for i in sorted(set(idx))]


def job_id(experiment: str, branch_id: str, rep: int, method: str, seed: int) -> str:
    return f"{experiment}__{branch_id}__rep{rep}__{method}__seed{seed}"


def expand_forward_jobs(cfg: dict, records: list[dict]) -> list[dict]:
    jobs = []
    sel = cfg.get("branches", "auto")
    if sel == "auto":
        chosen = select_pilot_branches(records, k=cfg.get("n_pilot_branches", 3))
    elif sel == "all_main":
        chosen = [r for r in records if r.get("group") == "main"]
    elif sel == "all":
        chosen = records
    else:
        chosen = [r for r in records if r["branch_id"] in sel]
    n_reps = cfg.get("param_sets_per_branch", 1)
    for rec in chosen:
        for rep in range(min(n_reps, len(rec["representative_sets"]))):
            for method in cfg.get("methods", ["vanilla"]):
                for seed in cfg.get("seeds", [0]):
                    jobs.append({"experiment": cfg.get("experiment_name", "forward"),
                                 "record": rec, "rep": rep, "method": method,
                                 "seed": seed})
    return jobs


def _run_one_forward(job: dict, cfg: dict, root: Path, device: str, force: bool) -> dict:
    from .trainers import train_forward  # local import: torch init per process

    rec = job["record"]
    jid = job_id(job["experiment"], rec["branch_id"], job["rep"], job["method"], job["seed"])
    run_dir = root / "results" / job["experiment"] / jid
    result_path = run_dir / "result.json"
    if result_path.exists() and not force:
        log.info("skip completed %s", jid)
        with open(result_path) as f:
            return json.load(f)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = root / "checkpoints" / job["experiment"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sol = branch_solution_from_record(rec, job["rep"], z_max=cfg.get("domain", {}).get("z_max", 1.0))
    status = {"job": jid, "state": "running"}
    save_json(status, run_dir / "status.json")
    try:
        metrics = train_forward(sol, cfg, job["method"], job["seed"],
                                run_dir=run_dir, device=device,
                                resume=not force)
        out = {"job": jid, "experiment": job["experiment"],
               "branch_id": rec["branch_id"], "branch": f"({rec['n']},{rec['m']})",
               "rep": job["rep"], "method": job["method"], "seed": job["seed"],
               "state": "done", **metrics}
    except Exception as e:  # noqa: BLE001
        out = {"job": jid, "experiment": job["experiment"],
               "branch_id": rec["branch_id"], "branch": f"({rec['n']},{rec['m']})",
               "rep": job["rep"], "method": job["method"], "seed": job["seed"],
               "state": "failed", "error": str(e),
               "traceback": traceback.format_exc()[-3000:]}
        log.error("job %s FAILED: %s", jid, e)
    save_json(out, result_path)
    save_json({"job": jid, "state": out["state"]}, run_dir / "status.json")
    return out


def run_forward_experiment(cfg: dict, root: Path, device: str, force: bool = False,
                           workers: int = 1, job_filter=None) -> list[dict]:
    records = load_catalog(root)
    jobs = expand_forward_jobs(cfg, records)
    if job_filter:
        jobs = [j for j in jobs if job_filter(j)]
    log.info("forward experiment '%s': %d jobs (%d workers)",
             cfg.get("experiment_name", "forward"), len(jobs), workers)
    exp_dir = root / "results" / cfg.get("experiment_name", "forward")
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_json(environment_info(), exp_dir / "environment.json")
    save_json({k: v for k, v in cfg.items() if k != "record"}, exp_dir / "config_used.json")

    if workers <= 1:
        results = [_run_one_forward(j, cfg, root, device, force) for j in jobs]
    else:
        with mp.get_context("spawn").Pool(workers, initializer=_limit_threads) as pool:
            results = pool.starmap(
                _run_one_forward, [(j, cfg, root, device, force) for j in jobs])
    agg = aggregate_forward(results)
    save_json(results, exp_dir / "all_results.json")
    save_json(agg, exp_dir / "aggregate.json")
    return results


def _limit_threads():
    import torch
    torch.set_num_threads(1)


def aggregate_forward(results: list[dict]) -> dict:
    """Aggregate seed statistics per (branch, rep, method) cell."""
    cells = {}
    for r in results:
        key = f"{r.get('branch_id')}__rep{r.get('rep')}__{r.get('method')}"
        cells.setdefault(key, []).append(r)
    out = {}
    for key, rows in cells.items():
        done = [r for r in rows if r.get("state") == "done"]
        l2 = [r["l2re_complex"] for r in done]
        out[key] = {
            "n_seeds": len(rows),
            "n_done": len(done),
            "n_failed": len(rows) - len(done),
            "success_rate": float(np.mean([r.get("success", False) for r in done])) if done else 0.0,
            "l2re": aggregate_stats(l2),
            "residual_rms": aggregate_stats([r["pde_residual_rms_independent"] for r in done]),
            "train_time_s": aggregate_stats([r["train_time_s"] for r in done]),
        }
    return out
