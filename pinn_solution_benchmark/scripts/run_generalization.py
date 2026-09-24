#!/usr/bin/env python
"""Phase 8: conditioned cross-branch PINN protocols (resumable per protocol).

Protocols: interpolation, extrapolation, leave-one-branch-out (zero-shot),
fine-tuning with 1%/5% held-out data, and a matched-budget from-scratch
baseline.

Usage:
    python scripts/run_generalization.py --config configs/generalization_pilot.yaml
        [--seeds 0] [--protocols interp,lobo] [--force]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pinn_benchmark import utils  # noqa: E402
from pinn_benchmark.utils import log, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "generalization_pilot.yaml"))
    ap.add_argument("--seeds", default="")
    ap.add_argument("--protocols", default="interp,extrap,lobo")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    utils.setup_logging()
    cfg = utils.load_config(args.config)
    if args.seeds:
        cfg["seeds"] = [int(s) for s in args.seeds.split(",")]
    protocols = args.protocols.split(",")
    device = utils.resolve_device(cfg.get("device", "auto"))

    import torch
    torch.set_num_threads(2)
    from pinn_benchmark.experiment_runner import load_catalog
    from pinn_benchmark.conditioned_pinn import (ConditionedTrainer,
                                                 make_rep_from_constraints,
                                                 task_from_rep)

    records = {r["branch_id"]: r for r in load_catalog(ROOT)}
    train_recs = [records[b] for b in cfg["train_branches"]]
    exp_dir = ROOT / "results" / cfg["experiment_name"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_json(utils.environment_info(), exp_dir / "environment.json")

    def catalog_tasks(recs):
        tasks = []
        for rec in recs:
            for rep in rec["representative_sets"]:
                rep_f = {k: float(__import__("sympy").sympify(v)) for k, v in rep.items()}
                tasks.append(task_from_rep(rec, rep_f))
        return tasks

    def new_task(rec, assignments):
        free = rec["free_parameters"]
        assign = {k: v for k, v in assignments.items() if k in free}
        rep = make_rep_from_constraints(rec, assign)
        return task_from_rep(rec, rep)

    tr = cfg["training"]
    for seed in cfg["seeds"]:
        out_path = exp_dir / f"seed{seed}.json"
        if out_path.exists() and not args.force:
            log.info("seed %d cached, skipping", seed)
            continue
        results = {"seed": seed}

        if "interp" in protocols or "extrap" in protocols:
            log.info("[seed %d] training conditioned PINN on all train branches ...", seed)
            trainer = ConditionedTrainer(catalog_tasks(train_recs), cfg, device, seed)
            t0 = time.perf_counter()
            trainer.train(tr["steps"], lr=tr.get("adam_lr", 2e-3))
            results["train_time_s"] = time.perf_counter() - t0
            results["train_task_eval"] = {}
            for task in catalog_tasks(train_recs):
                key = f"({task.n},{task.m})_l2={task.lambda2}"
                results["train_task_eval"][key] = trainer.evaluate(task)["l2re_complex"]
            for proto, akey in [("interp", "interp_assignments"),
                                ("extrap", "extrap_assignments")]:
                if proto not in protocols:
                    continue
                results[proto] = {}
                for rec in train_recs:
                    try:
                        task = new_task(rec, cfg[akey])
                        m = trainer.evaluate(task)
                        results[proto][rec["branch_id"]] = m["l2re_complex"]
                    except Exception as e:  # noqa: BLE001
                        results[proto][rec["branch_id"]] = f"error: {e}"
                log.info("[seed %d] %s: %s", seed, proto, results[proto])

        if "lobo" in protocols:
            results["lobo"] = {}
            for held in cfg["train_branches"]:
                keep = [records[b] for b in cfg["train_branches"] if b != held]
                held_rec = records[held]
                log.info("[seed %d] LOBO hold out %s ...", seed, held)
                trainer = ConditionedTrainer(catalog_tasks(keep), cfg, device, seed)
                trainer.train(tr["steps"], lr=tr.get("adam_lr", 2e-3))
                entry = {"train_branches": [r["branch_id"] for r in keep]}
                held_tasks = catalog_tasks([held_rec])
                entry["zero_shot"] = {
                    f"rep{i}": trainer.evaluate(task)["l2re_complex"]
                    for i, task in enumerate(held_tasks)}
                # fine-tune with sparse held-out data (matched budget baselines)
                for frac in cfg.get("fine_tune_fractions", []):
                    ft = copy.deepcopy(trainer)
                    ft.tasks = held_tasks
                    ft._build_data()
                    ft.train(tr["fine_tune_steps"], lr=tr.get("adam_lr", 2e-3) / 2,
                             data_fraction=frac)
                    entry[f"fine_tune_{frac}"] = {
                        f"rep{i}": ft.evaluate(task)["l2re_complex"]
                        for i, task in enumerate(held_tasks)}
                # from-scratch baseline at the same budget as fine-tuning
                scratch = ConditionedTrainer(held_tasks, cfg, device, seed)
                scratch.train(tr["scratch_steps"], lr=tr.get("adam_lr", 2e-3))
                entry["scratch_same_budget"] = {
                    f"rep{i}": scratch.evaluate(task)["l2re_complex"]
                    for i, task in enumerate(held_tasks)}
                results["lobo"][held] = entry
                log.info("[seed %d] LOBO %s: zero-shot %s | scratch %s", seed, held,
                         entry["zero_shot"], entry["scratch_same_budget"])

        save_json(results, out_path)
    # aggregate all seeds
    allres = []
    for f in sorted(exp_dir.glob("seed*.json")):
        with open(f) as fh:
            allres.append(json.load(fh))
    save_json(allres, exp_dir / "all_results.json")
    log.info("generalization results in %s", exp_dir)


if __name__ == "__main__":
    main()
