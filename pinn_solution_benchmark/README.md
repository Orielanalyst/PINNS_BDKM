# pinn_solution_benchmark

Reproducible audit of the 49-branch traveling-wave solution catalogue proposed
in `../paper-5.pdf` (iB-function ansatz for a fourth-order gain–loss NLS-type
equation), and a PINN benchmark on the branches that **survive independent
verification**.

## Headline results

- **The 49-branch claim is not supported.** Under the paper's constant-phase
  real-amplitude ansatz, only 9 (n,m) pairs admit any verified nontrivial
  solution family — 3 smooth/bounded ("main": constant (0,0), bright sech
  (1,0), dark kink (1,1)) and 6 singular/unbounded ("stress") — and every one
  is a *reduced* model (several coefficients forced to zero). Proofs: exact
  symbolic certificate + 50-digit residuals (`reports/mathematical_audit.md`).
- The paper's own classification (Eq. 28) allows only n = m; its 42
  off-diagonal pairs are unjustified — yet the one bright branch that closes
  is off-diagonal, so index matching is neither necessary nor sufficient.
- The paper's residual split (Eqs. 37–38) misplaces the β₃/β₄ terms; the
  corrected split is derived and unit-tested here.
- An optional, clearly separated extended-phase experiment
  (`--extended-phase`) shows that restoring a carrier phase
  e^{i(κz+ωt)} immediately recovers traveling bright/dark NLS solitons.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or CUDA wheel
pip install -r requirements.txt && pip install -e .
```

## Exact commands

```bash
# Phase 1–2: audit all 49 pairs (fast; resumable per pair)
python scripts/run_audit.py --config configs/audit.yaml
python scripts/run_audit.py --config configs/audit.yaml --extended-phase   # optional

# unit tests (run before any training)
python -m pytest tests/ -q

# Phase 3: independent numerical solver + convergence study
python scripts/run_numerical.py

# Phase 4–5: forward PINN
python scripts/run_forward.py --config configs/smoke.yaml   # minutes
python scripts/run_forward.py --config configs/pilot.yaml   # 90 jobs, hours on CPU
python scripts/run_forward.py --config configs/full.yaml    # recommended on GPU

# Phase 6: ablations        Phase 7: inverse        Phase 8: conditioned PINN
python scripts/run_ablations.py
python scripts/run_inverse.py --config configs/inverse_pilot.yaml
python scripts/run_generalization.py --config configs/generalization_pilot.yaml

# everything in order, resumably:
python scripts/run_full_benchmark.py --mode smoke|pilot|full

# final report
python scripts/make_report.py
```

Every experiment **skips completed jobs** (rerun with `--force`) and resumes
interrupted training from checkpoints. Restrict any run with
`--seeds/--methods/--stages/...`; the same command without restrictions
resumes the remainder.

## Layout

- `src/pinn_benchmark/` — library (audit: `symbolic_audit`, `closure_solver`,
  `branch_classifier`; PINN: `networks`, `losses`, `trainers`,
  `adaptive_sampling`, `inverse_problem`, `conditioned_pinn`; validation:
  `numerical_solver`; infrastructure: `datasets`, `metrics`, `plotting`,
  `experiment_runner`, `utils`)
- `configs/` — audit/smoke/pilot/full + inverse + generalization YAMLs.
  `pilot.yaml` is CPU-scaled (this machine: 4 cores, no GPU); `full.yaml`
  holds the recommended settings (6×64 tanh net, 10k collocation points).
- `data/verified_branches/` — branch_catalog.{csv,json}, rejected_pairs.csv
- `reports/` — paper_equation_inventory.md, mathematical_audit.md,
  numerical_solver_validation.md, ablations.md, final_benchmark_report.md,
  figures/
- `results/`, `checkpoints/` — raw per-run outputs (never overwritten without
  `--force`)
- `notebooks/run_benchmark_colab.ipynb` — thin Colab front end

## Conventions

- PDE and residual split: see `src/pinn_benchmark/equations.py` docstring;
  c = g₀T₂², ℓ = (g₀−α)/2. Only c and ℓ are identifiable (not g₀, T₂, α
  separately) — proven by the sensitivity analysis in Phase 7.
- Fractional powers of sinh are never interpreted silently: conventions
  `principal` / `signed` / `positive_xi` are explicit
  (`src/pinn_benchmark/ansatz.py`); fractional and singular branches live in
  the stress group and are excluded from the main forward benchmark.
- Success criterion (configurable): complex L2RE < 1e-2 AND finite
  independently-sampled PDE residual; raw errors are always reported.
