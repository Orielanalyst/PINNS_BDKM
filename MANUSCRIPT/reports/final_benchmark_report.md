# Final benchmark report

Testing whether conventional PINNs accurately reproduce the *verified* analytical traveling-wave solutions of the fourth-order gain-loss NLS-type equation of paper-5.pdf, and identifying their success and failure regimes. This study does not 'prove that PINNs work'; every claim below is scoped to the concrete configurations run.

## 1. Mathematical audit (Phase 1-2)

- Of the paper's **49 proposed (n,m) branches, only 9 admit any verified nontrivial solution family** under the paper's constant-phase real-amplitude ansatz (3 smooth 'main' + 6 singular/unbounded 'stress'); all are Tier B reduced models; none support the full nine-coefficient model. Proof artifacts: symbolic certificate + 50-digit residuals (max|F| < 1e-40 on every accepted family).
- The paper's own Eq. (28) restricts to n = m; its 42 off-diagonal pairs are unjustified — yet the only bright branch that DOES close, (1,0) = sech, is off-diagonal, so the classification is neither necessary nor sufficient. Its Eqs. (37)-(38) misplace the beta3/beta4 terms (see reports/mathematical_audit.md).
- OPTIONAL extended-phase experiment (clearly separated from the paper's formulation): adding a carrier exp(i(kappa z + omega t)) immediately recovers traveling bright/dark NLS solitons with gamma, beta2 != 0 and a Tier-A CW solution — the paper's "a real without loss of generality" convention is what eliminates the physically interesting branches (reports/mathematical_audit_extended.md).

## 2. Independent numerical solver (Phase 3)

- `n0_m0_f0`: best complex L2RE 1.6e-13 (6th-order FD + DOP853; observed orders ['-0.6', '-2.3']); Fourier-ETDRK4 cross-check 8.6e-18
- `n1_m0_f0`: best complex L2RE 5.8e-11 (6th-order FD + DOP853; observed orders ['6.0', '6.0']); Fourier-ETDRK4 cross-check 2.3e-06
- `n1_m1_f0`: best complex L2RE 7.5e-11 (6th-order FD + DOP853; observed orders ['6.0', '4.6'])
- Details: reports/numerical_solver_validation.md

## 3. Forward PINN (Phase 4-5)

### forward_smoke: 3 done / 0 failed / 3 total

| cell (branch__rep__method) | L2RE mean+-std | median | success rate | res RMS mean | seeds done |
|---|---|---|---|---|---|
| n0_m0_f0__rep0__vanilla | 6.72e-03 +- 0.0e+00 | 6.72e-03 | 1.00 | 1.12e-02 | 1 |
| n1_m0_f0__rep0__vanilla | 3.08e-02 +- 0.0e+00 | 3.08e-02 | 0.00 | 4.18e-02 | 1 |
| n1_m1_f0__rep0__vanilla | 5.28e-03 +- 0.0e+00 | 5.28e-03 | 1.00 | 1.92e-02 | 1 |

### forward_pilot: 90 done / 0 failed / 90 total

| cell (branch__rep__method) | L2RE mean+-std | median | success rate | res RMS mean | seeds done |
|---|---|---|---|---|---|
| n0_m0_f0__rep0__adaptive_sampling | 1.22e-03 +- 1.3e-03 | 6.68e-04 | 1.00 | 2.89e-03 | 5 |
| n0_m0_f0__rep0__adaptive_weights | 3.54e-03 +- 3.2e-03 | 3.76e-03 | 1.00 | 6.24e-03 | 5 |
| n0_m0_f0__rep0__vanilla | 7.58e-04 +- 8.0e-04 | 4.05e-04 | 1.00 | 1.81e-03 | 5 |
| n0_m0_f0__rep1__adaptive_sampling | 1.94e-03 +- 9.7e-04 | 1.69e-03 | 1.00 | 4.81e-03 | 5 |
| n0_m0_f0__rep1__adaptive_weights | 4.60e-03 +- 3.5e-03 | 5.02e-03 | 1.00 | 9.68e-03 | 5 |
| n0_m0_f0__rep1__vanilla | 1.17e-03 +- 8.7e-04 | 6.13e-04 | 1.00 | 3.48e-03 | 5 |
| n1_m0_f0__rep0__adaptive_sampling | 1.80e-02 +- 2.7e-03 | 1.70e-02 | 0.00 | 2.29e-02 | 5 |
| n1_m0_f0__rep0__adaptive_weights | 7.39e-03 +- 1.9e-03 | 7.86e-03 | 1.00 | 1.14e-02 | 5 |
| n1_m0_f0__rep0__vanilla | 1.89e-02 +- 2.4e-03 | 1.94e-02 | 0.00 | 2.58e-02 | 5 |
| n1_m0_f0__rep1__adaptive_sampling | 1.66e-02 +- 4.8e-03 | 1.63e-02 | 0.00 | 2.44e-02 | 5 |
| n1_m0_f0__rep1__adaptive_weights | 2.31e-02 +- 4.2e-03 | 2.30e-02 | 0.00 | 2.61e-02 | 5 |
| n1_m0_f0__rep1__vanilla | 1.94e-02 +- 4.6e-03 | 1.89e-02 | 0.00 | 2.76e-02 | 5 |
| n1_m1_f0__rep0__adaptive_sampling | 5.65e-03 +- 1.6e-03 | 6.10e-03 | 1.00 | 1.74e-02 | 5 |
| n1_m1_f0__rep0__adaptive_weights | 1.68e-03 +- 4.5e-04 | 1.52e-03 | 1.00 | 5.31e-03 | 5 |
| n1_m1_f0__rep0__vanilla | 5.23e-03 +- 7.4e-04 | 5.35e-03 | 1.00 | 1.76e-02 | 5 |
| n1_m1_f0__rep1__adaptive_sampling | 5.36e-03 +- 1.7e-03 | 5.92e-03 | 1.00 | 2.12e-02 | 5 |
| n1_m1_f0__rep1__adaptive_weights | 4.48e-03 +- 1.4e-03 | 4.70e-03 | 1.00 | 1.91e-02 | 5 |
| n1_m1_f0__rep1__vanilla | 4.40e-03 +- 1.4e-03 | 3.80e-03 | 1.00 | 2.09e-02 | 5 |

Branch difficulty (mean L2RE ascending): (0,0) (2.2e-03) < (1,1) (4.5e-03) < (1,0) (1.7e-02)

## 4. Inverse problem (Phase 7)

- `n1_m1_f0`: (g0, T2, alpha) sensitivity rank = **2 of 3** (they enter only via c = g0*T2^2 and ell = (g0-alpha)/2 — separate recovery is impossible); reduced-parameter Jacobian rank 7/8, condition number 7.6e+10.
- `n1_m0_f0`: (g0, T2, alpha) sensitivity rank = **2 of 3** (they enter only via c = g0*T2^2 and ell = (g0-alpha)/2 — separate recovery is impossible); reduced-parameter Jacobian rank 7/8, condition number 1.1e+11.
- 108/108 recovery runs done (results/inverse_pilot/, figures in reports/figures/inverse_pilot/).

| stage | param | noise | median rel err (abs err if true=0) | n |
|---|---|---|---|---|
| dispersion | beta2 | 0.0 | 9.06e-03 | 6 |
| dispersion | beta2 | 0.01 | 6.89e-03 | 6 |
| dispersion | beta2 | 0.05 | 1.82e-02 | 6 |
| dispersion | beta3 | 0.0 | 2.61e-02 | 6 |
| dispersion | beta3 | 0.01 | 2.05e-02 | 6 |
| dispersion | beta3 | 0.05 | 7.21e-02 | 6 |
| dispersion | beta4 | 0.0 | 4.32e-02 | 6 |
| dispersion | beta4 | 0.01 | 6.59e-02 | 6 |
| dispersion | beta4 | 0.05 | 7.69e-02 | 6 |
| gainloss | c | 0.0 | 1.46e-02 | 18 |
| gainloss | c | 0.01 | 1.52e-02 | 18 |
| gainloss | c | 0.05 | 4.58e-02 | 18 |
| gainloss | ell | 0.0 | 3.20e-03 | 18 |
| gainloss | ell | 0.01 | 3.86e-03 | 18 |
| gainloss | ell | 0.05 | 1.90e-02 | 18 |
| joint | Gamma | 0.0 | 1.46e-02 | 6 |
| joint | Gamma | 0.01 | 1.46e-02 | 6 |
| joint | Gamma | 0.05 | 1.46e-02 | 6 |
| joint | alpha2 | 0.0 | 1.46e-01 | 6 |
| joint | alpha2 | 0.01 | 1.45e-01 | 6 |
| joint | alpha2 | 0.05 | 1.46e-01 | 6 |
| joint | c | 0.0 | 1.24e-01 | 6 |
| joint | c | 0.01 | 1.18e-01 | 6 |
| joint | c | 0.05 | 1.01e-01 | 6 |
| joint | ell | 0.0 | 1.37e-01 | 6 |
| joint | ell | 0.01 | 1.36e-01 | 6 |
| joint | ell | 0.05 | 1.60e-01 | 6 |
| nonlinear | Gamma | 0.0 | 2.31e-02 | 6 |
| nonlinear | Gamma | 0.01 | 2.39e-02 | 6 |
| nonlinear | Gamma | 0.05 | 3.03e-02 | 6 |
| nonlinear | alpha2 | 0.0 | 1.51e-02 | 6 |
| nonlinear | alpha2 | 0.01 | 1.22e-02 | 6 |
| nonlinear | alpha2 | 0.05 | 2.40e-02 | 6 |
| nonlinear | gamma | 0.0 | 1.50e-03 | 6 |
| nonlinear | gamma | 0.01 | 7.30e-04 | 6 |
| nonlinear | gamma | 0.05 | 2.52e-03 | 6 |

## 5. Ablations (Phase 6)

- saturation::Gamma0: mean L2RE 6.14e-04 +- 2.1e-04 (n=3)
- saturation::GammaPos: mean L2RE 8.81e-04 +- 3.0e-04 (n=3)
- fourth_order::beta4_0: mean L2RE 9.19e-04 +- 3.5e-04 (n=3)
- fourth_order::beta4_pos: mean L2RE 1.11e-03 +- 6.7e-04 (n=3)
- Incompatible requested ablations documented in reports/ablations.md (conservative closures do not exist among verified branches; Gamma>0 and beta4!=0 are closure-forbidden off the constant branch).

## 6. Conditioned cross-branch PINN (Phase 8)

- seed 0:
  - interpolation L2RE: {'n0_m0_f0': 0.054725730995417575, 'n1_m0_f0': 0.24765072999109752, 'n1_m1_f0': 0.06315686250442767}
  - extrapolation L2RE: {'n0_m0_f0': 0.12179601385865912, 'n1_m0_f0': 0.9622436104358344, 'n1_m1_f0': 0.8141446850587059}
  - LOBO n0_m0_f0: zero-shot {'rep0': 0.826535547800969, 'rep1': 0.8880133967248187}, fine-tune(1%) {'rep0': 0.0066384251790510655, 'rep1': 0.009848885245360718}, fine-tune(5%) {'rep0': 0.005833388772365701, 'rep1': 0.006645800374707377}, scratch-same-budget {'rep0': 0.010126141071229501, 'rep1': 0.006307070534832212}
  - LOBO n1_m0_f0: zero-shot {'rep0': 1.2699008217109933, 'rep1': 1.5950956951030455}, fine-tune(1%) {'rep0': 0.054669711667925044, 'rep1': 0.048696416010111196}, fine-tune(5%) {'rep0': 0.04514531712988278, 'rep1': 0.050051832190559475}, scratch-same-budget {'rep0': 0.05347242881054626, 'rep1': 0.13690423581169428}
  - LOBO n1_m1_f0: zero-shot {'rep0': 1.0830521462913498, 'rep1': 1.3088147573819513}, fine-tune(1%) {'rep0': 0.01882569132981764, 'rep1': 0.04540977690875274}, fine-tune(5%) {'rep0': 0.01550251638980788, 'rep1': 0.03346220871324501}, scratch-same-budget {'rep0': 0.06536282598830137, 'rep1': 0.047568008570461374}
- seed 1:
  - interpolation L2RE: {'n0_m0_f0': 0.06649824255746363, 'n1_m0_f0': 0.1640920962692794, 'n1_m1_f0': 0.18965826940462352}
  - extrapolation L2RE: {'n0_m0_f0': 0.47007429176765336, 'n1_m0_f0': 0.9312953323227879, 'n1_m1_f0': 0.7276376107217298}
  - LOBO n0_m0_f0: zero-shot {'rep0': 0.44261168483630237, 'rep1': 0.685337068022348}, fine-tune(1%) {'rep0': 0.009524408399079578, 'rep1': 0.007020246461512905}, fine-tune(5%) {'rep0': 0.0069379989624596804, 'rep1': 0.005545924722340453}, scratch-same-budget {'rep0': 0.019135344388698166, 'rep1': 0.008529495221008663}
  - LOBO n1_m0_f0: zero-shot {'rep0': 1.2091752789474506, 'rep1': 1.2585225528306763}, fine-tune(1%) {'rep0': 0.05148080242038255, 'rep1': 0.11189032377387272}, fine-tune(5%) {'rep0': 0.03578424177750008, 'rep1': 0.06181036020203956}, scratch-same-budget {'rep0': 0.09545141273268608, 'rep1': 0.10914284271948405}
  - LOBO n1_m1_f0: zero-shot {'rep0': 1.1167036799362609, 'rep1': 1.06970516647508}, fine-tune(1%) {'rep0': 0.04810604866101327, 'rep1': 0.12883694205696297}, fine-tune(5%) {'rep0': 0.02495100470054932, 'rep1': 0.028002620374385313}, scratch-same-budget {'rep0': 0.051680432913549866, 'rep1': 0.06758343707530433}

## 7. Provenance and epistemics

- **Mathematically proven (symbolic zero + 50-digit numeric):** every family in data/verified_branches/branch_catalog.json.
- **Unsupported candidates:** all remaining proposed pairs (data/verified_branches/rejected_pairs.csv, with reasons).
- **PINN successes/failures:** per-run result.json files under results/; success = complex L2RE < 1e-2 AND independent residual reported; raw errors always retained.
- **Proposed but not executed:** any stage marked NOT RUN above; each carries its exact resume command. All stages are resumable and skip completed jobs unless --force is passed.

Commands: smoke `python scripts/run_full_benchmark.py --mode smoke`; pilot `python scripts/run_full_benchmark.py --mode pilot`; full `python scripts/run_full_benchmark.py --mode full`.