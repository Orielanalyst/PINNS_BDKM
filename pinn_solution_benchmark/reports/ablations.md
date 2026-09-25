# Physical and methodological ablations

Matched closure-respecting pairs (constant branch, one free parameter changed per pair):

| ablation | arm | mean L2RE | std | n |
|---|---|---|---|---|
| fourth_order | beta4_0 | 9.19e-04 | 3.5e-04 | 3 |
| fourth_order | beta4_pos | 1.11e-03 | 6.7e-04 | 3 |
| saturation | Gamma0 | 6.14e-04 | 2.1e-04 | 3 |
| saturation | GammaPos | 8.81e-04 | 3.0e-04 | 3 |

## Statistical comparisons

Welch two-sample t-tests and exact two-sided Mann-Whitney U tests are reported as sensitivity analyses. Shapiro-Wilk values are diagnostics only because n=3 per arm.

- **fourth_order** (beta4_0 vs beta4_pos): Welch p=0.6912; Mann-Whitney p=1.
- **saturation** (Gamma0 vs GammaPos): Welch p=0.2769; Mann-Whitney p=0.4.

## Power analysis

At n=3 per arm, alpha=0.05, and target power=0.80, the two-sided normal-approximation standardized minimum detectable effect is d=2.287.
This is a planning calculation, not evidence that the observed ablation has that effect size.

## Requested ablations that are incompatible with the verified closures

- **saturation on n1_m0_f0 / n1_m1_f0**: closure forces Gamma = 0: any Gamma > 0 collapses the family to gamma = alpha2 = 0 and a vacuous equation
- **fourth_order on n1_m0_f0 / n1_m1_f0**: closure forces beta4 = 0
- **conservative (ell = 0, alpha2 = 0)**: no verified branch admits it: ell = 0 forces c*lambda2^2 = 0 hence a fully vacuous equation; the surviving solution set is intrinsically dissipative

Fixed-vs-adaptive weighting and uniform-vs-residual-adaptive collocation are covered by the pilot matrix (results/forward_pilot/aggregate.json; see final report).