# Physical and methodological ablations

Matched closure-respecting pairs (constant branch, one free parameter changed per pair):

| ablation | arm | mean L2RE | std | n |
|---|---|---|---|---|
| saturation | Gamma0 | 6.14e-04 | 2.1e-04 | 3 |
| saturation | GammaPos | 8.81e-04 | 3.0e-04 | 3 |
| fourth_order | beta4_0 | 9.19e-04 | 3.5e-04 | 3 |
| fourth_order | beta4_pos | 1.11e-03 | 6.7e-04 | 3 |

## Requested ablations that are incompatible with the verified closures

- **saturation on n1_m0_f0 / n1_m1_f0**: closure forces Gamma = 0: any Gamma > 0 collapses the family to gamma = alpha2 = 0 and a vacuous equation
- **fourth_order on n1_m0_f0 / n1_m1_f0**: closure forces beta4 = 0
- **conservative (ell = 0, alpha2 = 0)**: no verified branch admits it: ell = 0 forces c*lambda2^2 = 0 hence a fully vacuous equation; the surviving solution set is intrinsically dissipative

Fixed-vs-adaptive weighting and uniform-vs-residual-adaptive collocation are covered by the pilot matrix (results/forward_pilot/aggregate.json; see final report).