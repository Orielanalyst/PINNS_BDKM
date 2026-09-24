# Independent numerical solver validation

Primary scheme: 6th-order finite differences in t with three ghost
nodes per side filled from the exact analytic solution (Dirichlet-type
boundary strip - NOT periodic; kinks are never given periodic BCs),
integrated in z with scipy DOP853. Cross-check scheme for
periodic-compatible (localized/constant) profiles: Fourier
pseudospectral + ETDRK4 on a 3x padded window.

## n0_m0_f0

| scheme | nt | rtol/steps | final L2RE | final max err | runtime (s) |
|---|---|---|---|---|---|
| fd_mol | 65 | 1e-08 | 9.95e-12 | 1.45e-11 | 0.04 |
| fd_mol | 65 | 1e-10 | 1.63e-13 | 2.43e-13 | 0.04 |
| fd_mol | 129 | 1e-08 | 8.38e-12 | 1.19e-11 | 0.73 |
| fd_mol | 129 | 1e-10 | 2.53e-13 | 4.48e-13 | 0.74 |
| fd_mol | 257 | 1e-08 | 5.27e-11 | 7.48e-11 | 13.47 |
| fd_mol | 257 | 1e-10 | 1.20e-12 | 1.58e-12 | 13.42 |
| fourier_etdrk4 | 256 | 400 steps | 8.58e-18 | 8.58e-18 | 0.06 |

Observed spatial convergence order: ['-0.64', '-2.25'] (6th-order stencils).

## n1_m0_f0

| scheme | nt | rtol/steps | final L2RE | final max err | runtime (s) |
|---|---|---|---|---|---|
| fd_mol | 65 | 1e-08 | 2.28e-07 | 4.41e-07 | 0.01 |
| fd_mol | 65 | 1e-10 | 2.28e-07 | 4.41e-07 | 0.02 |
| fd_mol | 129 | 1e-08 | 7.59e-09 | 1.23e-08 | 0.05 |
| fd_mol | 129 | 1e-10 | 3.70e-09 | 7.18e-09 | 0.05 |
| fd_mol | 257 | 1e-08 | 4.63e-10 | 4.98e-10 | 0.20 |
| fd_mol | 257 | 1e-10 | 5.84e-11 | 1.15e-10 | 0.21 |
| fourier_etdrk4 | 256 | 400 steps | 2.30e-06 | 5.25e-06 | 0.06 |

Observed spatial convergence order: ['6.02', '6.02'] (6th-order stencils).

## n1_m1_f0

| scheme | nt | rtol/steps | final L2RE | final max err | runtime (s) |
|---|---|---|---|---|---|
| fd_mol | 65 | 1e-08 | 1.12e-07 | 3.08e-07 | 0.01 |
| fd_mol | 65 | 1e-10 | 1.12e-07 | 3.08e-07 | 0.02 |
| fd_mol | 129 | 1e-08 | 1.82e-09 | 5.23e-09 | 0.05 |
| fd_mol | 129 | 1e-10 | 1.82e-09 | 5.23e-09 | 0.05 |
| fd_mol | 257 | 1e-08 | 3.66e-09 | 4.97e-09 | 0.21 |
| fd_mol | 257 | 1e-10 | 7.54e-11 | 1.04e-10 | 0.21 |

Observed spatial convergence order: ['6.01', '4.62'] (6th-order stencils).
