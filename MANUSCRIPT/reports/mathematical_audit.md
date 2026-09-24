# Mathematical audit of the 49-branch catalogue

Generated automatically by `scripts/run_audit.py`. Tolerances: max|F| < 1e-10, relative RMS < 1e-10 at 50-digit precision, on a dense xi-grid plus random (z,t) points, using the ORIGINAL complex PDE.

## 1. Verification of the paper's coefficient algebra

- Independent re-derivation of the grouped coefficients C1-C9 (paper Eqs. 17-25), keeping n and m symbolic: **MATCH** for all nine offsets. The paper's Section 5 algebra is internally correct.

## 2. The classification issue (paper Section 7)

- Solving n+r = 3n+s and m+r = 3m+s simultaneously gives n = m = (r-s)/2: **Eq. (28) supports only diagonal pairs (n = m)** (verified: True).
- The paper's Section 7.3 nevertheless forms the Cartesian product of the seven values in each coordinate, adding **42 off-diagonal pairs with no mathematical justification**.
- Deeper: the argument treats distinct J_{p,q} as linearly independent, but J_{2,0} = J_{0,0} - J_{2,2} (sech^2 = 1 - tanh^2). Index matching is neither necessary nor sufficient for closure; this audit instead uses an exact certificate in the T = tanh(xi) basis, in which {T^j} and {W T^j, W = sech(xi)} ARE linearly independent families.
- Consequence (see Section 4 below): some off-diagonal pairs the paper's own Eq. (28) would exclude DO close (e.g. the sech branch (1,0)), while most 'combinatorially supported' diagonal pairs are vacuous. The combinatorial frequency table (paper Table 2) has no bearing on existence.

## 3. The residual split (paper Eqs. 37-38)

The complex residual was split into real/imaginary parts symbolically and proven against F_complex = F_u + i F_v on random smooth fields (tests/test_symbolic_math.py). Discrepancies against the paper's printed Eqs. (37)-(38):
- Paper Eq. (37) places -(beta3/6) v_ttt in F_u; the correct real part of (beta3/6)A_ttt is +(beta3/6) u_ttt.
- Paper Eq. (37) places -(beta4/24) u_tttt in F_u; the correct real part of (i beta4/24)A_tttt is -(beta4/24) v_tttt.
- Paper Eq. (38) places +(beta3/6) u_ttt in F_v; correct is +(beta3/6) v_ttt.
- Paper Eq. (38) places -(beta4/24) v_tttt in F_v; correct is +(beta4/24) u_tttt.
- Paper Eqs. (37)-(38) keep the complex coefficient (gamma + i alpha2/2) outside Re[.]/Im[.], which is ill-formed for real residuals; the correct split is -(alpha2/2)Q u - gamma Q v in F_u and -(alpha2/2)Q v + gamma Q u in F_v with Q = |A|^2/(1+Gamma|A|^2).
- Abstract and Section 2 say 'eight physical parameters' but list nine: beta2, beta3, beta4, g0, T2, alpha, alpha2, gamma, Gamma (Section 9.3 itself says 'up to nine independent physical coefficients').
- Paper Eq. (36) prints 'C6 J_{2,2} + C7 J_{2,2}'; for (n,m)=(1,1) the C7 term multiplies J_{3,3} = tanh^3, not J_{2,2}.

## 4. Branch-by-branch closure audit

Of the 49 audited pairs, **9 pairs yield at least one verified, nontrivial solution family** (9 families total); 40 pairs are rejected.

### Accepted families

| branch | (n,m) | tier | group | free params | constraints | max|F| | rel RMS |
|---|---|---|---|---|---|---|---|
| nneg1_mneg1_f0 | (-1,-1) | B | stress | lambda2, asq, c | lambda1=0; beta3=0; ell=-c*lambda2**2; Gamma=0; beta4=0; beta2=0; gamma=0; alpha2=-2*c*lambda2**2/asq | 2.1e-50 | 2.4e-51 |
| nneg1_m0_f1 | (-1,0) | B | stress | lambda2, asq, beta3, beta4, c, Gamma | lambda1=-beta3*lambda2**3/6; ell=c*lambda2**2/2; alpha2=0; beta2=-beta4*lambda2**2/12; gamma=0 | 2.1e-50 | 4.8e-52 |
| nneg1_m1_f1 | (-1,1) | B | stress | lambda2, asq, beta3, beta4, c, Gamma | lambda1=-2*beta3*lambda2**3/3; ell=2*c*lambda2**2; alpha2=0; beta2=-beta4*lambda2**2/3; gamma=0 | 3.8e-48 | 2.5e-51 |
| n0_mneg1_f0 | (0,-1) | B | stress | lambda2, asq, c | Gamma=0; beta4=0; beta3=0; lambda1=0; gamma=0; beta2=0; alpha2=-2*c*lambda2**2/asq; ell=c*lambda2**2/2 | 2.1e-50 | 3.7e-51 |
| n0_m0_f0 | (0,0) | B | main | lambda1, lambda2, asq, beta2, beta3, beta4, c, alpha2, Gamma | gamma=0; ell=alpha2*asq/(2*(Gamma*asq + 1)) | 0.0e+00 | 0.0e+00 |
| n0_m1_f1 | (0,1) | B | stress | lambda2, asq, beta3, beta4, c, Gamma | ell=c*lambda2**2/2; alpha2=0; lambda1=-beta3*lambda2**3/6; beta2=-beta4*lambda2**2/12; gamma=0 | 2.0e-50 | 4.7e-52 |
| n1_mneg1_f0 | (1,-1) | B | stress | lambda2, asq, c | Gamma=0; beta4=0; gamma=0; beta2=0; beta3=0; lambda1=0; ell=2*c*lambda2**2; alpha2=-2*c*lambda2**2/asq | 4.3e-50 | 8.4e-51 |
| n1_m0_f0 | (1,0) | B | main | lambda2, asq, c | Gamma=0; beta4=0; beta3=0; lambda1=0; gamma=0; beta2=0; alpha2=2*c*lambda2**2/asq; ell=c*lambda2**2/2 | 1.3e-51 | 5.9e-52 |
| n1_m1_f0 | (1,1) | B | main | lambda2, asq, c | Gamma=0; beta4=0; beta2=0; gamma=0; beta3=0; lambda1=0; ell=-c*lambda2**2; alpha2=-2*c*lambda2**2/asq | 2.0e-51 | 6.0e-52 |

### Rejected pairs

| (n,m) | reason |
|---|---|
| (-3/2,-3/2) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (-3/2,-1) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (-3/2,-1/2) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (-3/2,0) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (-3/2,1/2) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (-3/2,1) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (-3/2,3/2) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (-1,-3/2) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (-1,-1/2) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (-1,1/2) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (-1,3/2) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (-1/2,-3/2) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (-1/2,-1) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (-1/2,-1/2) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (-1/2,0) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (-1/2,1/2) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (-1/2,1) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (-1/2,3/2) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (0,-3/2) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (0,-1/2) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (0,1/2) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (0,3/2) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (1/2,-3/2) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (1/2,-1) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (1/2,-1/2) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (1/2,0) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (1/2,1/2) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (1/2,1) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (1/2,3/2) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (1,-3/2) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (1,-1/2) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (1,1/2) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (1,3/2) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (3/2,-3/2) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (3/2,-1) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (3/2,-1/2) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (3/2,0) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (3/2,1/2) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (3/2,1) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |
| (3/2,3/2) | lambda2 = 0 (xi has no t-dependence) | vacuous solution: at most one PDE term is active (governing equation trivializes on this family) |

## 5. Tier definitions and headline verdict

- Tier A (full model: 4th-order dispersion + saturation + gain/loss all active): **0**
- Tier B (nontrivial reduced model): **9**
- Tier C / rejected: the remainder of the 49 proposed pairs.

**The paper's central catalogue claim — 49 exact-solution branches — is not supported.** Only the families listed above are genuine, and every one of them requires several physical coefficients to vanish (reduced models). No branch supports the full nine-coefficient model under the paper's constant-phase real-amplitude ansatz.
