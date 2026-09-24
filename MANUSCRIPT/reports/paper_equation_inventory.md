# Paper equation inventory — paper-5.pdf

Source: `../paper-5.pdf` ("Exact Traveling-Wave Solution Families of a Higher-Order
Gain–Loss Nonlinear Schrödinger-Type Equation via the iB-Function (BDKm) Ansatz",
dated 2026-08-20, 17 pp.). The task specified `PAPER_PATH = ./paper-5(1).pdf`,
which does not exist; `paper-5.pdf` is the only PDF in the working directory and
was selected. The PDF was not modified.

## 1. Governing PDE (paper Eq. (1))

A_z + (i/2)(β₂ + i g₀T₂²) A_tt + (β₃/6) A_ttt + (i β₄/24) A_tttt
    + i(γ + i α₂/2) |A|²A / (1 + Γ|A|²) + (g₀ − α)/2 · A = 0

with complex envelope A(z,t); z = propagation coordinate, t = retarded time.

## 2. Parameters and roles

| symbol | role | notes |
|---|---|---|
| β₂ | group-velocity dispersion (GVD) | real |
| β₃ | third-order dispersion (TOD) | real |
| β₄ | fourth-order dispersion (FOD) | real |
| g₀ | small-signal gain | ≥ 0 physically |
| T₂ | gain-medium dipole relaxation time (gain bandwidth) | ≥ 0 |
| α  | linear loss | ≥ 0 physically |
| α₂ | two-photon-type loss | sign not restricted by the paper |
| γ  | Kerr self-phase modulation | real |
| Γ  | nonlinearity saturation | ≥ 0 |

That is **nine** physical coefficients. The abstract and Section 2 call these
"the eight physical parameters (β₂, β₃, β₄, g₀, T₂, α, α₂, γ, Γ)" — a count of
eight over a list of nine. Section 9.3 says "up to nine independent physical
coefficients". **Inconsistency confirmed: the model has nine coefficients.**
Note additionally that g₀, T₂, α enter the traveling-wave algebra only through
the two combinations c = g₀T₂² and ℓ = (g₀−α)/2 (identifiability consequence:
they are not separately recoverable from a single solution field).

## 3. Traveling-wave transformation (paper Eqs. (2)–(5))

ξ = λ₁ z + λ₂ t, A(z,t) = A(ξ), λ₁, λ₂ real, giving

λ₁A′ + (i/2)(β₂ + i g₀T₂²)λ₂²A″ + (β₃/6)λ₂³A‴ + (iβ₄/24)λ₂⁴A⁗
    + i(γ + iα₂/2)|A|²A/(1 + Γ|A|²) + (g₀−α)/2 · A = 0.

## 4. Ansatz (paper Eqs. (6)–(7))

A(ξ) = a J_{n,m}(ξ), J_{n,m}(ξ) = sinhᵐ(ξ)/coshⁿ(ξ),
with differentiation rule dJ_{n,m}/dξ = m J_{n−1,m−1} − n J_{n+1,m+1}.
Section 5 takes **a real** ("without loss of generality for the modulus") and
uses |A|²A = a³J_{3n,3m} — i.e., a constant-phase real-amplitude convention.
This "without loss of generality" is FALSE in general: a z-dependent carrier
phase e^{iκz} cannot be gauged away in this non-autonomous-in-phase equation,
and its absence is what kills most nonlinear closures (see mathematical_audit.md).

## 5. Coefficient equations (paper Eqs. (8)–(25))

Derivatives A′..A⁗ expand over offsets J_{n+r,m+r}, r = −4..4, with grouped
coefficients C₁..C₉ (paper Eqs. 17–25), Ξ := β₂ + i g₀T₂². Our independent
re-derivation reproduces all nine C_k exactly for symbolic (n,m)
(`tests/test_symbolic_math.py::TestOffsetCoefficients`), and spot-checks of the
printed appendix column (n,m) = (1,1) and the n = 3/2 rows of C₈, C₉ agree.

## 6. The "Range equation" (paper Eq. (26))

Multiplying by (1 + Γa²J_{2n,2m}) gives the 19-term fraction-free equation:
block 1 = Σₖ C_k J_{n+r,m+r}; block 2 = Γa² Σₖ C_k J_{3n+r,3m+r};
nonlinear term i(γ + iα₂/2)a² J_{3n,3m}.

## 7. Classification and the 49 pairs (paper Section 7)

Index matching J_{n+r,m+r} = J_{3n+s,3m+s} requires n+r = 3n+s AND m+r = 3m+s,
giving n = (r−s)/2 and m = (r−s)/2, hence **n = m** (paper Eq. (28) states this
itself). The paper then restricts to the seven values {−3/2, −1, −1/2, 0, 1/2,
1, 3/2} but takes the **Cartesian product** F_R = {…}² of 49 ordered pairs
(Eq. (31)) — the 42 off-diagonal pairs (n ≠ m) contradict the paper's own
Eq. (28) and receive no other justification. In addition, the whole
index-matching frame assumes distinct J_{p,q} are linearly independent, which
is false (J_{2,0} = J_{0,0} − J_{2,2}). See mathematical_audit.md §2.

The combinatorial frequency table (paper Table 2, N(q) = 9 − 2|q|) is an index-
multiplicity count; the paper itself concedes (Discussion) it "measures
index-matching multiplicity, not solution existence".

## 8. Proposed PINN losses (paper Section 9)

- Forward loss (Eq. 39): MSE on initial data at z₀ + MSE on Dirichlet data at
  t = ±L + MSE of (F_u² + F_v²) at collocation points.
- Inverse loss (Eq. 40): MSE on sampled field data + physics residual, with the
  physical coefficients promoted to trainable scalars.
- Residual split (Eqs. 37–38): **contains errors** — β₃ and β₄ terms are
  attached to the wrong components (β₃: v_ttt instead of u_ttt in F_u, u_ttt
  instead of v_ttt in F_v; β₄: u_tttt instead of −v_tttt in F_u, −v_tttt
  instead of +u_tttt in F_v), and a complex coefficient (γ + iα₂/2) is left
  outside Re[·]/Im[·], which is ill-formed. The corrected split is derived and
  unit-tested in `src/pinn_benchmark/equations.py`.

## 9. Worked branch (paper Section 8, (n,m) = (1,1))

The paper says "C₄ = 0 fixes λ₁ = β₃λ₂³/3" — but C₆ = −λ₁ + (4β₃/3)λ₂³ and
C₈ = −β₃λ₂³ must vanish simultaneously, forcing β₃λ₂³ = 0 (hence λ₁ = 0), and
C₉ = iβ₄λ₂⁴ = 0 forces β₄ = 0. The paper does not carry the closure through and
does not notice that the branch survives only in a heavily reduced model.
Eq. (36) also misprints "C₆J₂,₂ + C₇J₂,₂" (C₇ multiplies J₃,₃ = tanh³).

## 10. Ambiguities / missing information

1. "Eight physical parameters" vs. the nine listed (see §2).
2. No closure conditions are actually solved for any branch; the "49 solution
   branches" claim (abstract) is asserted from index bookkeeping alone. The
   Discussion admits "a systematic audit … is needed before the full catalog
   can be used as-is".
3. Off-diagonal pairs are unjustified (paper's own Eq. (28)).
4. Fractional powers sinhᵐ with m = ±1/2, ±3/2 for ξ < 0 (sinh < 0) are never
   assigned a branch convention; the profiles are complex or non-smooth at
   ξ = 0 depending on convention. Never addressed in the paper.
5. Appendix Tables 5/6 are placed under the section header "A.1 n = −3/2"
   while sections A.2, A.4–A.7 are empty headers; the n ∈ {0, 1/2, 1, 3/2}
   tables appear after the references/data-availability text (layout defect).
6. Paper Eqs. (37)–(38) errors (see §8).
7. a is declared "generally complex" (Eq. 6) but treated as real from
   Section 5 on, silently dropping every solution with a carrier phase.
8. λ₂ = 0 and a = 0 degenerations are never excluded by the paper.
9. No domain of validity ((z,t) window, singularity structure) is given for
   any branch; coth-type profiles are singular at ξ = 0.
