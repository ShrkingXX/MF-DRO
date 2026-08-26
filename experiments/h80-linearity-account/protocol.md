# H80 — Does the linearity account predict out of sample?

**CONFIRMATORY.** Locked before any h80 number exists.

## The account being tested

Pinning `rho` to the OLS slope of `f_H` on `f_L` helps where that relation is
genuinely linear and hurts where it is not, and the cost appears in the
**variance** rather than the mean: `delta` shrinks when `rho` is right, and grows
when `rho` is imposed on a relation that is not linear.

Derived from two benchmarks:

| benchmark | R² | var_delta ratio (RHOTRUE/fitted) | RHOTRUE regret |
|---|---|---|---|
| Borehole | 1.00000 | **0.72x** (shrank) | **+1.9%** |
| Hartmann | 0.85574 | **1.24x** (grew) | **−16.6%** |

Two points define any line. This tests the account where it was **not** fitted.

## Design — two independent tests

**A. Held-out benchmark.** Currin 2D has R² = **0.99471**, close to Borehole's
1.00000 and far from Hartmann's 0.85574. It was never used to build the account
and RHOTRUE was never run on it. Fit the KO model with `rho` fitted vs pinned at
Currin's measured slope 1.0104, seeds 44/46/48, and measure the `var_delta` ratio.

**B. Synthetic R² sweep.** Construct `f_H = s*f_L + a*g(x)` with `g` a fixed
nonlinear perturbation and `a` swept so R² runs from ~1.00 down to ~0.80. At each
level, fit KO with `rho` fitted vs pinned at that level's own OLS slope, and
record the `var_delta` ratio. This tests the *mechanism* across a continuum
rather than at isolated points.

Both are KO fits only — no BO runs. Seconds of compute, negligible beside h78.

## Locked predictions

1. **PRIMARY (A).** On Currin, `var_delta` **shrinks**: ratio **< 1.0**.
2. **SECONDARY (A).** The ratio is nearer Borehole's 0.72 than Hartmann's 1.24 —
   concretely **< 0.95**.
3. **PRIMARY (B).** Across the sweep the `var_delta` ratio is **monotonically
   non-decreasing as R² falls**, and **crosses 1.0** somewhere between R²=1.00
   and R²=0.80. That crossing is the account's central claim: a threshold in
   linearity below which pinning rho stops helping and starts hurting.
4. **NULL.** Currin's ratio **> 1.0**, or the sweep shows no monotone trend.
   Then the account does not generalise beyond the two benchmarks it was built
   from and must be reported as a description of those two, not a mechanism.

## What this cannot settle

It tests the *variance* mechanism, not the regret consequence — Currin does not
discriminate on regret (every non-degenerate method finishes inside 0.6%), so no
regret claim follows from A. The synthetic sweep uses one perturbation family and
one LF function; a different nonlinearity could behave differently.
