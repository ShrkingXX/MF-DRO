# H80 result — the account predicts out of sample; my monotonicity bar was mis-specified

**CONFIRMATORY** against `protocol.md`.

## Verdict

| prediction | bar | result | verdict |
|---|---|---|---|
| PRIMARY (A) | Currin `var_delta` ratio < 1.0 | **0.311** | **MET** |
| SECONDARY (A) | ratio < 0.95 | 0.311 | **MET** |
| PRIMARY (B) | monotone as R² falls **and** crosses 1.0 | crosses **yes**, monotone **no** | **NOT MET** |
| NULL | Currin > 1.0 **or** no monotone trend | fired on the second clause | see below |

## A — the held-out benchmark behaves as predicted

Currin (R² = 0.99471) was never used to build the account and RHOTRUE was never
run on it. Predicted: `delta` shrinks, as on Borehole. Measured **0.311**.

| benchmark | R² | var_delta ratio |
|---|---|---|
| Borehole | 1.00000 | 0.72 |
| **Currin (held out)** | **0.99471** | **0.311** |
| Hartmann | 0.85574 | 1.24 |

**Caveat, and it is a real one:** the per-seed values are **0.884 / 0.029 /
0.021**. The mean is dominated by two very small numbers and n=3. The *sign* is
consistent across all three seeds (all < 1.0), which is what the prediction was
about, but the magnitude is not reliable.

## B — the threshold is real; the monotonicity requirement was not

| a | R² | OLS slope | var_delta ratio |
|---|---|---|---|
| 0.00 | 1.00000 | 1.1500 | 0.265 |
| 0.05 | 0.99553 | 1.1525 | 0.270 |
| 0.10 | 0.98245 | 1.1550 | 0.331 |
| 0.20 | **0.93383** | 1.1600 | **2.035** <- crosses 1.0 |
| 0.35 | 0.82357 | 1.1674 | 4.228 |
| 0.55 | 0.65786 | 1.1774 | 3.899 |
| 0.80 | 0.48137 | 1.1899 | 3.361 |

**The account's central claim holds: there is a linearity threshold, and it sits
between R² = 0.982 and R² = 0.934.** Above it, pinning `rho` shrinks `delta`;
below it, `delta` grows several-fold.

That threshold **correctly sorts all three real benchmarks**:

| benchmark | R² | side of threshold | predicted | observed |
|---|---|---|---|---|
| Borehole | 1.00000 | above | helps | **+1.9%** |
| Currin | 0.99471 | above | helps | delta 0.311 (no regret test — Currin does not discriminate) |
| Hartmann | 0.85574 | **below** | hurts | **−16.6%** |

**What failed is my bar, not the account.** I required the ratio to be monotone
across the *whole* sweep. It rises to a peak at R² ≈ 0.82 and then declines
(4.228 -> 3.899 -> 3.361). That is sensible in hindsight: once the relation is
badly nonlinear, `delta` must model nearly everything *regardless* of `rho`, so
the marginal damage from pinning saturates. The account only ever implied a
threshold; monotonicity everywhere was my addition and it was wrong.

## The NULL fired on a mis-specified compound condition

The NULL read "Currin's ratio > 1.0, **or** the sweep shows no monotone trend".
The second clause fired. Its stated conclusion — "the account does not generalise
and must be reported as a description of two benchmarks" — **does not follow**:
the held-out benchmark was predicted correctly and the threshold is present and
correctly placed.

This is the same defect as h79b's NULL: a compound condition where one clause
tests the claim and the other tests an incidental property, so firing does not
identify which. **A NULL branch should be as carefully specified as the PRIMARY,
because it is the branch that retracts things.**

## What this establishes

Pinning `rho` to the OLS slope helps when `f_H` is linear in `f_L` above roughly
R² ≈ 0.95, and hurts below it, with the damage appearing in `var_delta` and hence
in the acquisition's uncertainty rather than in prediction accuracy. **This is
checkable on any new benchmark pair by computing one R² before running
anything.**

## What this cannot settle

n=3 on Currin with a wide per-seed spread. One perturbation family and one LF
function in the sweep. It tests the variance mechanism, not regret — no regret
claim follows from Currin, which does not discriminate.
