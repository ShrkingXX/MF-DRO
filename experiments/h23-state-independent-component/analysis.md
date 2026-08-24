# H23 — the asserted explanation is CONFIRMED, and now quantified

Decomposing the scoring coefficients over the 10 distinct τ=0 states as
`w(s) = w̄ + δ(s)`:

| | result |
|---|---|
| **PRED 1** — `w̄` alone reproduces the full argmax | **12/12 pools — PASS** |
| **PRED 2** — median margin / max δ-contribution | **77.16 — PASS** (bar was 5) |
| **SECONDARY** — `bias_head` changes the argmax | **0/12 — CONFIRMED** |
| `‖w̄‖` | 1.5980 |
| mean `‖δ(s)‖` | 0.002062 |
| **`‖δ‖ / ‖w̄‖`** | **0.00129 (0.13%)** |

## What is now established rather than inferred

The paper's sentence — "the ranking is dominated by a state-independent
component of `w`" — is exactly right, and the magnitudes are stark:

- **The state moves the coefficient vector by 0.13% of its length.**
- **Scoring with the state-independent part alone gives the identical decision
  on every pool tested (12/12).**
- The top-1/top-2 margin exceeds anything `δ(s)` can contribute by a **median
  factor of 77**.

So MF-DRO's learned policy is, to within a tenth of a percent, a **fixed linear
acquisition function**. It is not a policy that conditions weakly; it is a
constant acquisition rule with a negligible state-dependent perturbation
superimposed.

## A structural point worth stating

`bias_head(h)` adds the **same constant to every candidate**, so it cannot change
an argmax under any circumstances — confirmed empirically at 0/12. One of the two
state-dependent quantities the architecture exposes at the decision is therefore
**decision-irrelevant by construction**, independent of anything training does.

## Consistency with H22, and its limit

The per-pool ratios span **4.9 to 136.5**. Two pools are near-marginal, which is
qualitatively consistent with H22 finding exactly one isolated 1/12 blip at
λ=60 and nothing sustained through λ=100. We do **not** push this to a
quantitative prediction: H22 scaled the *state input* while this ratio is defined
on δ directly, and the map between them is only approximately linear.

## Method note

This experiment exists because the claim it tests was **inferred, not measured**,
and three earlier inferences in this project (H5's state swap, H18's diversity
claim, H13's decomposition rule) failed re-measurement. The inference happened to
be right this time. Checking it cost one probe.
