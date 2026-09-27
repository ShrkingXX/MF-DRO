# The label arms are not a clean A/B — `mes_entropy` and `terminal_improvement` consume different RNG

EXPLORATORY, found while questioning h211's P-LABEL. Measurement on saved runs; no new runs.

## The defect

`simulate_mf_trajectory` line 1927:

```python
b_tau = _rollout_gumbel_b(current_ko) if rollout_reward == "mes_entropy" else None
```

`_rollout_gumbel_b` → `thompson_sample_y_star` → `posterior.rsample()`, which draws from the
**global torch RNG**. So `mes_entropy` consumes T+1 Thompson draws (K_rtg = 100 samples
each) *per rollout*; `terminal_improvement` consumes none. At 60–120 rollouts × 8 steps per
BO iteration, the two labels are running on completely different random streams from the
first iteration onward.

**Measured** — Hartmann, same seed, same initial design, first three REAL queries:

| seed | CTRL-K1 (`mes_entropy`) first y | K1-TI (`terminal_improvement`) first y | identical? |
|---|---|---|---|
| 42 | 1.992218 | 1.915485 | **No** |
| 43 | 1.191680 | 0.951391 | **No** |
| 44 | 1.116973 | 1.368100 | **No** |

The fantasy draws diverge before the first decision.

## What this invalidates

**h211's P-LABEL cannot attribute its +5.80 to the label.** K1-TI-H differs from CTRL-K1-H
in *three* ways at once: (a) the RTG label values, (b) the inference target schema (pinned
0.5 vs varying), (c) **the entire random stream, hence different rollouts and a different
training set**. The +5.80 on 0/5 is a real *configuration* effect — 0/5 one-sided is ~3%
under pure noise — but its cause is unattributed.

Same defect, same shape, as h198's config override: a difference credited to a label that
the code never isolated.

**Which existing comparisons are affected** (different labels on the two sides):

| comparison | clean? |
|---|---|
| h207 P-NIR (NIR `terminal_improvement` vs h206N `mes_entropy`) | **NO** |
| h210 CTRL-K1-H vs NIR-H | **NO** |
| h211 P-LABEL (K1-TI vs CTRL-K1) | **NO** |
| h209/h210 MIXR vs NIR (both `terminal_improvement`) | yes |
| h210 L1 arms vs NIR (both `terminal_improvement`) | yes |
| **h211 P-B / P-H (MIXR-K1 vs CTRL-K1, both `mes_entropy`)** | **yes** |

The headline result — MIXR-K1-B 7.17 vs CTRL-K1 11.59, 5/5 — is on the clean side. Adding
the random half does change RNG consumption (120 rollouts vs 60), but that is the
intervention itself, and RNG divergence between otherwise-identical procedures produces
noise with mean zero, not a 5/5 consistent direction.

## What h212 can and cannot settle

h212's TI-PCT arms use `terminal_improvement` on **both** sides of the schema comparison, so
**P-FIX (TI-PCT-H vs K1-TI-H) is a clean test of the target schema** — same label, same RNG
pattern. What h212 cannot do is tell us whether the *label* is harmful, because its
reference (K1-TI-H) is itself the unattributed arm.

## The missing control (not yet registered)

An **RNG-matched label A/B**: compute the per-step Gumbel draw under *every* label and
discard it when unused, so the two labels consume identical RNG. One flag, and then
`terminal_improvement` vs `mes_entropy` becomes a genuine one-factor comparison — which it
has never been in this project.

## Bearing on the user's objection

The user's point stands and is sharpened: RTG sensitivity measures 1.1–1.4×, so a pinned
RTG *target* is a weak candidate for a 5.8-point effect. Ranking the candidates by what the
evidence supports: (1) RNG divergence / different training data — untested, and now known
to be present; (2) the RTG label's value distribution entering a saturating
`Linear(1→H)+LayerNorm` (h177/h178) — untested; (3) the target pinning — h212 tests it, and
a null would be consistent with the objection rather than a surprise.
