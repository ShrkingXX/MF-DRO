# H37 — cost-weighted regret: MF-DRO's advantage is ANYTIME, not terminal

**Status: EXPLORATORY** (user-requested analysis, no pre-registered protocol).
**Supplementary metric.** `PROTOCOL.md`'s success test is on *final* simple
regret and is unaffected by anything here. Nothing below may be used to claim
the frozen test passed — it did not.

## Cost-weighted regret (area under the regret-vs-cost curve / budget)

| method | AUC/budget | r@25 | r@50 | r@100 | r@150 | r@200 |
|---|---|---|---|---|---|---|
| **MF-DRO / joint MES** | **0.6805 ± 0.070** | 1.227 | 0.912 | **0.490** | 0.416 | 0.401 |
| MF-DRO / improvement | 0.8716 ± 0.057 | 1.394 | 1.004 | 0.815 | 0.638 | 0.505 |
| MF-MI-Greedy | 1.0639 ± 0.127 | 1.592 | 1.176 | **1.042** | 0.862 | 0.593 |
| MF-GP-UCB | 1.7934 ± 0.122 | 1.793 | 1.793 | 1.793 | 1.793 | 1.793 |

Paired against MF-DRO/joint-MES (negative = joint MES better):

| comparison | diff | seeds won | Wilcoxon |
|---|---|---|---|
| vs `improvement` | **−0.1911** | **9/10** | **p = 0.0371** |
| vs MF-MI-Greedy | −0.3834 | 7/10 | p = 0.0645 |
| vs MF-GP-UCB | −1.1129 | 10/10 | p = 0.0020 |

## Two findings the endpoint metric hid

**1. The reward change looks better on anytime performance — but the p-value is
NOT usable.** Final regret gave `improvement → joint MES` at p = 0.375 (6/10
seeds). Cost-weighted regret gives p = 0.0371 on 9/10 seeds.

**This is metric selection and I originally reported it as "significant", which
was wrong.** I computed the AUC comparison *only after* the pre-registered
metric failed. Correcting for the two metrics tested gives a threshold of
0.025 — and **p = 0.0371 does not survive it**. Counting the whole family I
actually reported in this table (final + AUC + five checkpoints = 7 tests), the
Bonferroni threshold is 0.00714 and **0 of 7 survive**.

Cost-weighted regret is **not in `PROTOCOL.md`**. It is post-hoc, full stop.

**What survives without a p-value** is the descriptive fact, which is worth
stating on its own terms: joint MES is lower on **9/10 seeds** and lower at
**every checkpoint** (r@25, 50, 100, 150, 200). Consistent direction across
seeds and across the whole trajectory is real evidence about behaviour; it is
just not a significance claim, and must not be dressed as one.

**2. MF-DRO's advantage over MI-Greedy is early, and MI-Greedy closes late.** At
cost 100 the gap is **0.490 vs 1.042 — a factor of 2.1**. By cost 200 it has
narrowed to 0.401 vs 0.593. MF-DRO reaches good solutions much faster; MI-Greedy
catches up substantially by the end of the budget.

This is a genuine characterisation difference and it is invisible in the frozen
metric, which samples only the final point. It also fits the fidelity finding:
MF-DRO queries HF far more often than the teacher does (~26–31% vs ~12%), which
should buy early incumbent progress at the cost of late-budget efficiency.

## What this does NOT change

The paired comparison against MI-Greedy is **p = 0.0645** — still not
significant at n = 10, the same underpowering that affects the final-regret
comparison. And the frozen success test remains **FAIL**. The honest headline is
unchanged: *the gap narrowed, it did not close.* What this adds is that the
narrowing is much larger mid-run than at the endpoint.

## A bug worth recording

My first attempt subtracted the initial-design cost (348) from `cost_curve`, but
that field is **already** post-init cumulative — `cumulative_cost_curve` is the
one including initialisation. Every lookup fell off the end of the array and
returned the final regret, producing a table where r@25 = r@50 = r@200 for every
arm. The tell was that identical-values pattern; caught before reporting.
