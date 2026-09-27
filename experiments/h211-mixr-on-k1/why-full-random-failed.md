# Why the all-random training schema collapsed — answered, and it was dropped for the wrong reason

EXPLORATORY. Re-analysis of saved traces (h149, h84, h211, h83). No new runs.

## The measurement

Real post-init queries, unit coordinates. `|centre−0.5|` = distance of a run's query-cloud
centre from the box centre; `dispersion` = ‖per-dim std‖.

| arm | bench | dispersion | \|centre−0.5\| | regret |
|---|---|---|---|---|
| **h149 RANDOM-POOL** (all-random teacher) | Borehole | **0.090** | **0.024** | 43.94 (floor) |
| **h149 RANDOM-POOL** | Hartmann | **0.091** | **0.024** | — |
| CTRL-K1 (all MES) | Borehole | 0.237 | 0.875 | 11.59 |
| MIXR-K1 (half random) | Borehole | 0.276 | 0.834 | 7.17 |
| MF-MES (no DT) | Borehole | 0.259 | 0.969 | 6.40 |
| CTRL-K1 | Hartmann | 0.158 | 0.660 | 5.93 |
| MIXR-K1 | Hartmann | 0.160 | 0.548 | 11.19 |

## The mechanism

h149 set `rollout_policy="random"` and nothing else. `use_roi` defaults to **False**
(mf_dro.py:1058), so `roi_candidates` is a plain uniform draw over the whole box — the
random teacher sampled **uniformly over the domain**. Under THE_ANSWER (the DT's MSE
location head emits the conditional *mean* of its training actions at the read position),
the mean of uniform-over-the-box is **the box centre**.

That is exactly what the traces show: the query cloud sits **0.024** from the box centre
with dispersion **0.090**, and — the tell — the numbers are **identical on two different
benchmarks** (0.024/0.024, 0.090/0.091). A state-dependent policy cannot produce that.
It is a state-independent constant predictor, which is why it never improves on its
initial design (h149: 0/3 seeds, regret 43.94 = the saturation floor). It also reproduces
h167's earlier observation that failing arms "emit queries within 0.04 of the box centre".

**So the all-random schema did not fail because random rollouts are low quality. It failed
because the mean of uniform actions is a single, useless, state-independent point.**

## What this retracts about the dropped arms

h207 dropped arm **R** (all-random data, RTG selects) and arm **W** (argmin winners) on
Stage 0b's finding that the best of 100 random 8-step rollouts lands 27–43 units *below*
the real incumbent. **That reasoning is wrong for R.** Under THE_ANSWER the rollouts'
*outcomes* are not what reaches the real query — the *action mean* is. Outcome quality is
the right objection to W (whose winners are selected by outcome) but not to R.

The right objection to R is the one measured here, and it is conditional on the pool:
uniform-over-box ⇒ box centre ⇒ collapse. **With `use_roi=True` the random teacher draws
from the ROI-filtered pool, so its action mean is the ROI centroid — a point that tracks
the model's own belief about where the optimum is, and moves as the ROI tightens.**
All-random-*with-ROI* has never been run. Its prediction is sharp: it should emit
approximately the ROI centroid each iteration.

## What it refutes in my own MIXR account

I had proposed that the random half **shrinks** the DT's query toward a centroid. The same
table refutes it: MIXR-K1 is **more** dispersed than CTRL-K1 (0.276 vs 0.237), and more
dispersed than MF-MES (0.259). The half-random arm spreads queries out, it does not
contract them.

One correlation worth recording, EXPLORATORY and untested: the dispersion increase is
benchmark-specific — Borehole **+16%** (0.237 → 0.276), Hartmann **+1%** (0.158 → 0.160) —
and Borehole is the benchmark where MIXR helps (−4.42) while Hartmann is where it hurts
(+5.26). Whether added dispersion is the mechanism or a by-product is not established by
this analysis.

## Candidate next arm (not registered)

**R-ROI**: all-random teacher with `use_roi=True`, K=1, Borehole 42–46, 5 runs. If it
lands near CTRL-K1 rather than at the 43.94 floor, the random half's whole contribution is
the ROI centroid and MIXR reduces to a blend of two known points. If it collapses again,
the ROI pool is not what rescues it and the mixture itself matters.
