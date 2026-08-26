# H71 — Is MF-DRO's pool load-bearing the way the baselines' is?

**CONFIRMATORY.** Predictions locked, with minimum effect sizes, before any h71
number exists.

## Why

h70 showed that on Borehole, raising a greedy single-fidelity baseline's
candidate pool from 200 to 1000 does not merely narrow the gap to MI-Greedy — it
**reproduces MI-Greedy exactly, seed for seed** (8.27% both, residual +0.00). The
GP construction contributed 0.00 there despite the builders producing materially
different models.

That exposed an uncontrolled factor across the whole h57 comparison:

| method | acquisition pool |
|---|---|
| **MF-DRO** | **200** (`n_roi_candidates`) |
| SF-MES / SF-EI | 200 |
| MF-MI-Greedy, MF-GP-UCB | **1000** |
| MF-MES (Takeno) | **2048 Sobol + top-K L-BFGS-B refinement** |

MF-DRO has been measured throughout against baselines given 5-10x more
acquisition-optimisation effort. The open question is whether MF-DRO's own pool is
load-bearing in the same way, or whether its deficit is insensitive to it.

Existing evidence is mixed and does not answer it: h61 found POOL600 moved
Borehole 23.7% -> 19.5%, but h66 found POOL600 on Hartmann at n=10 wins only
5/10 against MF-MES (the withdrawn claim).

## CORRECTION, made while h71 was running and BEFORE any result existed

The framing above is **wrong** and is corrected here rather than quietly edited.

`n_roi_candidates` is **not** an inference-time acquisition pool for MF-DRO.
`_propose_next_query` builds no candidate set at all ("No roi_candidates here
(Fix 1)"), and with the regression head — the default since h45 — the real query
is `x_t = action_head(h).clamp(0,1)`, a direct regression output. **MF-DRO does
zero inference-time acquisition search.** `n_roi_candidates` governs the
*rollout teacher's* pool, which shapes the demonstrations the DT is trained on.

So MF-DRO at "200" and MI-Greedy at "1000" are **not two sizes of one mechanism**;
they are different mechanisms. The comparison is a categorical asymmetry, not a
5-10x effort gap, and the h70 write-up overstated it. See the correction recorded
in findings.md.

**What h71 therefore actually tests:** whether a *better-optimised teacher*
(1000-candidate rollouts instead of 200) produces a better learned policy. That
remains a real and useful question — it is the training-signal-quality lever, and
h61/h64/h66 all probed it — but it is not a pool-matching exercise and no
"now they are matched" claim follows from it.

The locked predictions below are UNCHANGED. They were stated as regret
thresholds against h57 BASE and do not depend on the mistaken framing.

## Design

MF-DRO with `n_roi_candidates = 1000` (teacher rollout pool). One change from
h57's MF-DRO; every other config identical, including budget, initial design,
regret convention and the regression head.

Borehole 8D (primary — where h70 measured the pool at 4.72 points) and
Hartmann 6D (control), seeds 44/46/48. 6 jobs. h57 supplies BASE, no rerun.

## Locked predictions — each with a minimum effect size

The last two protocols (h68, h65) both passed a locked prediction while the
substantive claim failed, because the bars specified only a direction. Every bar
below names a magnitude.

1. **PRIMARY.** On Borehole, POOL1000 beats h57 BASE (23.7%) by **>= 2.0 points**
   of relative regret **and** in **>= 2/3 seeds**. Rationale is measured: h61's
   POOL600 already moved it 4.2 points, so a null here would contradict h61.
2. **SECONDARY.** POOL1000 does **not** reach MI-Greedy's 8.27% — specifically it
   stays **above 12%**. If MF-DRO's pool were the whole story as it was for
   SF-EI, it would land near 8.3%; anything above 12% means MF-DRO carries a
   deficit the pool does not explain.
3. **NULL.** Movement **< 2.0 points** on Borehole. Then MF-DRO's pool is *not*
   load-bearing even though the baselines' is, h61's POOL600 gain was noise, and
   the pool-size confound — while real for the baselines — does not explain
   MF-DRO's standing.
4. **CONTROL.** On Hartmann, POOL1000 must not be **worse than BASE by > 2.0
   points**. h66 already showed POOL600 is a coin flip there, so a large
   degradation would mean pool widening trades benchmarks rather than helping.

## AMENDMENT, recorded while h71 is INCOMPLETE (2/6, Borehole 1/3)

**The locked bars are unchanged.** This records context that did not exist when
h71 was written, before its verdict is read.

h71's PRIMARY compares POOL1000 against h57's BASE at **n=3 vs n=3** — which is
exactly what lesson 26 says does not estimate a direction. That weakness was
noted when h75 was launched, and h75 has since measured the same BASE cell at
n=10:

| | Borehole MF-DRO |
|---|---|
| h57 BASE, n=3 (the locked reference) | **23.71%** |
| h75 BASE, n=10 | **22.89%** (sd 2.94) |
| shift | −0.82 pts |
| three-seed range | [19.56, 25.45] |

**The locked reference turns out to be accurate to 0.82 points**, so h71's bar
("beats BASE by >= 2.0 points") is not materially distorted by having been set
against three seeds. The verdict will be evaluated against the locked n=3 value
as specified, with the n=10 value reported alongside.

What this does *not* fix: **POOL1000 itself will still be n=3.** Whatever h71
returns is a three-seed direction, and this project's record on those is three
failures out of four at n=10. The verdict must be read with that weight, not as
a settled result — and its NULL branch, which fires on movement < 2.0 points, is
the more likely outcome to be trustworthy since it asserts an absence.

**Operational note:** POOL1000's Borehole runs are taking ~529 min against h57
BASE's 82-114 min — a 5-6x slowdown, consistent with the 5x teacher pool.

## What this cannot settle

n=3. It equalises the pool against MI-Greedy and MF-GP-UCB but **not** against
MF-MES, which additionally runs top-K L-BFGS-B refinement on a 2048-point Sobol
pool — that arm remains unmatched and no claim about MF-DRO vs MF-MES follows
from this experiment. It also cannot resolve whether a *fully* effort-matched
comparison changes the standings; that needs the refinement equalised too.
