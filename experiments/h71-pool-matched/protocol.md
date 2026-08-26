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

## Design

MF-DRO with `n_roi_candidates = 1000`, matching MI-Greedy and MF-GP-UCB exactly.
One change from h57's MF-DRO; every other config identical, including budget,
initial design, regret convention and the regression head.

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

## What this cannot settle

n=3. It equalises the pool against MI-Greedy and MF-GP-UCB but **not** against
MF-MES, which additionally runs top-K L-BFGS-B refinement on a 2048-point Sobol
pool — that arm remains unmatched and no claim about MF-DRO vs MF-MES follows
from this experiment. It also cannot resolve whether a *fully* effort-matched
comparison changes the standings; that needs the refinement equalised too.
