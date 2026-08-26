# H81 — Can MF-DRO's ensemble size M shrink from 10 to 3?

**CONFIRMATORY.** Locked before any h81 number exists. User-directed.

## Why

The DRO paper's SF-DRO ablation reports M=3, M=5 and M=10 performing very
similarly. If that carries to MF-DRO, M can shrink and the freed compute can be
spent on the teacher's acquisition optimisation, which h61/h71 showed is the one
lever that improves MF-DRO (Borehole 23.71% -> 17.66% at pool 1000, 3/3).

## Design

MF-DRO on **Hartmann 6D**, M in **{3, 5, 10}**, seeds **42-46** (n=5), cost
budget **300**. 15 jobs.

**Run at the TARGET shipping configuration**, not the current default:
`n_roi_candidates=1000` + refinement (`teacher_refine_samples=100`,
`teacher_refine_noise=0.05`). Rationale: an M decision validated at the old
200-point pool need not hold at 1000, and a wrong M would propagate into the
~100-run pitch-talk comparison downstream. Testing M where it will actually be
used is the point.

**Note this combination is itself untested.** h71 tested pool=1000 alone; h61
tested refinement alone at pool=200. Pool-1000-plus-refinement has never been run.
h81 is therefore also the first measurement of the shipping config.

## Locked predictions

1. **PRIMARY.** M=3 is within **2.0 points** of relative regret of M=10 on the
   mean, and wins or ties **>= 2 of 5** seeds. Both required (lesson 27: a
   magnitude bar and a win count catch different failures).
2. **SECONDARY.** The three arms' means span **< 3.0 points** — "very similar",
   as the paper reports for SF-DRO.
3. **NULL / SHRINK REFUSED.** M=3 is worse than M=10 by **>= 2.0 points** or wins
   <= 1 of 5. Then M does not shrink safely for MF-DRO despite doing so for
   SF-DRO, and the default stays at 10.

## Decision rule, fixed in advance

If PRIMARY and SECONDARY both hold, the default becomes the **smallest M that is
within 2.0 points of M=10**, with `n_roi_candidates=1000` and refinement enabled.
If NULL fires, M stays 10 and the pool/refinement change is applied on its own.

## What this cannot settle

Hartmann only, n=5, one budget. It does not establish that pool-1000-plus-
refinement beats the current default — that comparison is not run here, and h78
(the n=10 replication of pool-1000 alone) was halted before completing.
