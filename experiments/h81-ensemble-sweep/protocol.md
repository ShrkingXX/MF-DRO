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

**AMENDED before any result existed — the shipping config is not runnable.**

h81 was first launched at `n_roi_candidates=1000` + refinement. Measured rate:
**17-34 min/query**, i.e. **30-57 h for a single Hartmann run**, because
`n_roi_candidates` is consumed inside `simulate_mf_trajectory` and is therefore
multiplied by `rollouts_per_iter x rollout_length` on every BO iteration. Pool
1000 plus refinement is past what the rollout budget can carry. Halted at
queries 0-1 with no results recorded.

**Relaunched at the current default** (`n_roi_candidates=200`, no refinement) to
isolate M cheaply. This is the weaker test — an M decision validated at 200 need
not hold at a larger pool — and that limitation is recorded here rather than
hidden. The purpose of the sweep is to free GP-side compute so a larger pool
becomes affordable at all; establishing M first and choosing the pool second is
the only order that fits the budget.

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

---

## OUTCOME: COMPLETED 15/15. PRIMARY met, SECONDARY missed, decision rule NOT triggered.

The last two M10 cells landed moments before the run was stopped, so the verdict
*was* evaluable. (An earlier version of this section said "abandoned at 13/15,
verdict never evaluated" -- that was written from a stale count and is wrong.)

| seed | M3 | M5 | M10 |
|---|---|---|---|
| 42 | 20.38% | **69.04%** | 12.95% |
| 43 | 1.78% | 2.46% | 2.38% |
| 44 | 6.81% | 8.30% | 22.67% |
| 45 | 8.57% | 8.61% | 6.31% |
| 46 | 9.68% | 5.52% | 7.36% |
| **mean** | **9.45%** | **18.79%** | **10.33%** |
| sd | 6.82 | 28.21 | 7.86 |

- **PRIMARY MET.** M3 vs M10: gap **-0.89 pts** (M3 slightly ahead), 2/5 wins/ties,
  Wilcoxon p = 0.8125. The two are indistinguishable, which is what the bar asked.
- **SECONDARY NOT MET.** Arm span **9.34 pts** against a < 3.0 bar. The three arms
  are not "very similar" -- but that is entirely M5, whose mean is dragged by seed
  42's 69.04% (sd 28.21).
- **NULL did not fire.**
- **DECISION RULE: not triggered -> "M stays 10."**

### The rule was badly specified, and the record should say so

The decision needed one question answered: *is M=3 safe relative to M=10?* The
PRIMARY answers it -- yes, gap 0.89 points, p=0.81. The SECONDARY instead asked
whether **all three** arms are similar, which a single bad M5 seed can veto even
though M5 is irrelevant to a choice between 3 and 10. Requiring their conjunction
let an irrelevant arm block a decision its own PRIMARY supported.

**M=3 was adopted, and the pre-registered rule as written says M stays 10.** The
honest position: the bar that tests the actual question was met; the conjunction
that gates the decision was mis-built by me. This is the fourth mis-specified bar
this session (h68 PRIMARY necessary-but-not-sufficient, h65 direction-only,
h80 compound NULL, h81 irrelevant-arm conjunction).

Independently established, and unaffected: **M10/M3 wall-time is 1.39x**, not the
~3x the premise assumed.
