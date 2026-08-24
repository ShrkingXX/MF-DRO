# H13 — remove the two clips that starved H12's LF credit

## What H12's gate established

`kg_incumbent` fixed the "LF earns exactly 0.0" defect (dead-signal 63.0% ->
4.5%, G1 PASS) but LF steps earned nonzero reward only 27.2% of the time
(G2 FAIL, needed >50%). Diagnosed cause: two clips in my estimator of `V`.

- **(i)** `V = MAX over the reference set` only moves when an observation lands
  near the current argmax.
- **(ii)** `r = max(0, dV)` discards downward revisions, which are real progress.

## Intervention — two independent flags, tested as a 2x2

- `kg_signed` (bool): keep the sign of `dV` instead of clipping at 0.
- `kg_topk` (int): `V` = mean of the top-k HF posterior means instead of the
  hard max. `k=1` reproduces H12 exactly.

Four arms on an identical 200-trajectory batch, same seed:

| arm | kg_topk | kg_signed |
|---|---|---|
| baseline `improvement` | — | — |
| A (= H12) | 1 | False |
| B (clip removed only) | 1 | True |
| C (both) | 5 | True |

Running B separately is the point: it says whether the clip or the hard max is
the dominant cause, rather than changing two things and learning neither.

## GATE (same standard as H12; arm C is the candidate)

- **G1** dead-signal fraction < 20%
- **G2** LF steps with nonzero reward > 50%

If arm C fails either, H13 reports the gate failure and stops. No comparison,
no regret claim.

## Locked predictions

1. **PRIMARY**: arm C passes G2 (>50%).
2. **DECOMPOSITION**: arm B's LF-nonzero fraction lands strictly between A
   (27.2%) and C. If B ~= C the clip was the whole story and `kg_topk` is
   unnecessary complexity that should be dropped; if B ~= A the hard max was,
   and `kg_signed` should be dropped. Whichever is inert gets removed before
   anything further is built on this reward.
3. **SIDE-EFFECT CHECK**: signed rewards make RTG able to go negative. Report
   `neg_rtg_frac`. If >50% of trajectories carry a negative `rtg[0]`, the
   floored-dynamic target `max(batch_max, alpha*running_max)` is being fed a
   mostly-negative signal and the schema interaction must be re-examined before
   the reward is used in any run.

## Out of scope

No regret comparison, no change to `PROTOCOL.md`, no cost normalisation.
