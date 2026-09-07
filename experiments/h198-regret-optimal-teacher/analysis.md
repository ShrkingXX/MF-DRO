# h198 — **P3 on quality, and a BIT-IDENTICAL label fork.**

**CONFIRMATORY**, 5/5 finals both arms. Quality by final simple regret only.

## Result

| arm | final regret |
|---|---|
| h198a regret-lookahead / `mes_entropy` label | **13.46** |
| h198b regret-lookahead / `improvement` label | **13.46** |
| h194 CTRL-K1 (MES teacher, K=1) | 11.59 |

Both arms: paired **+1.87** vs CTRL-K1 (se 0.29), better on **0/5** -> **P3, the
regret-lookahead teacher HURTS** at K=1. Consistent with h201B: a teacher change alone,
without a window exposing a read position where that teacher's action is worth reading,
does not help -- and here it actively costs 1.87.

## The label fork produced BIT-IDENTICAL runs

| seed | arm a | arm b | |
|---|---|---|---|
| 42 | n_q=139, md5 aeb2f495f58e | n_q=139, md5 aeb2f495f58e | **identical** |
| 43 | n_q=133, md5 0e4e53c1f216 | n_q=133, md5 0e4e53c1f216 | **identical** |
| 46 | n_q=141, md5 5c098b8e08a2 | n_q=141, md5 5c098b8e08a2 | **identical** |

Paired a-b = **+0.00, se 0.00**, per-seed identical to 2 decimals. md5 over the full
(cost, fidelity, y) trace matches exactly.

`rollout_reward` changes BOTH the RTG label on every training trajectory AND the
`rtg_target` computed and fed at inference. Changing it produced **zero** difference in
behaviour. This is not "RTG is weakly used" -- the RTG channel is **causally
disconnected** from the emitted action, end to end.

**Corrects an earlier in-flight observation.** Mid-run the two arms' CHECKPOINTS differed
(n_q 79 vs 73, different last_y, different md5) and that was read as "they diverge later".
Wrong: those were snapshots taken at different points of progress, not behavioural
divergence. The completed runs are identical.

## What this establishes

Strongest confirmation yet of h180's architectural-inertness finding, now at the level of
the ENTIRE label pipeline rather than just the inference target. It also means the h198
label factorial -- registered specifically because the regret-lookahead teacher optimises
the labelled quantity by construction -- could not have resolved anything: the channel it
forks on does not reach the action.

Directly motivates the RTG question (why design rewards at all if the DT cannot tell good
from bad?) and is evidence that the cause is a DATA property: every rollout in a batch
comes from ONE teacher, so RTG varies only by GP/fantasy noise, never by behaviour, and a
per-timestep constant predictor attains near-optimal training loss while ignoring it.
