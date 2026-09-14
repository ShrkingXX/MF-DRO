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

---

## CORRECTION 2026-09-14 — the label fork was a SILENT NO-OP. "RTG causally disconnected" is RETRACTED.

Found while designing Q3, which rested on this result. Three independent checks:

1. **The per-iteration `rtg_target` recorded in both arms' result JSONs is identical to 4
   decimals over every iteration** (seeds 42/43/46, 109/103/111 iterations). Two different
   reward definitions (log-entropy ratio vs cumulative fantasy improvement) cannot produce
   the same target every iteration. The arms fed the DT the SAME labels.
2. **Cause located:** `experiments/h83-main-comparison/code/worker.py` `run()` executes
   `cfg.rollout_reward="mes_entropy"` AFTER `_build_mf_dro_config` returns. h198's
   `worker_b.py` set `c.rollout_reward='improvement'` INSIDE its `_build` patch, which
   `run()` then overwrote. Both arms ran `mes_entropy`. The `_h198` metadata dict recorded
   the intended value, not the effective one.
3. **A direct forward test on the model** (random init, float32): scaling RTG by 3x+1
   moves `forward_mf`'s location output by 6.5e-02 and `propose_mf`'s query by 6.1e-02.
   The channel is numerically wired. h60's REWARD arm -- a genuine fork, set AFTER the
   override -- also moved the outcome (0/3, distinct numbers on every seed), consistent
   with h177/h178's measured RTG embedding response of 0.52 over its operating range.

**Retracted:** "the RTG channel is causally disconnected end to end" and everything that
cited it as evidence -- research-state's h198 entry, the Layer-3 framing that RTG
inertness was *established*, and h205's protocol sentence quoting it. h198 tested nothing
about RTG. Its P3 quality result (both arms 13.46, +1.87 vs CTRL-K1) stands unchanged,
since both arms were the same run.

**What is still true about RTG:** the single-teacher hypothesis -- RTG varies only with
the world, which is already in the state, so a trained model has no reason to consult it
-- is a live explanation with NO direct evidence either way. It is now the thing Q3 tests
rather than a premise Q3 assumes.

**Fix committed:** h83's worker now reads `cfg.rollout_reward=ROLLOUT_REWARD`, a
module-level knob defaulting to `"mes_entropy"` (identity gate PASS, bit-identical).
Workers fork the reward via `h83.ROLLOUT_REWARD = ...`, never inside `_build`. The
override affected only h198b among all workers (h60 set its fork after the line; h199
set mes_entropy, matching the override).
