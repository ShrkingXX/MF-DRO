# h205 — absolute timestep embeddings, faithful to DT

**CONFIRMATORY.** Locked before any code is written and before any result exists.
**Human-proposed**, from reading `papers/DT.pdf` Algorithm 1 against our implementation.

## The defect

DT's Algorithm 1 (p.5) uses a learned **episode** positional embedding indexed by the
ABSOLUTE step, and slides the window over it:

    pos_embedding = embed_t(t)                  # per-timestep, learned over the EPISODE
    s, a, t = s + [new_s], a + [action], t + [len(R)]   # t is ABSOLUTE
    R, s, a, t = R[-K:], ...                    # window slides; t keeps growing

Training samples random length-K subsequences from FULL episodes, so a given absolute
step appears at EVERY within-window slot across training samples. The index therefore
cannot encode "I am at the end of something".

Ours (`_train_dt`, mf_dro.py:3136) does:

    timesteps = torch.arange(T_max).unsqueeze(0).repeat(B, 1)   # [0..7], EVERY example, EVERY iteration

so of `nn.Embedding(max_seq_length=80, ...)` **only indices 0-7 ever receive gradient**,
and index 7 is ALWAYS the fragment's last step -- where the telescoped RTG has gone to
~0. Index and within-fragment phase are **perfectly confounded, in every batch, forever**.

Measured context: real runs reach **110 (Borehole) to 159 (Hartmann)** real queries, so
the trained index range covers 7 of ~160 positions, and `max_seq_length=80` is smaller
than the real horizon regardless.

## Design

**Gated behind `absolute_timesteps` (default False = bit-identical).** The identity gate
(122.29066752728207) MUST still pass with the flag off; this changes the default training
path otherwise, so it cannot be an unconditional edit.

| | training timesteps | inference timesteps |
|---|---|---|
| current | `arange(T)` = [0..7] | `arange(T)` = [0..7] (K>1), `0` (K=1) |
| **h205** | `n_real_iter + arange(T)` | `[n-K+1 … n]` (K>1), `n` (K=1) |

`max_seq_length` 80 -> **256** (covers the worst observed horizon 159 + rollout_length 8,
with headroom; the table is `nn.Embedding`, so unused rows cost only memory).

### Arms

| arm | timesteps | training sequence |
|---|---|---|
| **A** | absolute | pure fantasy fragment `[n … n+7]` (as now) |
| **B** | absolute | **real prefix + fantasy**: `[real n-7 … n-1] ++ [sim n … n+7]`, loss on the SIMULATED positions only |
| CTRL-K1 | current `arange` | — **already in hand, 11.59** |

**Why B exists.** Under A, at iteration `n` the current batch trains position `n` as
tau=0 with ZERO predecessors, but inference queries position `n` with K-1 predecessors.
Position `n` WAS trained at depth 7 -- at iteration `n-7`, seven batches ago, and there
is NO replay buffer (`_train_dt` only ever sees the current batch), so those weights are
stale. A converts the mismatch from STRUCTURAL to RECENCY-BASED; B removes it, by making
the training sequence structurally identical to the inference sequence. The real prefix
already exists in `self._real_hist`, so B costs nothing to generate.

Loss is computed on simulated positions only because the real prefix's actions are the
DT's OWN past choices, not teacher actions -- training on them would be self-imitation.

## SCs, registered before running

1. **The embedding table is actually being exercised.** Record which `position_embedding`
   indices receive gradient. Under the current code this is {0..7}; under h205 it must
   span the run's real range. If it does not, the flag is a silent no-op -> **P2**, and
   this is checked FIRST because a no-op is indistinguishable from "the fix is immaterial"
   (the exact failure mode h196's SC caught).
2. **Identity gate PASSES with the flag off**, exactly at 122.29066752728207.
3. **No index overflow**: max absolute index used < `max_seq_length`, asserted in-run
   rather than discovered as a CUDA/index error mid-arm.
4. **Arm B prefix check**: the real prefix actually carries K-1 entries once available,
   and loss masks exclude those positions. (Same class of check as h196's `ax` slots,
   which were silently zero.)

## Prediction, committed now

The mechanism says the DT emits its teacher's action mean at the read position. Under
`arange` labelling the read position is index 7, permanently fused to "fragment end,
RTG~0". Absolute labelling breaks that fusion.

- **P1** -- A and/or B beat CTRL-K1 (< -1.26). The positional confound was load-bearing,
  and removing it is the first structural fix on this front to improve the endpoint.
- **P2** -- no change (|diff| <= 1.26). The confound is real but not what limits
  performance; the constant-predictor behaviour survives an honest positional signal.
- **P3** -- worse. Plausible and worth naming: `arange` may be doing useful work by
  keeping every training example in a narrow, well-learned index range, and spreading
  gradient over ~160 indices could simply dilute it (each index now sees ~1/8 as many
  updates per position).

**B vs A specifically** isolates the context-depth question, which is otherwise assumed
rather than tested.

## What each outcome RETRACTS

- **P1** retracts nothing previously claimed, but would demote the reading -- implicit
  across this session -- that the window's problems are mainly about fidelity saturation
  (h202) and the teacher's late-step content (h201). It would add a third, more basic
  cause: the position index never meant what inference needed it to mean.
- **P3** would retract the argument in THIS protocol (that `arange` labelling is a defect
  at all) and reframe it as a deliberate, if accidental, regularisation.
- Either way this does NOT bear on h201's ablation, which held labelling fixed across
  its arms.

## Relationship to the open RTG question

h198 just established (bit-identical runs, md5-matched traces, 5/5) that the RTG channel
is causally disconnected end to end. h205 does NOT claim to fix that -- it removes a
positional confound that plausibly lets the model succeed at training loss without ever
consulting RTG. Whether RTG starts mattering is a SEPARATE measurement, and it is
explicitly NOT what P1/P2/P3 above are scored on.

## Cost

Ordinary teacher (no lookahead). h197 measured ~80 min/seed for a K=8 MES-teacher run.
Borehole seeds 42-46, both arms = 10 workers. Compute is free (0/15 in use).
