# h205 — fixing the positional/phase confound at the readout

**CONFIRMATORY.** Locked before any code is written and before any result exists.
**Human-proposed**, from reading `papers/DT.pdf` Algorithm 1 against our implementation.

> **REVISED before running (design review).** The first version of this protocol had two
> arms, "absolute" and "absolute + real prefix", which **confounded two independent
> interventions** -- it could not have told us whether absolute labelling contributes
> anything once the prefix is present. The review also found that the real prefix fixes
> the mismatch ON ITS OWN, more precisely than absolute labelling does, and needs no
> embedding-table change. The arms below separate them. My prior expectation is now that
> **A carries most of the effect and B contributes little** -- close to the opposite of
> how the first version was framed. Recorded here rather than quietly rewritten.

## The defect

DT's Algorithm 1 (p.5) uses a learned **episode** positional embedding indexed by the
ABSOLUTE step, and slides the window over it:

    pos_embedding = embed_t(t)                           # learned over the EPISODE
    s, a, t = s + [new_s], a + [action], t + [len(R)]    # t is ABSOLUTE
    R, s, a, t = R[-K:], ...                             # window slides; t keeps growing

Training samples random length-K subsequences from FULL episodes, so a given absolute
step appears at EVERY within-window slot across training samples. The index therefore
cannot encode "I am at the end of something".

Ours (`_train_dt`, mf_dro.py:3136):

    timesteps = torch.arange(T_max).unsqueeze(0).repeat(B, 1)   # [0..7], EVERY example, EVERY iteration

so of `nn.Embedding(max_seq_length=80, ...)` **only indices 0-7 ever receive gradient**,
and index 7 is ALWAYS the fragment's last step -- where the telescoped RTG has gone to
~0. **Index and within-fragment phase are perfectly confounded, in every batch, forever.**

The mismatch that matters is at the READOUT:

| | index 7 means | RTG at index 7 |
|---|---|---|
| training (current) | last simulated step (fragment end) | **~0** |
| inference (K=8) | the CURRENT real step | **large** (`rtg_tgt`) |

Two different semantic objects wearing the same index.

Measured context: real runs reach **110 (Borehole) to 159 (Hartmann)** real queries, so
the trained index range covers 7 of ~160 positions, and `max_seq_length=80` is smaller
than the real horizon regardless.

## The two independent fixes

**Real prefix (arm A).** Train on `[real n-7 … n-1] ++ [sim n … n+7]` = 15 positions,
ordinary `arange(15)` labelling, **loss on the simulated positions only**. The fragment's
end moves to index 14, so index 7 becomes the FIRST simulated step -- τ=0, RTG large, with
seven predecessors. That is exactly what inference presents at index 7.

The match is exact, not approximate: under the causal mask the state token at index 7
attends only to indices 0..7, so it sees **identical context** whether the sequence is 8
long (inference) or 15 long (training). Same index, same depth, same RTG magnitude, same
meaning. No embedding-table change needed.

**Absolute labelling (arm B).** `n_real_iter + arange(T)` in training, `[n-K+1 … n]` at
inference; `max_seq_length` 80 -> **256** (worst observed horizon 159 + rollout_length 8,
with headroom). This makes the readout index semantically honest -- position `n` is τ=0,
the start of the remaining horizon, which is what inference asks for. But in the CURRENT
batch it still trains position `n` with **zero predecessors** while inference gives it
seven; position `n` was trained at depth 7 at iteration `n-7`, seven batches ago, and
there is **no replay buffer** (`_train_dt` only ever sees the current batch). So B
converts the mismatch from STRUCTURAL to RECENCY-BASED; it does not remove it.

## Arms

| arm | labelling | training sequence | fixes |
|---|---|---|---|
| **A** | `arange` (unchanged) | **real prefix + fantasy** | readout semantics + context depth |
| **B** | **absolute** | fantasy only (as now) | readout semantics only |
| **C** | absolute | real prefix + fantasy | both |
| CTRL-K1 | `arange` | fantasy only | — **already in hand, 11.59** |

All arms at **K=8** (the window is what exposes the defect; at K=1 the readout is index 0
and there is no confound to fix). Borehole, seeds 42-46, 15 workers.

**Everything is gated behind `absolute_timesteps` / `real_prefix_training`, both default
False.** The identity gate (122.29066752728207) MUST still pass with both off.

## SCs, registered before running

1. **Not a silent no-op.** For B/C, record which `position_embedding` indices receive
   gradient -- currently {0..7}; under absolute it must span the run's real range. For
   A/C, assert the prefix actually carries K-1 entries once available. **Either failing
   reads as P2**, and both are checked FIRST: a silent no-op is indistinguishable from
   "the fix is immaterial", which is exactly the failure h196's SC caught.
2. **Identity gate PASSES with both flags off**, exactly at 122.29066752728207.
3. **No index overflow** (B/C): max absolute index used < `max_seq_length`, asserted
   in-run rather than discovered as an index error mid-arm.
4. **Prefix RTG/BTG consistency (A/C).** Prefix positions need RTG/BTG as INPUTS even
   though their loss is masked. They must be labelled by h197's inference rule
   (`rtg_tgt + log b_tau - log b_now`), or we introduce a NEW train/inference mismatch
   while fixing this one. **This is the most likely place for a silent bug** and is
   checked explicitly, not assumed.
5. **Cold start (A/C).** For `n < K-1` the prefix is short; verify the loss mask is
   correct there rather than trusting it.

Note on prefix actions: they are the DT's own past choices, not teacher actions. Masking
the loss handles the self-imitation concern; they remain INPUTS, which is correct, since
inference feeds them too (the h196 fix).

## Prediction, committed now

- **P1** -- A beats CTRL-K1 (< -1.26). The positional/phase confound was load-bearing and
  the prefix fix removes it.
- **P2** -- no arm moves (|diff| <= 1.26). The confound is real but is not what limits
  performance; the constant-predictor behaviour survives an honest positional signal.
- **P3** -- worse. Registered but now considered UNLIKELY: the dilution worry that
  motivated it is quantified below and is not severe.

**Ordering prediction, stated so it can be wrong:** A >= C > B. A and C should be close
(both fix context depth); B should contribute little on its own.

**Gradient dilution, quantified** (this replaces the hand-waved version in v1):

| labelling | updates per index |
|---|---|
| `arange` | 60 traj x 100 epochs x ~110 iters = **660,000** (index 7) |
| absolute | 60 x 100 x 8 iters = **48,000**, then never again |

14x less, but 48k updates on a single 128-dim vector is ample. Useful corollary: position
`p` is covered by fragments from iterations `p-7…p`, so **the position being queried is
always among the 8 most recently trained** -- staleness is bounded.

## What each outcome RETRACTS

- **P1 on A** retracts nothing previously claimed, but demotes the reading -- implicit
  across this session -- that the window's problems are mainly fidelity saturation (h202)
  and late-step teacher content (h201). It adds a third, more basic cause: the readout
  index never meant what inference needed it to mean.
- **B >> A** would retract this protocol's central argument (that context depth is the
  binding constraint) and restore absolute labelling as the primary fix.
- **P3** would retract the claim that `arange` labelling is a defect at all, reframing it
  as accidental regularisation.
- None of these bear on h201's ablation, which held labelling fixed across its arms.

## Relationship to the open RTG question

h198 established (bit-identical runs, md5-matched traces, 5/5) that the RTG channel is
causally disconnected end to end. h205 does **not** claim to fix that. It removes a
positional confound that plausibly lets the model reach low training loss without ever
consulting RTG. Whether RTG starts mattering is a SEPARATE measurement and is explicitly
NOT what P1/P2/P3 are scored on.

## Cost

Ordinary MES teacher (no lookahead); h197 measured ~80 min/seed for a K=8 run. Arms A/C
carry ~15-position sequences instead of 8, so expect a modest training slowdown, not a
different order of magnitude. 3 arms x 5 seeds = 15 workers, exactly at the cap. Compute
is currently free (0/15).
