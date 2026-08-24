# H11 analysis — arms A/B interpretable (null), arm C VOID; and the void has a cause

## Result

| arm | context | RTG channel | argmax moved |
|---|---|---|---|
| A | T=1, timestep 0 | 1 token | 0/12 |
| B | T=8 real history | 8 tokens, constant | 0/12 |
| C | T=8 real history | 8 tokens, DT-style decrement | 0/12 — **VOID** |

Arm C's realised within-pass RTG sequence was **constant**:
`[0.5672] * 8`, span 1.00x against a pre-registered >= 3x requirement.

## Arms A and B ARE interpretable

Their manipulation is *structural* — the readout token attends over 1 vs 8 RTG
tokens and over 4 vs 32 tokens total — and that manipulation demonstrably
happened. Both sit at 0/12.

**Conclusion (valid):** feeding the real queried history at inference, on its
own, does not move the decision. Context length is not the lever.

## Arm C is void, and WHY it is void is the finding

The DT-style decrement `R_{t+1} = R_t - r_t` produced a constant sequence
because **every realised reward in the window was zero**. That is not an
accident of this seed. Under `rollout_reward="improvement"`
(`mf_dro.py:1267-1274`):

    if ell_tau == 1:  r_tau = max(0, y_tau - best_sim_hf)
    else:             r_tau = 0.0          # <-- LF earns EXACTLY zero

Measured on a real 200-trajectory batch (`_diag_lf_reward_credit.py`):

- **63.0%** of trajectories have `rtg[0] == 0` — dead conditioning signal
- only **23.7%** of steps carry a nonzero reward
- `Spearman(n_HF, rtg[0]) = +0.355` (p < 1e-6): the return partly just counts
  how many HF queries the rollout made

So there is nothing for the decrement to decrement by.

## This unifies THREE failed experiments

H9, H10 and H11-arm-C all tried to create RTG variation and all failed their own
manipulation checks. The common cause is now visible: **the reward is zero
almost everywhere, so no schema change can manufacture variation in a signal
that is identically zero.** H9 and H10 attacked the *normalisation* and the
*floor*; the actual constraint is upstream of both, in the reward definition.

The open question was "is RTG inert or starved?". The answer is **starved**, and
the starvation has a specific, fixable cause.

## A flaw in my own pre-registered logic

The protocol's NULL-GUARD fires on `max(arms) < 0.05` — and the script printed
its "closes the confound in the negative direction" verdict even though arm C
voided. **That verdict is not valid** and is retracted here. The guard should
have been conditioned on the manipulation check passing; a null across arms
means nothing for an arm whose manipulation never occurred. Only the A/B null
stands.

## A prediction I got wrong, in sign

I predicted the improvement reward would *pay a premium for uncertainty*, since
`E[max(0, y-best)]` for a fantasy draw is Expected Improvement, which is monotone
increasing in sigma (verified numerically: E[r] = 0.040 at sigma=0.1 rising to
1.596 at sigma=4.0). The analytic fact holds, but it does **not** dominate:
measured `Spearman(LF fraction, rtg[0]) = -0.355`, i.e. LF-heavy rollouts score
*lower*, because "more HF steps = more chances at a nonzero term" outweighs the
variance premium. Reported as measured, against my prediction.

## Next

The LF reward definition is now the blocking issue, not the network and not the
inference context. See H12.
