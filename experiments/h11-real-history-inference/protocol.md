# H11 — is the RTG channel *absent* at inference rather than inert?

## Motivation (user-raised, premise verified)

Real inference builds a **T=1** sequence at `timestep=0`
(`mf_dro.py:2597`, `decisionTransformer.py:608`), while training uses T=8 at
positions 0..7. Chen et al. 2021 (Decision Transformer), Algorithm 1, instead
feeds the last K real steps AND decrements RTG by the realized reward
(`R_{t+1} = R_t - r_t`), so the model sees a *declining RTG sequence* it can
attend across.

At T=1 there is exactly ONE RTG token and no decrement. H9 and H10 both tried to
resolve "RTG inert vs starved" by widening the band of that single scalar, and
both voided against the provable `1/alpha_rtg` cap. **This experiment changes the
channel's structure instead of its magnitude** — the manipulation is therefore
guaranteed large, which is precisely what H9/H10 lacked.

## Design (one variable: inference context only)

Train ONE model exactly as now (`rollout_reward="improvement"` unchanged, no
schema change, no retrain difference between arms). Then evaluate the SAME
trained weights under three inference contexts on identical candidate pools:

- **A (current)**: T=1, `timestep=0`, single RTG target.
- **B (history, flat RTG)**: T=k real steps, positions 0..k-1, same RTG target
  repeated at every step. Isolates *context length* from *RTG structure*.
- **C (history, DT-style RTG)**: T=k real steps, positions 0..k-1, RTG
  decremented by realized improvement per the DT paper.

k = 8 (the trained position range). Real history is taken from an actual MF-DRO
run's own query sequence; where fewer than k real steps exist, the sequence is
left-truncated as the DT paper does.

## Manipulation check (pre-registered, must pass before ANY interpretation)

The RTG channel must actually differ between arms:
- B vs A: number of RTG tokens the readout attends over goes 1 -> 8.
- C vs B: realised RTG values within a single forward pass must span
  **>= 3x** (max/min over the k positions), averaged over probes.

If C's within-pass RTG span is < 3x, arm C is VOID (same guard that killed H9
and H10). A and B remain interpretable in that case since their manipulation is
structural (token count), not magnitude-based.

## Locked predictions

1. **PRIMARY**: under arm C, sweeping the RTG target moves the argmax on
   **> 30%** of candidate pools, vs a measured 0/12 under arm A.
2. **SECONDARY**: arm B moves the argmax on *fewer* pools than arm C. If B ~ C,
   the effect is context length, not RTG structure, and prediction 1 must be
   restated accordingly.
3. **NULL-GUARD**: if all three arms sit at ~0%, the conclusion is that the DT is
   genuinely insensitive to RTG *and* to its own history — a stronger and
   cleaner negative than anything H8-H10 produced, and it closes the confound.

## What this does NOT test

Whether real-history inference improves *regret*. Positions 1..k-1 were trained
on FANTASY states (`sample_fantasy`); feeding real states there swaps one
train/inference mismatch for another. This measures the conditioning channel
only. A regret claim would require prefixing real steps into training rollouts
too, and is explicitly out of scope here.

## Evaluation untouched

`PROTOCOL.md` is not modified. No run in this experiment contributes to the
frozen success test.
