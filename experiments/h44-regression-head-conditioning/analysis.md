# H44 — is the DT input-inactive under the regression head?

**Status: PARTIAL** (the locked middle band). Confirmatory: the protocol was
committed before the probe ran.

## Question

Every "the DT is inert" result in this project (H5, H8, H9, H10, H11, h28's
0/12 argmax nulls, the 0.13% `||delta(s)||/||w_bar||`) was measured under
**candidate scoring**, where the DT emits a weight vector `w(h)` and the query
is `argmax_k <w(h), cf_k>` over 200 fresh uniform candidates. H42 then showed
the regression head (`x = action_head(h).clamp(0,1)`) does not re-freeze the
incumbent. That is a *behavioural* result and does not answer whether the DT
conditions -- queries move whenever the weights re-fit, and movement of queries
is exactly the evidence that misled this project before.

So: hold the weights fixed, vary the inputs, measure how far the proposal moves.

## Setup

Hartmann 6D, seed 42, 20 iterations, `initial_hf=6`, `initial_lf=45`,
`use_candidate_scoring=False`. One DT trained normally; real states / rtg / btg
recorded through a `propose_mf` wrapper. Then weights frozen and three sweeps
run, each varying one channel over the band that channel actually realised
during the run.

`||dx||` is normalised by the run's own mean pairwise query spread, so the
question is "large relative to how much this run's queries moved anyway", not
"large in absolute units".

**Sensitivity caveat, stated in the protocol before running**: this probe is
strictly more sensitive than the candidate-scoring probes. `argmax_k` over a
discrete pool absorbs small changes in `h`; `action_head(h)` is continuous and
must move unless its input weights are exactly zero. A nonzero reading was
therefore expected. The locked thresholds are what carry the claim.

## Locked predictions

- INACTIVE if `< 10%` of run spread on all three channels
- ACTIVE if any channel `> 50%`
- otherwise report the ratio and claim neither

## Result

Run's own mean pairwise query spread (normalised): **0.244892**.
Iterations 20, distinct queries **20/20**.

| channel | swept over | max `||dx||` | mean `||dx||` | max / run spread |
|---|---|---|---|---|
| state | 20 real states | 0.122096 | 0.083635 | **49.9%** |
| RTG | [0.562, 1.524] | 0.092748 | 0.060920 | **37.9%** |
| BTG | [23.8, 49.5] | 0.002042 | 0.001145 | **0.8%** |

Largest effect 49.9% -> **PARTIAL**. Neither locked verdict fires.

## What this does and does not establish

**Does**: under the regression head the map `h -> x` is non-degenerate in state
and in RTG. Swapping the state across the run's own realised states moves the
proposal by roughly half the distance the run's queries moved on their own.
RTG moves it by ~38%. This is the first non-null RTG probe in the project --
H8, H9, H10 and H11 were all exact nulls under candidate scoring.

**Does not**: say the movement is *useful*. Direction was not measured, regret
was not measured, and this is a single seed and a single training run. A head
that responds to its inputs in an arbitrary direction is input-active and still
worthless.

**Does not**: license a numeric comparison against "0/12". Different
instruments (see the sensitivity caveat). What is comparable is qualitative:
under candidate scoring the state's only channel to the decision was 11 numbers
scaled by `w(h)`, and the state-dependent part of `w` was 0.13% of its mean, with
`w_bar` alone reproducing every decision. Under the regression head the state
reaches the proposal.

**BTG is inert under both heads** (0.8%). Consistent with every earlier
budget-conditioning null.

## Consequence for the write-up

`FOR_ADVISOR.md` and the paper state the DT is inert. That claim was measured
entirely under candidate scoring and now needs an explicit scope qualifier: it
is a statement about the *linear score head*, not about the DT architecture.
Whether the regression head's responsiveness translates into better regret is
a separate, unrun experiment.
