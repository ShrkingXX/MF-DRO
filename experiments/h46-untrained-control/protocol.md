# H46 — is h44's responsiveness LEARNED, or just generic MLP sensitivity?

## The gap this closes

h44 measured, under the regression head, how far `x = action_head(h)` moves when
the inputs are swept with weights frozen:

| channel | max / run spread | mean / run spread |
|---|---|---|
| state | 49.9% | 34.2% |
| RTG | 37.9% | 24.9% |
| BTG | 0.8% | 0.47% |

h44 has **no control**. Any network with non-zero weights maps different inputs
to different outputs -- that is a property of the architecture, not evidence of
learned conditioning. Without a reference point, 49.9% is uninterpretable.

This is the same error class as the h5 AUDIT, where an h-swap compared a state
against itself and produced 12/12 "identical" as though it were a finding.

## Design

Re-run the h44 setup verbatim (Hartmann 6D, seed 42, 20 iters,
`use_candidate_scoring=False`, `initial_hf=6`, `initial_lf=45`), recording the
real tau=0 states / RTG / BTG as before. Then run the **identical three sweeps
twice**:

- **arm T (trained)** -- the DT that was actually trained during the run. This
  reproduces h44 and serves as the internal consistency check.
- **arm U (untrained)** -- a second DT built from the **same config**, therefore
  the same architecture, left at random initialisation. Swept over the **same
  recorded states**, normalised by the **same run spread**.

Both arms are put in `eval()` and swept under `torch.no_grad()` with
`fidelity_sampling=False`, so the only difference between them is training.

The recorded states are **saved to disk** this time (`states.npz`), so no future
control needs another run.

## Locked predictions

Let `R_T` and `R_U` be the state-channel max/run-spread ratios.

1. **GENERIC (h44 is an artifact)**: `R_U >= 0.7 * R_T`. An untrained network is
   about as responsive as the trained one; h44's number says nothing about
   learning and the h44 headline must be withdrawn as evidence of conditioning.
2. **LEARNED**: `R_U <= 0.3 * R_T`. Training substantially created the
   responsiveness.
3. **SUPPRESSED**: `R_U >= 1.5 * R_T`. Training *reduced* responsiveness -- the
   network is learning toward a state-insensitive rule, which would be the
   regression-head analogue of the score head's contraction (h21, h23).
4. Otherwise **AMBIGUOUS**: report both ratios and claim neither.

Reported for all three channels; the state channel decides the verdict.

## What this cannot settle

Direction and usefulness. A trained head that is more responsive than an
untrained one is still worthless if it moves the query the wrong way. Regret is
h45's job, not this one.

## Cost

One 20-iteration Hartmann run, single worker, 1 thread. Runs alongside h45's 10
workers: 12 processes total, within the 15-core rule.
