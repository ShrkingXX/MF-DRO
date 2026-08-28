> **ID DISAMBIGUATION (added 2026-08-27).** The id `H42` names TWO distinct
> protocols in this tree. This file is `h42-fixed-rule-control` (this one has NO results -- it was registered and abandoned).
> The other is `h42-regression-freeze` — "does the REGRESSION head freeze the incumbent on Hartmann 6D?" (3 result files).
> They were registered minutes apart by the same session, which reissued the
> number because nothing checked. Any bare reference to "H42" elsewhere is
> ambiguous and must be resolved by content. Numbers are now claimed via
> `tools/claim_id.sh`, which blocks reissue.

# H42 — is "not frozen" caused by the pipeline, not the DT?

## The claim under test

In the candidate-scoring arm the query moves between iterations. Two candidate
explanations:

  (a) the DT computes something useful per iteration, or
  (b) 200 FRESH uniform candidates are drawn each iteration and their GP
      features update as data accumulates -- so even a FIXED rule would move.

H23 already shows the DT's state-dependent part is 0.13% of the coefficient
vector and that `w-bar` alone reproduces the argmax 12/12. That makes (b) the
live explanation, but it has never been tested end-to-end in a real run.

## Design

Replace the DT's scoring entirely with a **fixed, hand-written acquisition**:

    score_k = mu_H(x_k)          # pure greedy on the GP mean, no DT, no learning

over the SAME 200 fresh uniform candidates per iteration, the same GP ensemble,
the same initial design, the same cost budget. Fidelity is drawn at the same
rate the DT arm realises, so cost profiles are comparable.

Nothing is trained. If the incumbent still moves, the movement was never the
DT's doing.

## Locked prediction

1. **PRIMARY**: the fixed-rule control shows **> 0 incumbent improvements on at
   least 8/10 seeds** -- i.e. it does not freeze either, and "not frozen" is a
   property of the pipeline (fresh pools + updating GP), not of the DT.
2. **FALSIFICATION**: if the fixed rule freezes on >= 3 seeds while the DT arm
   does not, then the DT *is* contributing something to incumbent progress and
   claim (a) survives.

## Scope

This tests the FREEZE question only. It is not a regret comparison -- a greedy
mu_H rule is expected to be worse than MES on regret, and that is not evidence
about the freeze.
