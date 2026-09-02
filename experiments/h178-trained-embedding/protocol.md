# h178 -- does the TRAINED embedding saturate, as the structural argument says?

STATUS: protocol locked, nothing run. TYPE: CONFIRMATORY.
This is the check h177 registered and did not do.

## What needs settling

h177 measured that the emitted action is **exactly** invariant to BTG (spread
0.0000 across 357 iterations) while RTG moves it 9%. The explanation offered was
structural: raw scalars into `Linear(1->H)` then `LayerNorm` saturate, and over
each scalar's operating range the relative embedding change is **0.4869 for RTG
(0.30-1.00) against 0.0056 for BTG (26.1-30.5)** -- 87x apart.

**That was computed on RANDOM weights.** A trained `Linear` could in principle
place its bias so the operating point is not saturated. The explanation is
labelled EXPLORATORY precisely because of this, and it should not stay that way
when the check is a few lines.

## The measurement

Extend the probe to record, at every real iteration, the TRAINED modules'
response directly:

    rtg_resp = || reward_ln(reward_embedding(r_hi)) - reward_ln(reward_embedding(r_lo)) ||
               / || reward_ln(reward_embedding(r_lo)) ||     over r in [0.30, 1.00]
    btg_resp = same with btg_ln(btg_embed(.))                over b in [26.1, 30.5]

Read-only, no RNG consumed (pure forward passes on two scalars), inside the
existing RNG-protected block.

## Predictions

P1 The trained responses match the structural estimate: rtg_resp ~0.49,
   btg_resp ~0.006, ratio ~50-100x.
P2 They do NOT match -- e.g. btg_resp is comparable to rtg_resp. Then the
   saturation explanation is WRONG for the trained network, and h177's exact zero
   needs a different cause.

## What this can RETRACT

R1 P2 holds -> **h177's architectural explanation is retracted.** It would be the
   seventh account to fall on this front, and the exact-zero would return to
   unexplained. The measurement (BTG inert) stands either way; only the *reason*
   is at stake.
R2 P1 holds -> the explanation moves from EXPLORATORY to measured on the trained
   network, and the proposed fix (standardise the conditioning scalars) rests on
   a measurement rather than an argument.

## Sanity check before any number is read

SC1 the probe populates with the two response fields (smoke test pre-launch --
    h169 was lost to skipping this, and the probe swallows exceptions).
SC2 bit-identity of the default path.

## Design

Hartmann_6D RANDOM-POOL seeds 42-46 -- the same arm as h168 and h177, so the
responses can be set against those runs' measured action movements directly.
5 workers.
