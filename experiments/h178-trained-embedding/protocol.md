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

## SC1 failed, was fixed, and passed — and the early reading is NOT informative

**SC1 first FAILED**: `RuntimeError('mat1 and mat2 must have the same dtype')`.
The probe used `state.dtype` (float64) against float32 modules — the DT runs in
float32 because `propose_mf` calls `state.float()`. Caught by the smoke test
before the arm ran; fixed to follow the module's own weight dtype. **Fourth time
this session a pre-launch smoke test has caught something that would otherwise
have surfaced as silently-missing data.**

After the fix, SC1 **PASSES** and SC2 (bit-identity) passes (122.2906675273).

**The early reading is recorded but explicitly does not count as evidence.** At
5 iterations it gives rtg_resp 0.4974 and btg_resp 0.0051, ratio 98.2× — a near-
perfect match to the random-weight estimate (0.4869 / 0.0056 / ~87×).

**That match is close to tautological.** After five iterations the DT has had few
gradient updates, so its embedding weights are still near initialisation, and
agreeing with a random-weight calculation is exactly what an untrained module
must do. **The hypothesis is only tested by a fully-trained network**, which is
what the full arm provides. Stating this now so the full-run numbers are not read
as merely confirming what the smoke already showed.
