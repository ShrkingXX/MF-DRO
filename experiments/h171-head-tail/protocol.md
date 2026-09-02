# h171 -- HEAD vs TAIL: the mechanism's interventional test

STATUS: protocol locked, nothing run. TYPE: CONFIRMATORY.

## Why this and not more correlation

h170 showed the DT's query sits 3.3x closer to its teacher's **tau=0** action
mean than to the box centre. That is correlational. Every account that has
fallen on this front (five of them) also fitted the correlations available at
the time.

The tau=0 mechanism makes an **interventional** prediction that no previous
account makes, and that is sharply falsifiable: **only the teacher's first step
reaches inference.** If true, the other seven steps of an eight-step rollout are
irrelevant to the real query.

## The two arms

  HEAD-MES   tau=0: MES argmax (the control's rule).  tau=1..7: uniform random.
  TAIL-MES   tau=0: uniform random.                   tau=1..7: MES argmax.

TAIL-MES is a *better* teacher by any conventional measure -- seven of its eight
steps follow the acquisition, against HEAD-MES's one.

## Predictions (opposed, and against intuition)

MECHANISM: **HEAD-MES works** (near the control's 15.82, improving ~5/5) and
  **TAIL-MES fails** (near 43.94, improving ~0/5) -- despite TAIL-MES being the
  better teacher on 7 of 8 steps.
ANY TRAJECTORY-QUALITY ACCOUNT: the reverse ordering, or both intermediate.

## What this can RETRACT

R1 HEAD-MES FAILS -> tau=0 alone is NOT sufficient. The mechanism is wrong and
   becomes the sixth account to fall. h170's 3.3x would then be a correlate of
   something the first step happens to track, not the operative channel.
R2 TAIL-MES WORKS -> tau=0 is NOT necessary. Same conclusion, from the other
   side, and h169/h170's whole framing is withdrawn.
R3 Both as predicted -> the mechanism is confirmed by intervention, which is a
   different and much stronger evidence class than the seven-arm sign agreement
   or h170's distances. It would also mean **7/8 of every rollout is wasted
   computation**, which is directly actionable.
R4 Both intermediate (25-35 rel%) -> inconclusive at n=5, reported as such.

R1 and R2 are each individually fatal to the account and are named first.

## Sanity checks, read before the numbers

SC1 realised HF fraction against the control's 0.88 -- a collapse voids the arm
    (h60's thompson precedent).
SC2 the tau=0 action in HEAD-MES must actually be the MES argmax, and in
    TAIL-MES must actually be random -- verified by a smoke test BEFORE launch,
    because both branches are new code and a mis-wired split would silently
    produce two copies of the same arm. **h169 was lost to exactly this.**
SC3 bit-identity of the default path (rollout_policy="mes").

## Design

Borehole_8D seeds 42-46, n=5 each, frozen metric, no p-values. 10 workers
alongside h166's 3 = 13 <= 15.
