# H97 — is q=0.10 the right ROI tightness, or did we stop at the first value that worked?

LOCKED BEFORE ANY RUN. Numbering: session B holds h90+; peer session holds h94-h96,
so this takes h97 to avoid the collisions that hit h88/h89.

## Why now

The ROI's Borehole gain is CONFIRMED (h90: P1/P2/P3 met, pooled 9/10, 83%
retention). Every ROI result this project has rests on **q = 0.10**, and that
value was never optimised — it was the first calibrated setting tried. The
project's own state file lists the tightness question as unclaimed and UNLOCATED:

  - h84 found tighter beat looser on Borehole under FIXED beta
  - teacher measurements imply a turning point BELOW q=0.10: at q=0.02 the
    closest approach to x* degrades 0.022 -> 0.110

Those two together bracket an optimum somewhere in (0.02, 0.10) but no experiment
tests it. With the ROI now confirmed to work, "how tight should it be" is the
direct next question for this session's primary aim.

## Design

| | |
|---|---|
| benchmark | Borehole_8D (the only benchmark with a confirmed ROI gain) |
| seeds | 47, 48, 49, 50, 51 |
| new arm | **ROI-Q05** — identical to ROI-Q10 but roi_target_accept=0.05 |
| comparators | ROI-Q10 and NO-ROI at the SAME seeds, already run in h90 |
| runs | 5 (only the new arm; nothing is re-run) |

Reusing h90's two arms is legitimate here in a way it would not normally be: they
are the same code, same seeds, same worker, same commit. The h90 worker is invoked
unmodified with one config value changed. Verified before launch: h90's ROI arm's
own logged accept_frac is 0.0998-0.1000 against its 0.10 target, so the
calibration demonstrably fires — this arm's q=0.05 must show accept_frac ~0.05 in
its own logs or the run is void (see gate).

## Predictions

**P1.** ROI-Q05 beats NO-ROI (negative paired mean, >=4/5). Registered POSITIVE
and expected to be easy: q=0.10 achieved -3.49 at 4/5, and 0.05 is still a
region rather than a point.

**P2 — the actual question.** ROI-Q05 vs ROI-Q10 is registered as **GENUINELY
UNCERTAIN, no direction predicted.** The two pieces of evidence point opposite
ways: h84's fixed-beta result says tighter is better, the teacher measurement at
q=0.02 says too tight destroys reach. q=0.05 sits between them and I have no
basis for calling which side of the turning point it lands on. Six mechanism
predictions in this session were refuted; declining to guess is the honest
posture, not a hedge.

**P3.** ROI-Q05 does not make MF-DRO competitive with MF-MES on Borehole.
Registered POSITIVE. Every intervention so far has failed to close that gap and
nothing about tightening a region addresses boundary aversion.

## Gate (G3, adopted from a peer session's failure this evening)

A bit-identity gate on the OFF path is structurally incapable of catching a
broken ON path. Before reading any regret number, this arm's `roi_summary` must
show **accept_frac in [0.045, 0.055]** — the side effect OBSERVED, not merely
the absence of a crash. If it shows ~0.10 the flag did not take and the runs are
void regardless of what the regret says.

## What each outcome means

  - **Q05 beats Q10:** the optimum is below 0.10 and every ROI number this project
    reports is from an unoptimised setting. Worth locating properly.
  - **Q10 beats Q05:** 0.10 is at or near a local optimum from below, and the
    turning point the teacher measurement implies lies between 0.02 and 0.05.
  - **Indistinguishable:** the gain is robust to tightness over 2x, which is a
    more useful claim than a tuned optimum — it says the mechanism is the region,
    not the threshold.
