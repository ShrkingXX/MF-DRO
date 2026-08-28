# h138 — Does the best configuration close the DIAGNOSIS'S OWN gap, on the benchmark where it works?

STATUS: LOCKED before computing. ZERO NEW COMPUTE.
TYPE: CONFIRMATORY.
STATISTIC / QUANTITY: mean HF query score via h84 `analyse.py:score`, the
founding diagnosis's own formula `(y - best_init_HF_y) / (-y_opt - best_init_HF_y)`
over post-init HF queries. Also `frac_neg`, its "worse than the initial design"
statistic. Borehole, seeds 42-51 (n=10), paired.

## Why

The founding diagnosis is stated in one metric — **mean HF query score, MF-DRO
0.336 against MF-MES 0.747** — and one benchmark, Hartmann. Everything since has
established that the interventions work on **Borehole** and not Hartmann.

**Nobody has computed the diagnosis's own metric for the best configuration on
the benchmark where it works, against the baseline the diagnosis named.** h129 P4
did control-vs-ROI (0.381 -> 0.495, effect 2.66) but never brought MF-MES in, and
never used the composite arm. So the question the whole investigation was set —
does this stop MF-DRO wasting HF budget relative to the method that doesn't — has
no answer in the diagnosis's own terms on the benchmark that matters.

Arms, all seeds 42-51, no new runs:
  control  h83 `MF-DRO` 42-46 + h90 `NO-ROI` 47-51
  ROI+L1   h113 `ROI-L1` 42-51
  MF-MES   h83 42-46 + h115 42-51(47-51)

## Predictions (locked)

**P1.** ROI+L1's mean HF query score exceeds the control's: effect >= 1.0.
FALSIFIED if effect < 1.0. (Partitioned; calibrated by h129 P4's 2.66 on the same
statistic and benchmark, ROI-alone.)

**P2 — three-way, and the residual is named for what it is.** Let
`d = ROI+L1 - MF-MES` on mean HF query score.
- **CLOSES THE GAP** if `d.mean() > 0`, effect >= 1.0, >= 8/10 seeds higher.
- **STILL BELOW** if `d.mean() < 0`, effect >= 1.0, >= 8/10 seeds lower.
- **NOT SEPARABLE** otherwise.

The residual is `NOT SEPARABLE`, **not "TIED"**. h137's gate returned "TIED" for a
result that was 8/10 unfavourable, and the word did work the thresholds did not.
Partitioning is necessary and not sufficient: a residual category must be named
for the evidential state it represents.

**P3 (no direction).** `frac_neg` — the diagnosis's "worse than the initial
design" number — for all three arms, reported whatever it shows.

## What this could RETRACT

1. **"The ROI moves the channel the founding diagnosis prescribed"** (h129 P4). If
   ROI+L1 does not exceed the control here at n=10, that claim rested on n=5 and
   one arm, and I should say so.
2. **My framing that the diagnosis "identified a real property".** If MF-MES's
   Borehole query score is *not* far above MF-DRO's, then the 0.336-vs-0.747 gap
   is a Hartmann fact and the diagnosis never described Borehole at all — which
   would make every Borehole mechanism result an answer to a question nobody
   asked.

**Asymmetric risk:** P1 passing and P2 returning STILL BELOW confirms the story I
already hold. That is the outcome to scrutinise hardest.
