# H128 — Is a badly-set ROI WORSE than no ROI at all?

STATUS: LOCKED before any statistic was computed.
TYPE: CONFIRMATORY.
COMPUTE: zero new runs.
READ POINT: raw regret @ `cost_curve` 200 (the post-init budget the worker
terminates on). Stated per the convention agreed with the peer session today.

## The arithmetic that motivates it

Two established Borehole results point the same way:

  control -> q=0.10   the ROI HELPS by 3.5-4.2 pts (sd 0.37, 9-10 of 10 seeds)
  q=0.10  -> q=0.493  loosening COSTS +9.018 pts (h125, effect 5.69, 5/5)

If both hold and roughly compose, q=0.493 should land about **5 points WORSE
than not using an ROI at all**. That has never been checked directly, and it
is a materially different claim from "tightness matters": it says the
heuristic is not merely less effective when mis-set, but actively harmful.

## Data

Borehole_8D, seeds 42-46, paired within seed:
  control  = h83 `MF-DRO`  (complete 5/5; established bit-identical to h84's
             `ROI-OFF`, 137 and 132 queries, 0 differing, three commits)
  loose    = h84 `ROI-ANN` (measured realized q = 0.493 every seed)
  tight    = h84 `ROI-Q10` (measured realized q = 0.100 every seed) — reported
             as the reference point, not part of the primary test.

Cross-experiment control substitution under the measured-equivalence exception
already recorded in h120 Amendment 3.

## Prediction (locked)

P1 (PRIMARY). ROI-ANN is WORSE than the no-ROI control on Borehole, in >= 4/5
    seeds, with |mean|/sd >= 1.0.

P2. The magnitude is near +5 pts, i.e. the two established effects roughly
    compose. Registered as a quantitative expectation so that a large deviation
    is visible as a failure of composition rather than absorbed silently.
    Tolerance: within +2 to +8 pts. Outside that range, composition fails even
    if P1 passes, and that must be reported.

## Why the answer is not already known

h125 compared two ROI settings to each other and never to the control. The
3.5-4.2 pt benefit was measured at q=0.10 only. Nobody has run the loose setting
against no-ROI. The two numbers have simply never been put in the same frame.

## What a pass would mean for the primary question

It converts the tightness finding from "choose the setting well" into "an
ROI set loosely is worse than omitting the heuristic". For a paper that
recommends the DRO Sec 4.2 ROI, that is the difference between a tuning note
and a warning.

## What a failure would mean

If ROI-ANN is NOT worse than control, then the two established effects do not
compose, and at least one of them is measuring something narrower than assumed —
most likely that the 3.5-4.2 pt benefit is specific to q=0.10 rather than being
an "ROI vs no-ROI" effect at all.

## Limitations

- n=5, one benchmark, no p-values.
- ROI-ANN is a constant q~0.493 arm ONLY because its annealing never ran
  (today's bug). It is used here purely as the loose setting. h128 tests no
  schedule.
- Composition of two separately-measured effects is an expectation, not a
  theorem; P2 exists to make its failure visible.
