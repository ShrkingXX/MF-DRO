# H128 — analysis

**BOTH LOCKED PREDICTIONS FAIL, and the cause is an error in my own protocol.**

Borehole, seeds 42-46, paired. Read point: raw regret @ `cost_curve` 200.

| arm | raw mean | vs control (raw) | vs control (rel% of optimum) | \|m\|/sd | better |
|---|---|---|---|---|---|
| control (h83 MF-DRO) | 48.959 | — | — | — | — |
| tight q=0.100 | 35.882 | **-13.077** | **-4.22%** | **1.74** | 5/5 |
| loose q=0.493 | 44.900 | -4.060 | -1.31% | 0.54 | 3/5 |

P1 (loose WORSE than control, >=4/5, effect >=1.0): **FAIL** — loose is BETTER
by 4.06 raw, in 3/5 seeds, effect 0.54.
P2 (magnitude +2 to +8 pts): **FAIL** — got -4.060.

## Why I predicted wrongly: I mixed units inside the locked protocol

The motivating arithmetic was "the ROI helps by 3.5-4.2 pts, loosening costs
+9.018, so loose should be ~+5 worse than control". Those two numbers are in
**different units**. The established 3.5-4.2 figure is a **percentage of the
optimum**; h125's +9.018 is **raw regret**. Adding them is meaningless.

Converted consistently (Borehole optimum 309.576):

  tight vs control   -13.077 raw = **-4.22%**   <- reproduces the established 3.5-4.2%
  loose vs tight      +9.018 raw = **+2.91%**   <- h125
  loose vs control    -4.060 raw = **-1.31%**

So the "3.5-4.2 pts" on record is confirmed by an independent route: measured
here at 4.22% with effect 1.74 in 5/5 seeds.

**CORRECTION (peer's refinement, accepted).** "Never a real prediction"
overstates it. The identity governs the MEANS only; per-seed sign counts and
effect sizes are not recoverable from it, so "3/5 better, effect 0.54" was real
information about the paired structure. What was empty was the COMPOSITION
CLAIM, not the test.

**And P2 was never a real prediction.** (loose - control) = (tight - control) +
(loose - tight) is an arithmetic identity over the same three means. I registered
a test of an identity and called it a composition check. It could only ever
"pass" or reveal my own unit error, which is what it did.

This is the fifth unit/statistic mismatch logged today and **the first one I
committed inside a locked prediction rather than caught in someone else's
number.** The rule I proposed to the peer this morning — name the statistic and
the read point in the number — would have prevented it, had I applied it to the
figure I was quoting from our own record rather than only to new results.

## What is actually established

**A loose ROI does not reverse the benefit; it gives most of it away.**

  no ROI          baseline
  q = 0.100       -4.22%   effect 1.74, better in 5/5   <- the useful setting
  q = 0.493       -1.31%   effect 0.54, better in 3/5   <- ~69% of the gain lost

So the corrected claim for the primary question is narrower than the one I
predicted and still worth stating: setting the ROI loosely costs roughly
**69% of its benefit** (4.22% -> 1.31%) and drops it below separability, but it
does not make the heuristic actively harmful. "Tune it or lose most of it",
not "tune it or be worse off than without it".

## Limitations

- n=5, one benchmark, no p-values.
- The loose arm is ROI-ANN, a constant q~0.493 only because its annealing never
  ran. h128 tests no schedule.
- Control substitution (h83 MF-DRO for ROI-OFF) under the measured-equivalence
  exception in h120 Amendment 3.
- The -1.31% for the loose arm is NOT separable (0.54, 3/5). The honest reading
  is "no longer distinguishable from no-ROI", not "still helps a little".
