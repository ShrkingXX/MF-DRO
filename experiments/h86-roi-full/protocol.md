# H86 — does the calibrated ROI change the h83 headline?

LOCKED BEFORE ANY RUN. Human-directed: "if you find the ROI improving the
performance of the MF-DRO, rerun all h83 MF-DRO experiments."

## Scope: 10 runs, not 20

h84 already ran MF-DRO + ROI-Q10 on Hartmann_6D and Borehole_8D at seeds 42-46,
under the identical configuration this experiment would use. Those ARE the
reruns for those two benchmarks and are not repeated. H86 runs only the two
benchmarks the ROI has never been tested on:

  Currin_2D, Ackley_10D  x  seeds 42-46  =  10 runs

The final table combines h84 (Hartmann, Borehole) with h86 (Currin, Ackley).

## Configuration

MF-DRO with `use_roi=True, roi_beta_mode='quantile', roi_target_accept=0.10`.
Everything else identical to h83: M=3, n_roi_candidates=600, refinement off,
budget 200 post-init, regression head (use_candidate_scoring=False), same
initial designs.

## Why q=0.10 and not something else -- stated as a limitation

q=0.10 was a guess that happened to work, not a tuned optimum. h84 shows tight
beats loose on both benchmarks it tested, and the pre-result teacher
measurements imply a turning point below q=0.10 (at q=0.02 the teacher's
closest-ever approach to x* degrades from 0.022 to 0.110). The optimum for q is
therefore somewhere in (0.02, 0.10) and is UNLOCATED. Carrying q=0.10 to two new
benchmarks tests GENERALISATION of an untuned setting, which is the honest test;
it is not a claim that 0.10 is right.

## What h84 established, as the basis for this experiment

  ROI-Q10 vs ROI-OFF, paired, 5 seeds each:
    Borehole_8D   d(rel.regret) = -4.22 pts  (better 5/5)   15.82% -> 11.59%
    Hartmann_6D   d(rel.regret) = -1.62 pts  (better 3/5)    7.55% ->  5.95%

  P1 (mean query score, +0.10 on >=4/5, BOTH benchmarks) FAILED: Borehole met it
  (+0.114, 5/5), Hartmann did not (+0.001, 3/5). The improvement is in the UPPER
  TAIL of the query distribution, not the bulk (Lesson 22).

## The question this experiment exists to answer

h83's headline was that MF-DRO beats no baseline on any benchmark. On two
benchmarks it was CLOSE:

  Ackley_10D   SF-DRO 3.43  vs  MF-DRO 3.83   (absolute SR; gap 0.40)
  Currin_2D    MI-Greedy 0.00% vs MF-DRO 0.01%

Borehole's ROI gain was a 27% relative reduction in regret. If Ackley saw a
similar proportional gain, 3.83 would fall to about 2.8 and MF-DRO would beat
SF-DRO there. That is the stake.

## Predictions (pre-registered, independent)

- **P1 (direction).** ROI-Q10 lowers MF-DRO's regret vs h83's ROI-OFF on BOTH
  Currin and Ackley, on >= 3/5 seeds each. Direction only, no magnitude.
- **P2 (THE HEADLINE).** MF-DRO + ROI-Q10 does NOT beat the best baseline on
  Currin. Registered as NEGATIVE: MI-Greedy is at 0.00% there, a ceiling that
  leaves nothing to win.
- **P3 (THE OPEN ONE).** On Ackley, MF-DRO + ROI-Q10 beats SF-DRO's 3.43
  absolute SR on >= 3/5 seeds. Registered as GENUINELY UNCERTAIN -- this is the
  one benchmark where the h83 headline could flip, and it is deliberately stated
  so that failure is visible rather than absorbed.
- **P4 (generalisation).** The ROI does not HARM either benchmark: no benchmark
  gets worse by more than 1 point of relative regret. A fix that helps two
  benchmarks and breaks a third is not a fix.

No p-values at n=5. Every run reported including failures.

## Gate

This experiment MUST NOT be launched until h84's reproduction control passes.
If the control fails, h84's arm A is not h83's configuration, every delta above
is void, and this experiment has no basis. The control is 4 runs (2 benchmarks
x 2 seeds) comparing a live ROI-OFF re-run against h83's stored MF-DRO output;
it passes only on |d(regret)| < 1e-9 AND identical query traces.
