# H86 analysis — 10/10 runs, 0 failures. The ROI is not a uniform improvement.

Completes the h83 MF-DRO rerun. h84 covered Hartmann and Borehole under the
identical configuration and seeds; h86 added Currin and Ackley.

## The full four-benchmark picture, h83's own metric and bar

| benchmark | MF-DRO h83 | +ROI-Q10 | delta | wins vs baseline | best baseline | verdict |
|---|---|---|---|---|---|---|
| Currin_2D | 0.01 | **0.13** | **+0.11** | 0/5 | MI-Greedy 0.00 | no |
| Hartmann_6D | 7.99 | **5.93** | **-2.05** | 4/5 | MF-MES 6.62 | **BEATS** |
| Borehole_8D | 15.82 | 11.59 | -4.22 | 1/5 | MF-MES 6.40 | no |
| Ackley_10D | 3.83 | 3.74 | -0.09 | 1/5 | SF-DRO 3.43 | no |

**The ROI helps on two benchmarks, does nothing on one, and HURTS on one.**
Currin gets worse on 5 of 5 seeds. That is consistent directional harm, not
noise, even though the magnitude (+0.11 pts) is small in absolute terms -- and
small only because Currin's regrets are near zero to begin with; in relative
terms 0.01 -> 0.13 is a 13x degradation.

## Pre-registered bars

**P1 — FAILED.** Required ROI-Q10 to lower regret vs ROI-OFF on BOTH Currin and
Ackley, >= 3/5 seeds each. Currin 0/5, Ackley 1/5. Failed on both halves.

**P2 — MET.** Registered as NEGATIVE: MF-DRO+ROI does not beat the best baseline
on Currin. It does not (0.13 vs MI-Greedy's 0.00). The reasoning was that
MI-Greedy sits at a ceiling leaving nothing to win; the actual outcome is worse
than that -- the ROI moved MF-DRO away from the ceiling rather than toward it.

**P3 — FAILED.** Registered as GENUINELY UNCERTAIN: does MF-DRO+ROI beat
SF-DRO's 3.43 on Ackley, >= 3/5 seeds? 1/5. The h83 headline does not flip on
Ackley.

**P4 — MET, but only technically.** Required no benchmark to get worse by more
than 1 point of relative regret. Currin degrades by 0.11 pts, inside the bar.
The bar was badly chosen: an absolute-points threshold is meaningless on a
benchmark whose regrets are ~0.01, and it records "no harm" where the data show
harm on 5 of 5 seeds. Reported as met, and flagged as a bar that should have
been relative.

## Why Currin is hurt — hypothesis, not measured

Currin is 2-D and MF-DRO already solves it essentially perfectly without the ROI
(0.01% relative regret). There is no exploration problem to fix, so restricting
the candidate pool can only remove options. The ROI is a device for finding a
region worth exploiting; on a benchmark already exploiting the right region it
is pure constraint. Consistent with the ROI's benefit tracking how much the
uniform pool under-represents the optimum: large on Borehole (boundary optimum),
moderate on Hartmann, nil on Ackley (needle at the centre), negative on Currin
(already solved). NOT tested.

## What this does to the headline

The claim is now precisely: **a calibrated ROI moves MF-DRO past the best
baseline on Hartmann, improves it substantially on Borehole without closing that
gap, does nothing on Ackley, and harms it on Currin.** Anyone quoting "the ROI
fixes MF-DRO" is over-reading by two benchmarks.

The three-axis control argument (fixed beta cannot set tightness across
benchmarks, within a run, or across seeds) is unaffected -- it is about
controllability, not performance, and it holds regardless of which benchmarks
benefit.
