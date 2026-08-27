# H87 — clean confirmation of the Hartmann flip at fresh seeds

LOCKED BEFORE ANY RUN.

## Why this experiment exists

The Hartmann result (MF-DRO + calibrated ROI 5.93% vs MF-MES 6.62%, 4/5 paired
seed wins on h83's own bar) has one weakness that no amount of re-analysis can
remove: **three ROI settings were run on Hartmann and the winner is reported**.
q=0.10 beat fixed beta=2 and q~0.49, both of which were WORSE than no ROI at all
(+1.56 and +6.32 pts). Selecting the best of three inflates an apparent effect.

Partial defences already on record, neither sufficient: q=0.10 was not selected
on Hartmann alone (it also won Borehole), and the paired difference is tight
(sd 0.45 against marginal spreads of 3.26/3.39, i.e. 3.39 s.e. from zero), which
rules out seed noise but not selection.

Only one thing settles it: fix the setting in advance, run seeds never used
before, and report whatever comes out.

## Design

| | |
|---|---|
| benchmark | Hartmann_6D only |
| arms | MF-DRO + ROI-Q10, and MF-MES |
| seeds | **47, 48, 49, 50, 51** -- never used in h83, h84 or h86 |
| config | q=0.10 FIXED IN ADVANCE. No other ROI setting will be run. |
| everything else | identical to h83: M=3, pool 600, refinement off, budget 200 post-init, regression head |

MF-MES must be re-run at the new seeds because h83 only covers 42-46. It is
cheap (~2 min/run on Hartmann against ~100 min for MF-DRO).

ONE ARM ONLY. If q=0.10 underperforms here, that is the result. No alternative
ROI setting will be run on these seeds afterwards and reported instead -- doing
so would recreate exactly the selection problem this experiment exists to
remove.

## Metric (frozen)

h83's metric, via h83's own sr_curve/grid: simple regret grid-interpolated at
exactly cost 200, expressed as relative regret. Analysis reuses
experiments/h86-roi-full/code/analyse.py's at200().

## Prediction (pre-registered)

**P1. The paired difference (MF-DRO+ROI minus MF-MES) is negative on >= 4/5 of
the fresh seeds, and the paired mean is negative.**

I expect this to HOLD. Stating the expectation explicitly, and the reasoning,
because my recent record justifies scrutiny: Lesson 23 records five refuted
mechanism claims in h84, four of which erred by UNDERESTIMATING the
intervention. This prediction goes the other way, and rests on measurement
rather than argument -- the paired differences on seeds 42-46 occupy a
0.36-point band with sd 0.45, which is not the signature of a noise-mined
result.

**P2. The margin shrinks.** The fresh-seed paired mean is expected to be
smaller in magnitude than -0.68 pts, because selection over three arms inflates
the original. If the margin holds at -0.68 or grows, selection was not
materially distorting it.

**P3 (the bar that matters).** MF-DRO+ROI beats MF-MES on h83's full bar --
strictly lower mean AND >= 4/5 seeds -- at the fresh seeds. This is the claim
the talk would make, tested once, cleanly.

## Falsifier

If P1 fails, the Hartmann flip does not survive a clean test and must be
withdrawn from the report and from findings.md as prominently as it was
announced. The h64 retraction in this project is the precedent for how that is
done.

## Gate

Launch only when compute frees (h84 and h86 must finish first; <= 15 workers).
