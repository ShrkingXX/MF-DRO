# H125 — Does ROI tightness change the outcome, across a 5x range?

STATUS: LOCKED before any outcome statistic was computed.
TYPE: CONFIRMATORY. **The locked prediction is a NULL**, stated with grounds.
COMPUTE: zero new runs. Re-analysis of h84.
DATA: h84 ONLY — Borehole_8D and Hartmann_6D, seeds 42-46, paired within seed.

## The contrast, chosen from MEASUREMENT not from arm names

Realized acceptance (`roi_summary.accept_frac`, measured today):

  ROI-Q10   0.100 exactly, every seed, BOTH benchmarks
  ROI-ANN   0.493 (Borehole) / 0.498 (Hartmann), every seed
  ROI-FIX2  0.214 (Borehole) / 0.129 (Hartmann), per-seed range 0.036-0.265

PRIMARY CONTRAST: **ROI-Q10 (q=0.100) vs ROI-ANN (q~0.495)** — a 5x tightness
range, both realized to three decimals on every seed, paired within seed.

ROI-FIX2 is EXCLUDED from the primary and reported separately. Justification is
the measurement above, not convenience: at a fixed beta its acceptance floats
over a 7x range across seeds, so it is not a tightness setting at all and cannot
be a rung on a ladder. Excluding it is a decision made from measured data and
recorded here before any outcome is computed.

NOTE on what ROI-ANN is: per today's bug finding it is NOT an annealed arm. Its
schedule never ran (`_prog = n_real_iter/bo_iterations`, denominator 4000 against
cost-terminated runs). It is a constant q~0.495 arm, and it is used here purely
as the loose end of the ladder. Any reading of h125 as testing a schedule is
wrong; h123 is the schedule test and has not run.

## Measures (per benchmark, paired per seed, n=5)

  A. `final_regret` (stored in every result file).
  B. waste_frac — fraction of non-init HF queries below the best initial-design
     HF point, as h121.

## Prediction (locked): NULL

P1. ROI-Q10 does NOT differ from ROI-ANN on final_regret at >= 4/5 seeds with
    |mean|/sd >= 1.0, on EITHER benchmark.
P2. Same for waste_frac.

Grounds, and they are specific: h97/h107/h110 found q=0.05 vs q=0.10 not
separable across three seed sets; h111 found no tightness effect off Borehole;
h118 found the ROI barely moves boundary mass (14.7% -> 18.6%) and failed its
waste gate at 0.62. **Tightness has been a null axis everywhere it has been
measured properly.** This protocol tests it across 5x rather than 2x, which is
the widest range the existing data supports, and expects the same answer.

P3. If either contrast DOES separate, that is a positive result against the
    stated expectation, must be reported at least as prominently as the null,
    and needs confirmation at fresh seeds before being believed — two benchmarks
    x two measures is four tests, so one clearing a 1.0 bar is weak.

## Why a predicted null is worth computing

The primary question asks for an ROI *strategy*. If outcome is flat across a 5x
acceptance range on both benchmarks, then there is no tightness strategy to
find, and the search should move to what the ROI's parameterisation cannot
reach — which is where the peer's h124 (changing the DRAW, not the filter) and
h123 (the paper's actual schedule) are aimed. A measured null here is what
justifies abandoning the tightness axis rather than assuming it is exhausted.

## Limitations

- n=5 per benchmark, no p-values.
- Two arms of a four-arm experiment; ROI-FIX2 reported but excluded from the
  primary, on measured grounds stated above.
- h84's ROI-OFF control is incomplete (seeds 42-43) on BOTH benchmarks, so this
  compares two ROI settings to each other, NOT to no-ROI. The h120/h122 control
  completions in flight will supply that separately.
- final_regret is the frozen evaluation and is not recomputed here.
