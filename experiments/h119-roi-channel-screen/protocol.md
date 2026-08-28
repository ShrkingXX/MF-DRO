# H119 — What DOES the ROI change? A pre-enumerated screen.

STATUS: LOCKED before any statistic was computed.
TYPE: **EXPLORATORY SCREEN.** No gate, no confirmatory claim. Its output is a
      registered hypothesis for a later confirmatory test, not a conclusion.
COMPUTE: zero new runs. Re-analysis of h90-borehole-confirm.
DATA: h90 ONLY — Borehole_8D, seeds 47-51, arms NO-ROI and ROI-Q10, paired.
      REFINE-100 is present in h90 but is `use_roi=False` and is NOT an ROI arm;
      it is excluded from this screen entirely.

## Why a screen, and why declare it as one

The ROI's Borehole benefit is the most reproducible quantity in this project
(3.5-4.2 pts, sd 0.37, 9-10 of every 10 seeds). Two mechanistic explanations
have now been ruled out by pre-registered tests:

  - dispersion across dimensions (h116, gate missed 0.17 / 0.10)
  - boundary resolution (h118, gate missed 0.62 / 0.76)

I do not have a third mechanism to pre-register. Pretending otherwise would mean
inventing a hypothesis to dress a fishing expedition as confirmatory. So this is
declared a screen, and the price of that is: **nothing it finds counts as a
result.** Whatever comes out is a hypothesis that must then be tested on data
that played no part in generating it.

## The enumerated candidate list (fixed now, before computation)

Exactly these seven quantities are computed. This list is committed BEFORE the
numbers so that reporting "the one that separated" cannot be cherry-picking from
an open-ended search. All seven are reported with their paired effect sizes
whatever they show, including the ones that separate nothing.

  C1. Fidelity allocation — fraction of the cost budget spent at HF.
  C2. HF query count — number of non-init HF queries completed in budget.
  C3. LF query count — number of non-init LF queries.
  C4. Time-to-incumbent — cost consumed before the final best HF y is first
      reached, as a fraction of total budget.
  C5. HF query quality — mean y of non-init HF queries, standardised within
      seed by the initial design's HF spread (comparable across seeds).
  C6. Fraction of HF queries worse than the best initial-design HF point
      (the founding diagnosis's own statistic, computed here on Borehole).
  C7. Early-vs-late HF dispersion — weighted per-dimension sd of the first half
      of non-init HF queries divided by that of the second half. A contraction
      ratio: does the ROI make the search settle faster?

## Statistic

Paired per seed (ROI-Q10 minus NO-ROI), n=5. Reported as mean, sd, |mean|/sd,
and the number of seeds moving in the majority direction. NO p-values.

No threshold is a pass here. As an ordering aid only, effects are labelled
"separable" at |mean|/sd >= 1.0 — the same descriptive bar used elsewhere in
this project — with the explicit note that clearing it in a screen of seven
quantities means less than clearing it in a pre-registered test of one.

## Pre-committed interpretation rules

1. If NOTHING separates, that is the finding and it is reported as such: the
   ROI's benefit does not show up in any of seven natural channels, and the
   screen has failed to generate a hypothesis.
2. If exactly one separates, it becomes a registered hypothesis for a
   confirmatory test at fresh seeds. It is NOT reported as the mechanism.
3. If several separate, they are all reported. With seven correlated quantities
   at n=5, several clearing a descriptive bar is expected under no effect, and
   this will be stated rather than resolved by picking the largest.

## Limitations

- One benchmark, n=5, two arms. Borehole is where the ROI effect lives; this
  screen cannot say anything about generality.
- C1-C7 are not independent of each other; several share the same underlying
  run structure. No multiplicity correction is attempted, because at n=5 with
  correlated measures any correction would be theatre. The defence against
  false positives here is rule 2 — confirmation on fresh data — not arithmetic.
