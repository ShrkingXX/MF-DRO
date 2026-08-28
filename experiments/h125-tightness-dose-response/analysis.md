# H125 — analysis

**THE LOCKED PREDICTION IS REFUTED.** I predicted a null on both measures and
both benchmarks. Three of four contrasts separate, and the primary one is the
largest effect measured anywhere in this project.

Data: h84 only, seeds 42-46, paired within seed, n=5. Zero new runs.
Contrast: ROI-Q10 (realized q = 0.100 exactly) vs ROI-ANN (realized q = 0.493
Borehole / 0.498 Hartmann). A 5x acceptance range, both realized to three
decimals on every seed.

| benchmark | measure | q=0.100 | q~0.495 | paired | sd | \|m\|/sd | dir | |
|---|---|---|---|---|---|---|---|---|
| Borehole | final_regret | 35.882 | 44.900 | **+9.018** | 1.585 | **5.69** | 5/5 | SEPARATES |
| Borehole | waste_frac | 0.030 | 0.062 | +0.033 | 0.026 | **1.28** | 5/5 | SEPARATES |
| Hartmann | final_regret | 0.197 | 0.303 | +0.105 | 0.104 | **1.01** | 4/5 | SEPARATES |
| Hartmann | waste_frac | 0.196 | 0.192 | -0.004 | 0.038 | 0.10 | 2/5 | no |

Secondary, pre-excluded on measured grounds (its acceptance floats 7x at fixed
beta): ROI-FIX2 vs ROI-Q10 separates on neither benchmark (0.35 and 0.25).

## Why the prior was wrong, and it is not a subtle reason

"Tightness has been a null axis wherever measured properly" rested on
h97/h107/h110 (q=0.05 vs q=0.10) and h111 (two settings spanning 2x). **Every
one of those is a 2x contrast or narrower.** This is 5x. The axis was not null;
the lever had never been moved far enough to see it.

Reading the two together gives a dose-response shape rather than a slope:

  q = 0.05 vs 0.10   flat, not separable across three seed sets (h97/h107/h110)
  q = 0.10 vs 0.495  +9.0 regret on Borehole, effect 5.69, 5/5

So there is a **plateau below ~0.10 and steep degradation by ~0.5**. Nothing here
says tighter-is-always-better; it says loose is bad and the useful region is
q <= 0.10, with no evidence of further gain below it.

## Is the contrast clean?

Yes. Both arms are quantile-calibrated; their only configuration difference is
the acceptance target. beta differs between them (1.86 vs 2.81 on Borehole) but
that is downstream, not a confound: the bisection solves for whatever beta hits
the requested acceptance, so beta differing IS the mechanism by which q differs.

One caveat carried forward: ROI-ANN is only a q~0.495 arm BECAUSE its annealing
never ran (today's bug). It is used here purely as the loose rung of a ladder.
Nothing in h125 tests a schedule.

## Multiplicity, honestly

Four tests were run and three separated, which the protocol pre-committed to
treating as weak if a single one had cleared the bar. Three of four is not that
case, and the primary Borehole regret contrast is 5.69 sd with 5/5 seeds — an
order of magnitude above the 1.0 bar. The two marginal ones (Hartmann regret
1.01, Borehole waste 1.28) should be treated as directionally consistent
support, not as independent findings.

## Consequence: h123's grounds are gone

h123 registers the paper's faithful `beta_t` as a WIDENING ROI, and its locked
prediction is a null justified by "tightness has been a null axis". **That
justification is now refuted, and worse, the direction it predicts is the one
h125 shows is harmful**: widening from 0.10 to 0.495 costs 9 regret points on
Borehole. h123's protocol must be amended before it runs — not to change the
prediction to fit, but because the grounds recorded for it are no longer true
and a reader would be misled.

There is a real tension to state rather than resolve by preference: GP-UCB's
theory says `beta_t` grows, which widens this ROI; this measurement says
widening hurts. Both can hold — the theory governs a confidence bound's validity,
not the usefulness of the induced acceptance set as a training-distribution
filter. h123 is worth running precisely because it puts a number on that gap.

## Limitations

- n=5, no p-values, one experiment.
- No data below q=0.05, so the plateau's left edge is unmeasured.
- Compares two ROI settings to EACH OTHER, not to no-ROI: h84's control is
  incomplete (seeds 42-43) on both benchmarks. The h120/h122 completions in
  flight address that separately.
- Hartmann's waste_frac null is consistent with h121: at 6-24 HF queries per run
  that statistic has very little resolution.
