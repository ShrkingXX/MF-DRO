# Spec: ROI containment instrumentation, so it is interpretable next time

Written for BOTH sessions to implement identically, so future ROI runs are
comparable. Not yet applied to `src/` — runs are in flight.

## The problem this fixes

`roi_stats` currently records, per ROI construction:

    min_dist_to_xstar   the closest accepted candidate to x*, UNWEIGHTED
    frac_within_0.2     fraction of accepted candidates within 0.2, UNWEIGHTED

Both are plain Euclidean distances over all d dimensions. On Borehole four of
eight dimensions carry ~0.4% of the output variance between them, and this
project has established that unweighted distance there **reverses conclusions**.
So every value ever read from these two fields is uninterpretable, and h100's
result is untested rather than negative.

The accepted candidate SET is discarded after summarising, so nothing can be
recovered by post-processing. It is a re-run either way.

## The design decision, and why it is NOT "store a weighted number"

The obvious fix is to record a sensitivity-weighted distance. **Do not do that.**
Three estimators of Borehole's sensitivities disagreed materially today:

    dim   midpoint-freeze   Sobol   binned S1
      6        8.0%          5.70%     4.11%
      7        0.1%          1.17%     0.94%

A single weighted number bakes ONE contested weighting into the log permanently,
and a future correction to the weights would require yet another re-run — which
is exactly the trap we are in now.

**Store per-dimension aggregates instead. Any weighting can then be applied
post-hoc, including weights nobody has computed yet.**

## What to record

In the `if roi_x_star is not None:` block, alongside the existing two fields,
with `_ad = ((roi_candidates - roi_x_star) / span).abs()`:

    perdim_mean_abs : [float] * d    _ad.mean(dim=0)
    perdim_min_abs  : [float] * d    _ad.min(dim=0).values

Cost: 2*d floats per ROI construction. On Borehole d=8 with ~6900 constructions
per run that is ~110k floats, so the worker must AVERAGE across constructions
into `roi_summary` rather than store every record — same as it already does for
the scalar fields.

In the worker's `roi_summary`, add the two vectors as element-wise means.

## Keep the existing fields

`min_dist_to_xstar` and `frac_within_0.2` stay, unchanged, so old and new runs
remain comparable on the measure they share. They are simply no longer the
measure any conclusion rests on.

## What this makes answerable

h100's Q1, properly: is the accepted region closer to x* **in the dimensions
that carry the variance** on the benchmark where the ROI works than on the ones
where it does not? That is the second of the two candidate mechanisms, and it is
currently untested rather than refuted.
