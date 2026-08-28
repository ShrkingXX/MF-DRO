# H100 — does the ROI region CONTAIN the optimum where it helps, and miss it where it doesn't?

ZERO NEW COMPUTE. The ROI already logs its own containment diagnostics on all
four benchmarks. LOCKED BEFORE COMPUTING.

## Why

H99 eliminated headroom as the gate: the ROI works on the benchmark with the
second LEAST left to gain (Borehole 0.0185) and fails on the one with ten times
more (Hartmann 0.1852). The relocation pattern from h96 is real and unexplained.

The hypothesis, recorded in findings.md as speculative: **the ROI restricts to a
region the SURROGATE believes plausible. Where the surrogate is good, that region
contains x* and restriction helps. Where it is poor -- which is exactly where
headroom is large, because the method has not found x* -- the region may EXCLUDE
x*, and restricting to it is inert or harmful.**

This is checkable without running anything, because `roi_stats` already records,
per ROI construction:

  min_dist_to_xstar   the closest accepted candidate to x*, normalized
  frac_within_0.2     the fraction of accepted candidates within 0.2 of x*

Confirmed present on all four benchmarks' ROI arms. These were added precisely
because "an ROI that never contains anything near the optimum starves the DT of
near-optimal training examples" was the failure that got the ROI deleted before.

## Predictions (registered before computing)

**Q1 (PRIMARY). Borehole's ROI sits CLOSER to x* than Hartmann's**, on
min_dist_to_xstar and on frac_within_0.2. Registered POSITIVE.

**Q2. The ordering of frac_within_0.2 across the four benchmarks matches the
ordering of the ROI's regret benefit** (Borehole best). Registered POSITIVE but
WEAKLY -- H99's ordering test already failed once at n=4 benchmarks, and I have
no reason to think this one is better powered.

**Q3 (FALSIFIER). If Borehole's ROI is no closer to x* than the others, the
surrogate-quality hypothesis is dead** and the ROI's benchmark-specificity has
no surviving explanation. Two candidate mechanisms will then have been
eliminated by measurement (headroom, containment) and the honest position is
that h96's pattern is unexplained.

## The comparability problem, stated before computing

**min_dist_to_xstar is a normalized distance in d-dimensional space and d
differs across benchmarks** (2, 6, 8, 10). Distances are not comparable across
dimensionalities -- the closest of N uniform points to a fixed target grows with
d, which is exactly the artifact that produced the withdrawn "zero rollout steps
within L2=0.2" claim recorded in findings.md.

So the PRIMARY comparison is Borehole vs Hartmann only where a same-benchmark
reference exists, and every cross-benchmark number will be reported alongside
the expected distance for a uniform draw at that d, so the reader can see the
baseline. **A raw cross-benchmark ranking of min_dist would be meaningless and
will not be presented as the verdict.**

## Label

EXPLORATORY. Diagnostic on already-logged data.
