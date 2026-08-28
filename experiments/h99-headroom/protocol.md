# H99 — is HEADROOM what gates whether the ROI can help?

ZERO NEW COMPUTE. Reads completed runs on all four benchmarks.
LOCKED BEFORE COMPUTING.

## Why, and a prescription of mine that this invalidates

h97 concluded: sharpen the head's centring in the SENSITIVE dimensions via a
sensitivity-weighted L_loc. **That prescription no longer follows from the
corrected analysis and I am withdrawing it pending this test.**

The correction (findings.md) established that simple regret depends on the
INCUMBENT, and that on Borehole the incumbent already sits at 1.000 in dim 0 --
the dimension carrying 82-84% of the output variance -- in BOTH arms. A
sensitivity-weighted loss would put ~82% of its weight on a dimension with
**zero headroom**. It would optimise hardest where nothing can be gained.

The quantity that should govern effort is not sensitivity, and not headroom, but
their PRODUCT: how much output variance is controlled by a dimension the method
has NOT yet solved.

## Measure

For each benchmark, from the no-ROI/baseline MF-DRO arm, over its seeds:

  HEADROOM_d  = mean over seeds of |incumbent_d - x*_d| in normalized coords
  WEIGHT_d    = first-order sensitivity share (binned S1, 40000 samples --
                NOT the midpoint-freeze shares, whose failure mode is recorded)
  PRODUCT_d   = WEIGHT_d * HEADROOM_d
  TOTAL       = sum_d PRODUCT_d   -- "how much of the objective is still on the
                table in dimensions the method has not solved"

Sensitivities are recomputed per benchmark by the binned estimator. Hartmann,
Ackley and Currin have never had one computed by that method.

## Predictions (registered before computing)

**P1. Currin's TOTAL is near zero.** Registered POSITIVE and near-certain --
both methods finish within 0.06% of its optimum. It is a sanity check on the
measure, not a finding.

**P2 (PRIMARY, RISKY). TOTAL predicts whether the ROI helped.** Ordering should
be Borehole highest, then the benchmarks where the ROI did nothing.

**I do NOT expect this to hold cleanly and am registering it anyway.** Hartmann
is where MF-DRO loses by the widest margin, so it should have LARGE headroom --
yet the ROI failed there (h87, 2/5, withdrawn). If Hartmann's TOTAL is high, P2
is refuted and headroom is necessary but not sufficient.

**P3. On Borehole specifically, dim 0 contributes near-zero PRODUCT despite
carrying 82-84% of the variance.** Registered POSITIVE. This is the concrete
claim that kills the sensitivity-weighted-loss prescription, stated so it is
tested rather than assumed.

**P4 (FALSIFIER FOR THE WHOLE FRAME). If TOTAL is uncorrelated with where the
ROI helped, then "the ROI helps where there is weighted headroom" is wrong**, and
the four-benchmark relocation table has no explanation beneath it -- it would
remain a true pattern with no mechanism. That must be said rather than smoothed.

## Limits, stated in advance

Four benchmarks. No p-values, no correlation coefficient over four points.
Different benchmarks have different dimensionalities, cost ratios and budgets,
so TOTAL is not comparable across them in absolute terms -- only its ORDERING
is being used, and even that is weak at n=4. First-order indices ignore
interactions; Borehole's first-order sum is 96.4%, so ~3.6% is unattributed.

## Label

EXPLORATORY. Diagnostic on existing data.
