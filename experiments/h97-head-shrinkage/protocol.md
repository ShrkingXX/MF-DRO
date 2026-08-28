# H97 — is boundary aversion a PARAMETERISATION failure or a SIGNAL-STRENGTH failure?

ZERO NEW COMPUTE. Reads h90's completed Borehole runs. LOCKED BEFORE COMPUTING.

## Why this distinction decides the next experiment

h96 bounded the ROI's gain: it lifts boundary reach in dim 0 (81.6% of the
output variance) from 0.701 to 0.747, and barely moves dims 3/5/6 (0.049, 0.003,
0.007 against MF-MES's 0.49, 0.34, 0.70). The residual gap to MF-MES is
boundary aversion, and an ROI cannot fix it because the ROI selects WHERE to
look, not what the head can emit.

"The head cannot reach the boundary" is a DESCRIPTION. Two mechanisms produce
it and they prescribe DIFFERENT fixes:

  **A. PARAMETERISATION.** `x = action_head(h).clamp(0,1)` plus an MSE loss
     shrinks predictions toward the conditional mean of the targets. A tight
     cloud near the middle cannot reach a bound in ANY dimension. If this is the
     cause, the aversion is roughly UNIFORM across dimensions and the fix is
     architectural -- change the output map so extremes are reachable.

  **B. SIGNAL STRENGTH.** The head moves its centre where the training signal is
     strong and defaults to the target mean where it is weak. If this is the
     cause, the aversion is CONCENTRATED in the low-variance dimensions and the
     fix is a LOSS change (weight L_loc by dimension sensitivity), not an
     architecture change.

These are distinguishable at zero compute, and the project has already spent a
lot of budget on the wrong lever before.

## Measures (defined before computing)

Post-init HF query locations, normalized, h90 NO-ROI and ROI-Q10, seeds 47-51.
Per dimension:

  MEAN   mean emitted coordinate
  SD     std of emitted coordinate (uniform reference: sqrt(1/12) = 0.289)
  SHRINK SD / 0.289 -- 1.0 means as spread as uniform, <<1 means shrunk
  |MEAN - 0.5|   how far the cloud's centre sits from the domain centre
  x*     the target coordinate, for reference

## Predictions (registered before computing)

**S1. Shrinkage is severe in EVERY dimension.** h95 measured mean per-dim SD at
0.071-0.099 against a uniform 0.289, so ~3-4x. Registered POSITIVE and I expect
it trivially -- it is stated so the more interesting S2/S3 are read against it.

**S2 (THE DISCRIMINATOR). The head's centre |MEAN - 0.5| is LARGE in dim 0 and
SMALL in dims 3/5/6.** Registered POSITIVE, i.e. I expect mechanism B. Reasoning:
dim 0 carries 81.6% of the variance and the head already reaches its bound 70%
of the time, so it demonstrably CAN move its centre to a bound. A pure
parameterisation limit would prevent that everywhere.

**S3. Shrinkage (SD/0.289) is SIMILAR across dimensions.** Registered POSITIVE.
Together with S2 this is mechanism B: the head shrinks uniformly but RELOCATES
its centre only where the signal is strong.

**S4 (FALSIFIER FOR B). If |MEAN - 0.5| is small in dim 0 too, the head never
moves its centre anywhere, mechanism A holds, and the fix is architectural.**

## What this cannot settle

Correlational, n=5, one benchmark. It cannot prove that a sensitivity-weighted
L_loc would fix anything -- that needs the causal run. What it CAN do is tell us
which of two experiments to spend compute on, and the project's own history says
that choice has been made badly before.

## Label

EXPLORATORY. Diagnostic on existing data; no pre-existing bar.
