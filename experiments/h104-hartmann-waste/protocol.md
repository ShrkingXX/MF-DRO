# H104 — does the ROI reduce WASTE on Hartmann, where it does not reduce REGRET?

ZERO NEW COMPUTE. h84's Hartmann ROI-Q10 arm vs its no-ROI control, seeds 42-46.
LOCKED BEFORE COMPUTING.

## Why this gap matters and why it is the commission's own question

The brief that opened this investigation states its diagnosis in HARTMANN terms:

    mean HF query score 0.336 vs MF-MES's 0.747
    20.8% of MF-DRO's HF queries land WORSE than the initial design
    proposals 3x more dispersed

and asks for "an ROI strategy that stops MF-DRO wasting HF budget on low-value
regions."

**Every waste measurement in this project so far is on BOREHOLE** (h95: mean
query regret -4.15, 5/5; waste halved wherever waste existed). Borehole is also
the one benchmark where the ROI improves regret. So the two things have only
ever been observed together, and nothing distinguishes "the ROI reduces waste"
from "the ROI improves regret" as descriptions of what it does.

Hartmann separates them. The ROI does NOT improve regret there (h87, 2/5,
withdrawn). If it nonetheless reduces waste, then the commission's target
quantity and the evaluation metric come apart -- and the brief asks for
something that would not, on this evidence, produce a better method.

## Measures (h95's, unchanged, so the two are comparable)

Post-init HF queries only, per run, then paired across seeds 42-46.

  W  waste fraction: share whose y is worse than the best HF value in the
     initial design
  Q  mean query regret, 100*(opt - y)/opt
  D  dispersion: mean per-dimension std of query locations, normalized

Control: h84's ROI-OFF where present, else h83's MF-DRO at the same seed --
the same reuse h84 itself uses under a reproduction control that passes at
0.000e+00.

## Predictions (registered before computing)

**W1. The ROI reduces the waste fraction on Hartmann, on >= 3/5 seeds.**
Registered POSITIVE but weakly. The ROI is a plausibility filter on the teacher's
pool and should exclude visibly bad regions regardless of whether that converts
into a better incumbent.

**W2. Mean query regret falls.** Registered POSITIVE, same reasoning.

**W3 (THE DISCRIMINATOR). If W1 and/or W2 hold while the regret result on
Hartmann stays failed, then WASTE AND REGRET ARE SEPARABLE**, and the
commission's framing names a quantity that does not determine the outcome. That
is a finding about the brief, and it must be stated as one rather than buried.

**W4 (FALSIFIER). If waste does NOT fall on Hartmann, the ROI's waste reduction
is Borehole-specific too**, and joins relocation as a one-benchmark
phenomenon -- leaving the ROI with no benchmark-general effect of any kind.

## Limits

n=5, one benchmark, EXPLORATORY. The waste measure has a floor at zero and this
project has already been caught by a bar on a floored measure (h95's M1), so
the per-seed values will be printed and any seed sitting at the floor in both
arms will be named rather than silently counted as a non-improvement.
