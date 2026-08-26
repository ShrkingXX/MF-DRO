# H79b — Where do the two divergent seeds split?

**CONFIRMATORY.** Locked before any h79b number exists.

## Why

h79 found SF-EI@1000 and MI-Greedy identical bit-for-bit on 8 of 10 Borehole
seeds, diverging on 45 and 49. That weakened h70's claim from "MI-Greedy's
advantage is *entirely* pool size" to "pool size on 8 of 10".

**How the remaining claim should be stated depends on why they split.** If the
trajectories agree for many iterations and then separate, the two methods are
algorithmically equivalent and the divergence is an implementation artifact —
most plausibly RNG-stream drift, since MI-Greedy's `_explore_lf` samples
candidates and computes information gains *even when it selects no LF point*,
consuming draws SF-EI never makes. If they differ from the first iteration, the
reduction to single-fidelity EI is genuinely incomplete.

The proposed LF-activation explanation was already **refuted** by iteration
counts (both run 100 on both divergent seeds; seed 43 differs in count yet
matches exactly). This asks a narrower, directly measurable question.

## Design

Re-run MI-Greedy on Borehole seeds **45 and 49** recording the full regret curve
(h72's worker saved only `final_regret`). Compare against h79's SF-EI@1000 curves
iteration by iteration. Two matched seeds (44, 46) are re-run as controls and
must agree at every iteration.

MI-Greedy costs ~0.6 min/run. Negligible load beside h78.

**Built-in control is real:** re-running the same seeds with the same code must
reproduce h72's `final_regret` bit-for-bit.

## Locked predictions

1. **PRIMARY.** On both divergent seeds the curves agree for **>= 10 iterations**
   before separating. That is the implementation-artifact signature.
2. **SECONDARY.** On the two matched control seeds the curves agree at **every**
   iteration, not merely at the endpoint — confirming the endpoint match is not
   two different trajectories coincidentally landing together.
3. **NULL.** Divergence begins at iteration **< 10** on either seed. Then the
   methods differ early and the reduction to single-fidelity EI is genuinely
   incomplete, not an artifact — and "pool size on 8 of 10" is the strongest form
   the claim can take.

## What this cannot settle

It locates the split; it does not prove a cause. Confirming RNG-stream drift
specifically would require instrumenting the candidate draws, which this does
not do. A late split is *consistent with* an artifact, not proof of one.
