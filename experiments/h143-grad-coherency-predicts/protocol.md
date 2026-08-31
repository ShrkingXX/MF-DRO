# h143 — Does gradient coherence predict regret? The last support for the direction.

STATUS: LOCKED before computing. ZERO NEW COMPUTE.
TYPE: CONFIRMATORY. **Decisive for the training-signal direction.**
STATISTIC / INCLUSION: identical to h142 P4 — fractional within-run change in
`grad_coherency_per_iter` (first third -> last third) against final regret as
rel% @cost_curve 200 via the frozen `sr_curve`+`grid`; Spearman within benchmark;
the same four inclusion criteria, every qualifying run used.

## Why this is decisive

The training-signal direction — which I recommended to the user as where MF-DRO
should be improved — now rests on exactly one locked result: **h140 P1, gradient
coherence degrading more on Hartmann, effect 1.13 at n=5.**

h142 just showed what n=5 is worth here. Its rho of -0.600 on both benchmarks
collapsed to -0.188 and -0.418 once every qualifying run was included, and the
registered retraction fired.

**h140 P1 is in the same n=5 regime.** If gradient coherence also fails to predict
regret at power, then **no measured training-signal diagnostic predicts outcome**,
and the direction has nothing behind it.

## Prediction (locked)

**P1.** Within each benchmark, across all qualifying runs: |rho| >= 0.5 with
**worse coherence degradation going with worse regret**. FALSIFIED if |rho| < 0.5
on both benchmarks.

**I expect this to FAIL.** h142's collapse and the shared n=5 provenance make it
the likely outcome, and saying so in advance is the point — if it fails I cannot
present it as having been anticipated after the fact, and if it passes that is
genuine evidence against my own expectation.

No cross-benchmark pooling. P3's sign flip stands as the reason.

## What failure retracts, in full

**My recommendation to the user that "the training signal on low-HF-budget
benchmarks" is the direction for improving MF-DRO.** Not softened — withdrawn.
What would survive is only the *description*: on Hartmann the DT's `L_loc` rises
and its gradient coherence degrades. **Neither would be shown to matter for the
outcome**, and recommending work on a quantity not known to be a lever is exactly
the h117 error the whole project has been trying to stop repeating.

If it fails I will say plainly that the direction I recommended is unsupported,
and that the honest position is that we do not yet know why MF-DRO loses on
low-HF-budget benchmarks.
