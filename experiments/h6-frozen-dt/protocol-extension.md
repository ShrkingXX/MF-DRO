# H6 extension — pre-registered BEFORE running, to fix an underpowered design

## Why an extension is needed, and why this is not p-hacking

H6 at n=10 established **nothing**:

| quantity | result |
|---|---|
| paired mean diff | +0.0978, 95% CI [-0.0966, +0.2923] (straddles zero) |
| Wilcoxon | p = 0.3223 |
| variance ratio | 4.79x, but Levene (robust) **p = 0.2090** |

The parametric variance tests (F p=0.029, Bartlett p=0.029) disagree with the
robust one. Regret here is right-skewed and bounded below, so normality is
doubtful and **Levene is the test to believe**. Both arms are also *paired*
(same seeds), which independent-sample variance tests do not respect.

A post-hoc power analysis on the observed paired sd (0.3138) shows the original
design could only resolve effects >= ~0.3 regret units, against arms that sit
near 0.5 total. The design was underpowered for its own question.

## Guard against optional stopping

Extending n after seeing a non-significant result inflates false positives if
one keeps extending until significance appears. To prevent that, this extension
**fixes the final sample size in advance**:

- **Final n = 30** (existing seeds 42-51, plus new seeds 52-71). Chosen now,
  before any new run, on power grounds alone.
- The **n=30 analysis is the primary and final one** for H6. I will not extend
  again on the basis of its outcome, and will not report an intermediate n.
- If n=30 is still inconclusive, the reported conclusion is
  *"inconclusive at n=30"* — not another extension.

## Power at the pre-specified n=30

    paired sd (observed)  = 0.3138
    SE at n=30            = 0.0573
    detectable at 80% pwr = ~0.29 regret units  (vs ~0.50 at n=10)

n=30 still cannot resolve the observed +0.098 mean difference (that needs ~80
seeds). **This is stated up front**: the extension is expected to sharpen the
variance comparison and tighten the CI, not to settle the mean. If the honest
outcome is "the mean difference remains unresolved", that is what gets reported.

## Locked predictions

1. **Primary**: the 95% CI on the paired mean difference at n=30 still contains
   zero. (i.e. I predict H6's mean question stays unresolved)
2. **Secondary**: Levene's test on FROZEN vs LIVE variance at n=30 gives
   p < 0.05, confirming freezing genuinely destabilises the method.

Prediction 1 is a prediction of a *null*, made in advance precisely so that a
null cannot later be spun as a finding.

## Scope note

This does NOT touch `PROTOCOL.md`. The frozen evaluation governs the MF-DRO vs
MF-MI-Greedy vs MF-GP-UCB comparison at 10 seeds, which is complete and reported
(FAIL). H6 is an internal MF-DRO-vs-MF-DRO method comparison, so changing its n
is a method-development decision, not a change to the frozen evaluation.

## Compute

20 new seeds, `num_workers=15 x threads_per_worker=1 = 15 <= 15`.
~50 min/run, 2 waves, ~100 min expected.
