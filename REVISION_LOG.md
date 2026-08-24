# Revision Log

## Known Limitation: Hartmann 6D Initialization

P(any of 18 LHS points within L2=0.3 of true optimum) ≈ 12% per seed.
Gate check (max > 2.0) failed for seed=42 (max=0.9632). This is expected
behavior under the curse of dimensionality, not a fixable initialization
problem. Results on Hartmann_6D will show high seed variance. N=5 seeds is
the minimum to characterize this variance. Report individual seed traces
alongside mean ± SE.

## Borehole 8D correlation note

Pearson r(HF, LF) = 1.0000. HF and LF are essentially proportional (differ
only by numerator 2π vs 5 and denominator constant 1 vs 1.5). MF-DRO's
learned policy on this benchmark tests whether DT can learn aggressive LF
exploration followed by targeted HF exploitation when LF is a near-perfect
proxy. Expected MF advantage here is high. Expected DT advantage over
greedy MES: unclear.
