# H88 — is MF-DRO's model limited by its DATA or by the surrogate itself?

LOCKED BEFORE ANY RUN. Cheap: a GP fit and an argmax per condition, no
optimisation loop, no new function evaluations beyond scoring recommendations.

## The question this breaks open

h83's traces showed inference regret equals simple regret exactly (ratio 1.00 on
all 20 runs): MF-DRO's surrogate never recommends a better point than its own
best query. That refutes "good model, bad policy", but it is CIRCULAR as
evidence -- the model is fit on the queries, so bad queries and a bad model
cannot be separated by observing completed runs.

The intervention: hold the surrogate class and the recommender FIXED, and swap
the DATA it is fit on.

## Design

For each benchmark in {Hartmann_6D, Borehole_8D} and each seed 42-46, fit the
SAME KennedyOHaganGP with the SAME hyperparameter procedure on:

  A. MF-DRO's own query set from h83 (its actual (x, y, fidelity) history)
  B. MF-MES's query set from h83, same benchmark, same seed -- a design of
     comparable cost produced by a method that scores 0.747 mean query quality
     on Hartmann against MF-DRO's 0.336

Then recommend `x_hat = argmax mu_H` over a dense Sobol pool (4096 points, far
denser than the 512-point y_star_pool the live metric uses, so the recommender
is NOT the bottleneck), and score f_HF(x_hat).

Both conditions use identical model code, identical pool, identical recommender.
The ONLY difference is which observations the GP saw.

## Metric

Recommendation regret = f(x*) - f(x_hat), reported as relative regret. Also
reported: each condition's own best observed HF value, so it is visible whether
a better recommendation merely reflects better data rather than better
generalisation.

## Predictions (pre-registered)

- **P1.** Fitting on MF-MES's data yields a BETTER recommendation than fitting
  on MF-DRO's, on >= 4/5 seeds, on both benchmarks. If met, MF-DRO's model is
  limited by the data its policy collects, and improving the policy is the
  lever.
- **P2 (the sharper one).** The recommendation from MF-DRO's own data is NO
  BETTER than MF-DRO's own best query -- i.e. the ratio-1.00 result survives a
  4096-point recommender. If this FAILS, the ratio-1.00 finding was an artifact
  of the weak 512-point single-member recommender and must be corrected.
- **P3 (NEGATIVE).** Fitting on MF-MES's data does NOT let the model recommend
  a point better than MF-MES's own best query. Registered negative because
  nothing in the setup gives the GP information beyond its observations; if it
  IS beaten, the surrogate generalises better than either policy exploits, which
  would be a genuinely surprising and useful result.

No p-values at n=5. EXPLORATORY throughout -- this is a diagnostic on existing
data, not a method proposal.

## What it cannot settle

Both designs were produced by optimisers, so neither is a neutral probe of the
surrogate class. A conclusive test would fit on a space-filling design of
matched cost; that is added as condition C if A/B prove informative.
