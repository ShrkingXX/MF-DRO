# H40 — replicate H39 on Hartmann 6D with 10 seeds

## Why

H39 found the regression head does **not** re-freeze (14/14 distinct proposals,
2 incumbent improvements) but was **n = 1, Currin 2D, 14 iterations** — an easy
benchmark that reaches regret 0.005 by iteration 6. Hartmann 6D is where the
incumbent-freeze pathology was originally observed (9/12 runs pre-leak-fix), so
it is the benchmark that can actually falsify H39.

## Design

Two arms, one variable: `use_candidate_scoring` ∈ {True, False}.
Seeds 42–51 (10). Hartmann 6D. Cost budget 200 post-init.

Initial design uses the **literature-standard sizing** (~10% of total cost, split
50/50 by cost): `n_HF = 6`, `n_LF = 45`. This differs from h1's `36/60`, which
was 64% of total budget — so H40 is **not** directly comparable to h1's regret
numbers, only to H39's freeze question.

## Locked predictions

1. **PRIMARY**: the regression arm's mean incumbent-improvement count is **> 0**
   on at least 8/10 seeds — i.e. H39 replicates and the freeze does not return.
2. **FREEZE RATE**: report runs with **zero** improvements for both arms. The
   pre-leak-fix reference was 9/12 (75%) frozen.
3. **NULL / FALSIFICATION**: if the regression arm freezes on ≥ 3/10 seeds while
   candidate scoring does not, H39 was a Currin artefact and the
   candidate-scoring rewrite **is** load-bearing for the freeze fix — which
   would restore the h1 confound rather than resolve it.

## Not claimed

Regret comparison between the two heads is secondary and underpowered at n = 10;
the pre-registered question is the **freeze**, not which head optimises better.

## Compute

20 jobs, 15 workers × 1 thread. Hartmann measured at ~37 s/iter with this init;
budget 200 at ~2.75 cost/iter ≈ 73 iterations ≈ 45 min/seed → **~90 min** over
two waves.
