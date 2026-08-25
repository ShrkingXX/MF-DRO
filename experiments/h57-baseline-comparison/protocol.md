# H57 — MF-DRO (regression head) vs three MFBO baselines, three benchmarks

**CONFIRMATORY. Protocol committed before any run.**

## Design

| | |
|---|---|
| benchmarks | Currin 2D (c_H=3, c_L=1), Hartmann 6D (c_H=8, c_L=1), Borehole 8D (c_H=2, c_L=1) |
| seeds | 44, 46, 48 |
| cost budget | **200 post-init**, identical for all three benchmarks |
| methods | MF-DRO (regression head), Takeno MF-MES, MF-MI-Greedy, MF-GP-UCB |

36 jobs = 4 methods x 3 seeds x 3 benchmarks.

### Initial design

Taken from `experiments/large-scale/code/config.py`, which sized them at ~10% of
total cost per Best Practices (Nat Comput Sci 2025):

| benchmark | n_HF | n_LF | init cost | share of total |
|---|---|---|---|---|
| Currin 2D | 5 | 15 | 30 | 13% |
| Hartmann 6D | 6 | 45 | 93 | 32% |
| Borehole 8D | 10 | 20 | 40 | 17% |

Hartmann's share is the highest and is a known risk: at `c_H=8` a 93-cost init
against a 200 budget leaves ~24 HF queries. h56 measured that the SAME init at a
**100** budget has zero resolving power (regret flat across all iterations, both
arms). At 200 it should resolve -- h45 resolved at this budget with an even
larger init (36/60) -- but if Hartmann comes back flat across methods, that is a
resolution failure and not a null result, and must be reported as such.

**All four methods draw the identical initial design** for a given (benchmark,
seed). Verified, not assumed: `mf_baselines._lhs_init_points` and
`DirectMFRegretOptimization._sample_initial_points` both route through
`src/utils/init_design.py:make_initial_design` with the shared (seed,
seed_offset=0 for HF / 1 for LF) convention.

### Which MF-MES

`src/baselines/mf_mes_takeno.py:run_mf_mes` -- the standalone implementation
validated V1-V6 in h48, which optimizes the acquisition with a 2048-point Sobol
pool plus L-BFGS-B refinement. NOT `GreedyMFMESOptimizer`, which argmaxes over
200 uniform random points; h47-variant-d measured that a 200-point pool finds an
acquisition value **4.3x worse** than a 4000-point one and is not saturated even
there, so the pooled version is a handicapped baseline rather than the published
method.

`ko_kwargs=dict(dkl_threshold=9999)` holds the surrogate identical to the MF-DRO
arm (h48's D1); without it the KO-GP switches on deep kernel learning at
n_hf>=30 and the acquisition comparison is confounded by the surrogate.

## Locked predictions

1. **PRIMARY**: MF-DRO's mean final HF simple regret vs each baseline, per
   benchmark, paired on the 3 shared seeds. Reported as direction and per-seed
   win counts.
2. **NO p-VALUES.** n=3 per cell. h17-vs-h31 needed 82 seeds for 80% power on an
   effect of this size, and h45's regression arm has s.e. 0.1483. Any p computed
   here would be uninterpretable.
3. **PRE-REGISTERED EXPECTATION**: MF-DRO does NOT beat Takeno MF-MES on
   Hartmann. h48 found the two indistinguishable at n=10 (p=0.625) with the
   surrogate matched, and h45's regression arm finished worst-on-mean of three
   arms (0.4987 vs 0.4007 scoring, 0.4781 teacher) over 10 seeds. Currin and
   Borehole are untested for MF-DRO and carry no prediction.

## What this cannot settle

n=3 sizes effects and fixes directions; it establishes nothing. Two of the three
benchmarks have never been run with MF-DRO in this repo, so a surprise on Currin
or Borehole is a lead to follow up, not a result.
