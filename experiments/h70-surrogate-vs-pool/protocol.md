# H70 — What is left of MI-Greedy's 5-point Borehole advantage?

**CONFIRMATORY.** Predictions locked before any h70 number exists.

## Why

The elimination chain for Borehole has one step left. h69 matched the acquisition
(SF-EI vs SF-MES, bit-for-bit surrogate) and moved regret by 0.29 of the 5.0
points separating SF-MES (13.28%) from MI-Greedy (8.3%). SF-EI sits at 12.99%,
still 4.7 points behind MI-Greedy **with the same acquisition class**.

Two implementation differences remain between SF-EI and MI-Greedy, both read
from source:

1. **Candidate pool size.** `MFMIGreedyOptimizer` draws `_CANDIDATE_POOL = 1000`
   candidates per HF selection. `GreedyMFBase` (SF-MES / SF-EI) draws
   `n_candidates = 200`. A **5x** difference in the greedy argmax pool.
2. **GP construction.** SF-EI builds through `_build_ko_style_gp` ->
   `KennedyOHaganGP._build_gp`, which applies a LogNormal lengthscale prior and
   a geometric-mean lengthscale initialisation. `mf_baselines._build_gp` applies
   neither. (The docstring claiming a shared "Interval lengthscale constraint" is
   **stale** — ko_gp replaced that constraint with a prior.)

The pool is the stronger suspect: h61 already measured that widening the pool on
Borehole buys **1.44x** acquisition value, and h64/h66's POOL600 arm was built on
that same observation.

## Design

Two arms, each exactly one change from h69's SF-EI:

- **POOL1000** — `n_candidates = 1000`, matching MI-Greedy. GP unchanged.
- **ALTGP** — GP built by `mf_baselines._build_gp`. Pool unchanged at 200.

Borehole 8D (primary) and Hartmann 6D (control), seeds 44/46/48, budget 200.
12 jobs, each a few seconds. SF-EI and MI-Greedy baselines reused, not rerun.

## Locked predictions

1. **PRIMARY.** POOL1000 closes at least half the remaining Borehole gap:
   SF-EI 12.99% -> **<= 10.6%**. Rationale is measured, not intuited — h61's
   1.44x acquisition-value gain on Borehole came from exactly this lever.
2. **SECONDARY.** ALTGP moves Borehole by **< 1 point**. The GP builder is a
   prior and an initialisation, not a different model class.
3. **NULL.** Neither arm moves Borehole by >= 1 point. Then MI-Greedy's advantage
   is not the pool and not the surrogate construction, the elimination chain has
   run out of identified differences, and what remains is its *loop* — the
   Explore-LF/EI alternation and its LF-informed `gp_error`, which are inert at
   100% HF and therefore should not matter, which would make the result genuinely
   puzzling rather than merely unresolved.
4. **CONTROL.** On Hartmann, POOL1000 should not *hurt*. A larger pool that
   degrades a benchmark would indicate the pool is trading exploration for
   exploitation rather than optimising the acquisition better.

## What this cannot settle

n=3. It explains a gap between two *greedy single-fidelity baselines*; it does
not by itself fix MF-DRO, whose pool is a separate knob already tested at
n=10 (h66, withdrawn). If POOL1000 succeeds, the honest reading is that a large
part of this project's Borehole story was **pool size in the baseline harness**,
not a property of DRO.
