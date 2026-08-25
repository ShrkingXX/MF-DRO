# H62 — re-run SF-DRO with working traces (and verify h59 reproduces)

**CONFIRMATORY. Protocol committed before any run.**

## Why

h59's SF-DRO cells were written with a silent instrumentation bug: the worker
looked for per-iteration `x`/`y` in `dro.iteration_log_history`, which carries
`regret`, `best`, `mean_reward`, `zero_frac`, `rtg_target`, `batch_max_rtg`,
`running_max_rtg` and **no coordinates**. The filter `if "x" in d` matched
nothing, so every SF-DRO result has `n_queries: 0` and an empty `query_x`
alongside a full `regret_curve`. Fixed afterwards (queries live on
`dro.data_x` / `dro.data_y`), but the existing results cannot be re-derived.

Consequence: **SF-DRO is the only method in this project with no query trace.**
It cannot be run through `freeze_watch.py`, `stall_diagnose.py`, or any
value-percentile analysis — and it is the method the north star is defined
against. Every diagnostic conclusion to date is about MF-DRO.

The specific question this unblocks: **SF-DRO's Currin seed 46 finished at
0.1844 against a 0.0012 median** — a 50x mean-vs-median gap, the same
catastrophic-seed signature as h45's regression head. That seed is the single
clearest instance of the variance problem in a fidelity-free setting, and it
currently cannot be diagnosed at all.

## Design

Exact re-run of h59's SF-DRO arm: Currin 2D, Hartmann 6D, Borehole 8D, seeds
44/46/48, cost budget 200 post-init, `initial_hf` 5/6/10 with no LF init,
`rtg_schema="fixed"`, `use_mes_reward=False`, repo-default DRO settings. 9 jobs.

Only difference from h59: the trace is actually recorded.

## Locked predictions

1. **REPRODUCIBILITY GATE (primary).** Each cell's `final_regret` must match
   h59's to within 1e-9. The runs are seeded and the fix touched only
   post-`run_optimization` bookkeeping, so they must be bit-identical.
   - **Match** -> h59's regret conclusions stand and the traces are usable.
   - **Diverge** -> something in the trace fix perturbed the run, h59's SF-DRO
     numbers become suspect, and the divergence is reported rather than the
     new numbers being quietly adopted.
2. **No new regret claim is made here.** This experiment produces
   instrumentation, not evidence about SF-DRO's performance. Any performance
   statement continues to come from h59.

## What this cannot settle

Nothing about whether SF-DRO is good — that is h59's job and it is already
answered (Currin 0.4%, Hartmann 11.5%, Borehole 15.1%; beats SF-MES on Hartmann
only). This is purely to make the diagnostics that exist for MF-DRO available
for SF-DRO.
