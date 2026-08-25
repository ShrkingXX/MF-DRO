# H59 — where does SF-DRO actually stand? (never measured in this repo)

**CONFIRMATORY. Protocol committed before any run.**

## Why this is the next experiment

The north star is "a novel method based on **SF-DRO** that is at least as good as
the baselines". SF-DRO is `src/policy/dro.py:58 DirectRegretOptimization` — the
single-fidelity method of `papers/DRO.pdf`, the thing MF-DRO extends.

**It has never been run here.** `grep -rl DirectRegretOptimization experiments/`
and `grep -rl _build_dro_config experiments/` both return nothing across every
experiment directory in this project. Every result to date — h1 through h58 —
measures the multi-fidelity extension. The baseline the north star is defined
against is unmeasured.

That is the gap: you cannot claim a novel method improves on SF-DRO, or that it
beats baselines "at least as well as SF-DRO does", without knowing where SF-DRO
sits.

### Smoke test done first (exploratory, 4 iterations, Currin 2D seed 44)

SF-DRO runs end to end: final regret 0.6037 over 4 iterations. Three blockers
hit, all harness errors on my side rather than defects in the method:

1. `rtg_schema` must be one of fixed/dynamic/floored/quantile/joint/entropy_joint
   — I guessed "regret".
2. The entry point is `run_optimization()`, not `run()`; results come from
   `dro.iteration_log_history`, not a return value.
3. `checkpoint.setup_dirs(exp_name)` must be called first — it creates
   `results/{exp}/checkpoints|logs|plots`, and the run crashes on the first
   iteration's log write without it.

Recorded because the next person to run SF-DRO will hit all three.

## Design

| | |
|---|---|
| benchmarks | Currin 2D (c_H=3), Hartmann 6D (c_H=8), Borehole 8D (c_H=2) |
| seeds | 44, 46, 48 |
| budget | **cost 200 post-init**, matching h57 exactly |
| arms | **SF-DRO** (`DirectRegretOptimization`), **SF-MES** (`SFMESOptimizer`) |

18 jobs. Cost 200 buys 66 / 25 / 100 HF queries respectively.

Single-fidelity arms take **no LF initial design** — initialization spends only
`n_initial_hf * c_H`, with `n_initial_hf` matched to h57's (5 / 6 / 10). This is
SF-MES's documented convention (`uses_lf=False`) and is applied to SF-DRO too so
the two arms match each other. It does mean the SF arms start from a cheaper
init than h57's MF arms; the **post-init cost axis is identical**, which is the
axis every h57 number is already reported on.

`rtg_schema="fixed"`, `use_mes_reward=False`, `rollout_acq_function="ei"` — the
dro_runner defaults, i.e. the method as this repo ships it, not a tuned variant.

Every result records its commit hash and the full query trace, as in h57.

## Locked predictions

1. **PRIMARY**: SF-DRO's mean final simple regret vs SF-MES's, per benchmark,
   paired on 3 seeds. Direction and win counts only.
2. **PRE-REGISTERED EXPECTATION**: SF-DRO does **not** beat SF-MES on Hartmann
   or Borehole. Basis: MF-DRO lost 0-3 to MF-MES on both in h57, and h48 found
   the MF pair indistinguishable at n=10 with the surrogate matched. If the
   rollout-and-distil architecture were adding something, MF-DRO should already
   have shown it.
3. **THE OUTCOME THAT WOULD CHANGE THE PROJECT**: SF-DRO beats SF-MES where
   MF-DRO lost. That would locate the failure in the **multi-fidelity
   extension** — most plausibly the fidelity head, measured in h57 at 81%/4%/28%
   HF across three Hartmann seeds differing only by initial design — rather than
   in the DRO architecture. The north star would then be reachable by dropping
   multi-fidelity, not by fixing it.
4. **NULL**: SF-DRO also loses. Then the rollout-and-distil architecture itself
   is what does not beat a well-optimised acquisition, and a novel method needs
   to change that, not the fidelity handling.

## What this cannot settle

n=3 per cell, no p-values. Cross-comparison to h57's MF numbers shares only the
post-init cost axis, not the initial design, so SF-vs-MF statements are
directional at best and are not a locked prediction here.
