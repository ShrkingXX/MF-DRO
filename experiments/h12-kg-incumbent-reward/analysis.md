# H12 analysis — GATE FAILED (G2). Stopped as pre-registered.

| metric | `improvement` | `kg_incumbent` | gate |
|---|---|---|---|
| trajectories with `rtg[0]==0` | 63.0% | **4.5%** | G1 **PASS** (<20%) |
| LF steps with nonzero reward | **0.0%** | **27.2%** | G2 **FAIL** (needed >50%) |
| `rtg[0]` mean / CV | 0.1022 / 1.964 | 0.1396 / 1.374 | — |
| batch wall time (200 traj) | 31.9 s | 32.9 s | +3% cost |

**H12 stops here. No comparison was run and no regret claim is made** — the
protocol committed to that before the numbers existed.

## What the gate established anyway

G1 passed decisively: the dead-signal fraction collapses from **63.0% to 4.5%**.
That confirms the diagnosis behind H9/H10/H11 — the conditioning signal really
was dead on two thirds of the training data, and a reward that credits both
fidelities fixes that. Cost is negligible (+3% wall time).

G2's failure is the useful part: crediting LF at all took it from 0.0% to 27.2%,
but that is still far from dense.

## Why G2 failed — two clipping mechanisms, both mine

    V_tau = MAX over the reference set of mu_H      <- (i)
    r_tau = MAX(0, V_tau - V_{tau-1})                <- (ii)

- **(i) the hard max is insensitive to most observations.** `V` only moves when
  an observation changes `mu_H` *near the current argmax*. Exploratory LF queries
  — the majority — land elsewhere and move `V` not at all.
- **(ii) `max(0, .)` discards downward revisions.** Learning that a region is
  *worse* than believed is genuine progress toward the optimum, and it scores
  zero.

So `kg_incumbent` fixed the "LF earns exactly zero" defect but inherited a
sparsity of its own from the estimator of `V`. The reward is no longer *blind*
to LF; it is still *deaf to most of what LF says*.

## Honest note on an earlier claim

While wiring this up I stated that `y_star_pool` was not passed at the
`simulate_mf_trajectory` call site. It was — my grep window ended a few lines
short. Corrected immediately; no result depended on it.

## Where this goes

The fix targets the two clips directly and is locked separately as H13 rather
than patched into H12 after seeing its gate fail.
