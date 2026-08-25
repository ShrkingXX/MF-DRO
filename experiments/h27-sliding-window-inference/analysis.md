# H27 — GATE FAILED (G2). Stopped as pre-registered; no frozen-eval run.

Sliding-window inference implemented (`inference_context_k`): the DT receives the
last K−1 real `(state, rtg, btg)` triples plus the current one, readout at the
final state token.

## Gate

**G1 PASS** — the context genuinely grows. Per-iteration, K=8 arm: ctx = 1, 2, 3,
4, 5, 6 as history accumulates (K=1 arm stays at 1 throughout).

**G2 FAIL** — the proposals are **bit-identical at every iteration**:

| iter | ctx (K=1 vs K=8) | max abs Δx | fidelity |
|---|---|---|---|
| 0 | 1 vs 1 | 0.000e+00 | 0 vs 0 |
| 1 | 1 vs 2 | **0.000e+00** | 1 vs 1 |
| 2 | 1 vs 3 | **0.000e+00** | 1 vs 1 |
| 3 | 1 vs 4 | **0.000e+00** | 1 vs 1 |
| 4 | 1 vs 5 | **0.000e+00** | 1 vs 1 |
| 5 | 1 vs 6 | **0.000e+00** | 1 vs 1 |

A 10-iteration run reproduced this: identical trace, identical final regret
(0.4061 both arms).

Per protocol, **H27 stops here.** Feeding the real trajectory does not change
what the policy proposes, so there is nothing for a regret comparison to
measure, and the ~2 hour frozen-evaluation run was not spent.

## An unresolved discrepancy, reported rather than smoothed over

A separate probe fed a **synthetic** history — states drawn from seven different
ensemble-model blocks with fabricated `rtg`/`btg` — and found the hidden state
moves a great deal:

    ||h(T=8) - h(T=1)|| = 7.102   (||h|| ~ 11.32, cosine 0.803)
    ||w(T=8) - w(T=1)|| = 0.317   -> 19.88% relative change in w

So the extra tokens are **not** ignored: the network processes them and the
coefficient vector moves by ~20%. Yet with the **real** history a run actually
produces, the decision never changes, to machine precision.

## LOOSE END CLOSED — explanation (a) is refuted, (b) is supported

Measured directly on a real 10-iteration run's own history:

| | real history | synthetic probe |
|---|---|---|
| mean pairwise L2 among history states | **3.5703** | — |
| `‖h(T=8) − h(T=1)‖` | 5.2035 (cosine **0.8947**, ~26.5°) | 7.102 (cosine 0.803) |
| **`‖Δw‖ / ‖w‖`** | **11.20%** | 19.88% |
| **argmax moved** | **0/12** | — |

**Explanation (a) is wrong.** Real history is not impoverished relative to the
synthetic one: it moves `w` by **11.2%**, the same order as the synthetic 19.9%.

**Explanation (b) is supported.** An 11.2% perturbation of `w` — roughly 85x the
0.13% that ordinary state variation produces (H23) — still reorders nothing. The
induced shift lies in directions that do not change the candidate ranking. This
is the same structure H22 found by a different route: amplifying inputs 100x
grows the coefficient spread 76x and still moves the argmax 0/12.

So the tokens are processed, `w` genuinely moves, and the decision is invariant
anyway. The loose end resolves *into* the main account rather than complicating
it.

## Status

This is the third independent input channel shown not to reach the decision,
after state (H5 audit, H21, H22) and RTG/BTG (H8, H26). It is consistent with
the fixed-acquisition-rule account (H23): `w̄` alone reproduces the argmax 12/12.
