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

We do not currently have a measurement reconciling these. Two candidate
explanations — (a) the real history carries far less variation than the
synthetic one, so the induced Δw is much smaller than 20%; (b) the induced shift
lies close to a rescaling of `w`, which preserves an argmax — are **untested**.
The run-level result stands on its own and is what answers the question asked;
the synthetic probe is flagged as an open loose end rather than folded into the
conclusion.

## Status

This is the third independent input channel shown not to reach the decision,
after state (H5 audit, H21, H22) and RTG/BTG (H8, H26). It is consistent with
the fixed-acquisition-rule account (H23): `w̄` alone reproduces the argmax 12/12.
