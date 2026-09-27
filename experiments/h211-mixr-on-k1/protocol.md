# h211 — Does the random-rollout effect survive off the K=8 base, and what is it worth against the real reference?

Registered BEFORE any h211 run. CONFIRMATORY. Core code unchanged since h207 (49fbcd1);
every arm is a config of existing gated flags. Stage 0 is a smoke, not a gate re-run.

## Why

h210 settled two things that together make this the highest-value untested cell.

1. **CTRL-K1-H reproduces the last full run bit-identically** (5.93; paired +0.00, se 0.00,
   5/5). No code drift. The K=8 window has never beaten K=1 on any benchmark — it costs
   ~1 point on Borehole (12.62 vs 11.59) and ~3.8 on Hartmann (9.75 vs 5.93).
2. **Both mechanism accounts of MIXR are retracted.** Not boundary geometry (L1-NIR-B's
   dims-on-face 0.97 ≈ MSE's 0.95, and it is *worse*), not fidelity composition (MIXR-P72
   keeps the gain at dims-on-face 0.92 with the HF mix matched), not data volume (h209
   NIR120). What remains is a data-composition effect of unknown mechanism.

**Consequence:** every MIXR number ever measured sits on a base that is itself worse than
plain K=1. If the effect is data composition, it should not depend on the window at all —
so it should transfer to the K=1 base, where it would be measured against the real
reference (CTRL-K1) instead of a handicapped one (NIR).

## Arms — 15 runs, one wave

| arm | bench | seeds | config | purpose |
|---|---|---|---|---|
| **MIXR-K1-B** | Borehole | 42–46 | CTRL-K1 + 20 random rollouts/member | does the effect survive off K=8? |
| **MIXR-K1-H** | Hartmann | 42–46 | same | does its Hartmann *harm* survive, or was that the window? |
| **K1-TI-H** | Hartmann | 42–46 | CTRL-K1 with `terminal_improvement` | isolates label from window in the 3.82-pt gap |

MIXR-K1 = h84's ROI-Q10 base (K=1, positional embedding ON, `mes_entropy`) plus
`rollout_mix=[('mes',20,None),('random',20,None)]`, `random_p_hf=0.5`. **The label stays
`mes_entropy`** so MIXR-K1 differs from CTRL-K1 in exactly one thing: the random half.

Controls in hand (same seeds): CTRL-K1 Borehole **11.59**, Hartmann **5.93**;
NIR-B 12.62, NIR-H 9.75; MIXR-B 6.22, MIXR-H 12.08; MF-MES 6.40 / 6.62.

## Predictions, committed now (band ±1.26 rel%; endpoint only; no p-values at n=5)

- **P-B:** MIXR-K1-B − CTRL-K1 < −1.26 on ≥ 4/5. The effect is data composition and
  transfers off the window. Lean: ~7, i.e. most of the −5.4 seen against CTRL-K1 via the
  K=8 route. **If it does not transfer (|Δ| ≤ 1.26), the effect is an INTERACTION with the
  K=8 configuration** — which would be a genuinely new fact and would retract "it is a
  data-composition effect independent of the window", the premise of this protocol.
- **P-H:** MIXR-K1-H − CTRL-K1-H > +1.26. The Hartmann harm is the random half, not the
  window, so it persists on the better base. Lean: ~8. **If instead |Δ| ≤ 1.26, the
  Hartmann harm was the K=8 configuration interacting with the random half**, and a single
  configuration good on both benchmarks becomes reachable.
- **P-LABEL:** |K1-TI-H − CTRL-K1-H| ≤ 1.26. The label is neutral on Hartmann as it was on
  Borehole (h207 P-NIR, +1.23). Then the 3.82-point gap is attributable to the window
  alone. If the label is worse by > 1.26, part of that gap is the label and h210's
  "the gap is the K=8 configuration" needs splitting.
- **Ordering, so it can be wrong:** MIXR-K1-B < CTRL-K1-B and MIXR-K1-H > CTRL-K1-H —
  i.e. the effect keeps its sign on both benchmarks, independent of the window.

## The decision rule, registered before results

**A single configuration is kept only if it is ≤ CTRL-K1 on BOTH benchmarks** (within the
band on the one where CTRL-K1 is better, and beyond it on the other). Anything else is
reported as per-benchmark numbers with no method claim. Selecting a configuration per
benchmark is not a result.

## What each outcome RETRACTS

- **P-B fail** retracts this protocol's own premise (data composition independent of the
  window) and reframes MIXR as a K=8-specific interaction.
- **P-H fail** (harm vanishes on K=1) would make MIXR-K1 the first configuration to beat
  the last full run on Borehole without losing Hartmann — the only outcome that justifies
  the four-benchmark rerun, with Currin and Ackley as genuine held-out tests.
- **P-LABEL fail** splits h210's attribution of the Hartmann gap.

## Stage 0 (smoke)

(a) each arm builds and runs for BUDGET=30; (b) MIXR-K1 has `inference_context_k == 1`,
`disable_position_embedding == False`, `rollout_reward == 'mes_entropy'`, and a batch of
60 MES + 60 random; (c) K1-TI-H has `rollout_reward == 'terminal_improvement'` and
`inference_context_k == 1`. A silent default is the failure mode.

## Compute

15 runs, cap 15, one wave. Hartmann first (longer). 1 thread/worker.

## Evaluation

Frozen: final simple regret, rel% of |optimum| @ cost 200, imported from h83's `grid`.
Finals only. Every run reported.
