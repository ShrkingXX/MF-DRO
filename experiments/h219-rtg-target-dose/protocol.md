# h219 — Dose the RTG target: does conditioning work, and where should we ask?

Registered BEFORE any h219 run. CONFIRMATORY. No new code: `rtg_target_schema='percentile'`
and `rtg_target_q` already exist (h212), default off, identity gate PASS.

## Why

`rtg-sensitivity-note.md` (same directory) establishes from saved probes and h212 that the
RTG channel is **not** inert — the target alone moves final regret by 3.52 and 5.27 in
one-factor tests. What is not known is the **shape** of that dependence. Two points exist
and they disagree in direction:

| target rule | Borehole regret | vs MIXR-K1 default |
|---|---|---|
| pinned at 0.5 (above all training data) | — | +5.27 worse (h212 P-FIX, Hartmann) |
| 90th percentile of the batch | 10.70 | **+3.52 worse** (h212 P-NEUTRAL) |
| `max(batch_max, alpha·running_max)` — the default | 7.17 | — |

Asking *above* everything is bad; asking *below* the best is also bad. A dose maps it.

## Arms — 15 runs, Borehole 42–46, CTRL-K1 base + MIXR-K1's random half (`mes_entropy`, K=1)

| arm | target rule |
|---|---|
| **Q50** | 50th percentile of this batch's `rtg[0]` — squarely in-distribution |
| **Q75** | 75th percentile |
| **Q100** | 100th percentile = the batch max, no alpha floor |

In hand at the same seeds: **Q90 = 10.70** (h212 MIXRK1-PCT) and the **floored default
= 7.17** (h211 MIXR-K1). Five points total: 50, 75, 90, 100, floored.

MIXR-K1 is the base rather than plain CTRL-K1 because both existing points sit on it, so
the five are directly comparable.

## Predictions, committed now (band ±1.26; endpoint only; no p-values at n=5)

- **P-MONO:** regret decreases monotonically in q — Q50 > Q75 > Q90 (10.70) > Q100 ≈
  floored (7.17). **Then RTG conditioning works as designed**: the "ask for the best
  rollout in the batch" heuristic is right, the channel is live, and the only bad regime is
  asking for something the data does not contain (the pinned 0.5).
- **P-FLAT:** all five within ±1.26 of each other. Then h212's P-NEUTRAL +3.52 was not the
  target level and needs another explanation, and the case for RTG mattering rests only on
  P-FIX.
- **P-PEAK:** an interior optimum — some q beats the floored default by > 1.26. Then the
  current "ask for the max" rule is **wrong** and there is a better place to ask, which is
  a deployable improvement, not just a diagnostic.
- **Lean: P-MONO**, from the two existing points. Registered so it can be wrong; my last
  two leans on this thread (label inert, h213; non-monotone dose, h214) were both wrong.

## Diagnostic — the fair state-vs-RTG comparison (not a gate)

The existing H168 probe compares states *within one iteration*, and h207's STATE-DIAG shows
those are nearly degenerate (`uniq_tau0_states=3` of 120). So d(x)/dSTATE is understated and
the "state vs RTG" ratio cannot be read from it. This experiment caches τ=0 states from
**earlier iterations** and probes across them, giving a state axis that spans a real range.
Reported alongside d(x)/dRTG so the user's "state dominance" hypothesis gets a fair test.

## What each outcome RETRACTS

- **P-FLAT** → retracts `rtg-sensitivity-note.md`'s reading that the target level is what
  h212's P-NEUTRAL measured.
- **P-PEAK** → retracts the target schema itself as currently written; "ask for the batch
  max" would be a tuned-by-accident choice with a better setting available.
- **P-MONO** → retracts nothing, and establishes the channel is live and correctly
  configured, which closes the RTG-insensitivity question in the negative.

## Compute

15 runs, cap 15, queued behind h214/h217/h218. 1 thread/worker.

## Evaluation

Frozen: final simple regret, rel% of |optimum| @ cost 200, imported from h83's `grid`.
Finals only. Every run reported.
