# h212 — Was `terminal_improvement`'s Hartmann penalty the label, or the target schema?

Registered BEFORE any h212 run. CONFIRMATORY. One new config flag
(`rtg_target_schema`, default `'floored'` = bit-identical; identity gate PASS at
122.29066752728207).

## The defect

`update_and_get_rtg_target` returns `max(batch_max rtg[0], alpha · running_max rtg[0])`,
alpha = 0.5, `running_max` initialised 0.0. That is a **floor** only while RTG is
non-negative — which is `Old_dro.py`'s regime, where the per-step reward is clamped at
`max(0, y − best)` (L816–829). `terminal_improvement` drops the clamp, so once the real
incumbent outgrows what an 8-step fantasy can reach, every `batch_max` is negative and
`alpha · (positive running_max)` becomes a permanent **ceiling above everything in the
batch**. Measured (`experiments/h211-mixr-on-k1/loop-audit.md`):

| run | label | distinct targets | range |
|---|---|---|---|
| MIXR-B | `terminal_improvement` | **9 over 117 iters** | 0.500–1.000 |
| CTRL-K1-H | `mes_entropy` | 82 over 116 | 0.299–0.859 |

And h211 measured the cost: `terminal_improvement` **alone at K=1** is +5.80 on Hartmann
(0/5) — worse than the same label *with* the K=8 window (NIR-H 9.75 recovers 1.98).

**The confound this experiment resolves:** every result using `terminal_improvement` —
h207 NIR/MIXR/MIXO, h209, h210 — conditioned inference on a value outside its own training
distribution. We do not currently know whether the label is bad or the target is.

## The fix under test

`rtg_target_schema='percentile'`: the target is the **q-th percentile (q=90) of this
batch's own `rtg[0]`**. Scale- and sign-agnostic, inside the training distribution by
construction, and it tracks the batch as the run progresses. Keeps the intent of the
original ("ask for a near-best rollout") without the sign assumption.

## Arms — 15 runs, one wave

| arm | bench | seeds | config |
|---|---|---|---|
| **TI-PCT-H** | Hartmann | 42–46 | K=1, `terminal_improvement`, `percentile` |
| **TI-PCT-B** | Borehole | 42–46 | K=1, `terminal_improvement`, `percentile` |
| **MIXRK1-PCT-B** | Borehole | 42–46 | MIXR-K1 + `percentile` (label stays `mes_entropy`) |

Controls in hand (same seeds): CTRL-K1 **5.93** H / **11.59** B; K1-TI-H **11.73**
(h211); MIXR-K1 **11.19** H / **7.17** B; NIR-H 9.75.

The third arm is the negative control: `mes_entropy`'s targets already vary (82–110
distinct), so the schema should barely move it. If it moves a lot, the schema change does
something beyond fixing the pinning and every reading below is confounded.

## Predictions, committed now (band ±1.26; endpoint only; no p-values at n=5)

- **P-FIX:** TI-PCT-H − K1-TI-H < −1.26 on ≥ 4/5, i.e. the +5.80 penalty shrinks
  substantially. Lean: TI-PCT-H ≈ 7, most of the gap closed. **If it lands within the band
  of K1-TI-H (11.73), the label itself is harmful on Hartmann and the pinned target was
  incidental** — which retracts the loop audit's central inference and leaves h207–h210's
  Hartmann readings intact as label effects.
- **P-NEUTRAL:** |MIXRK1-PCT-B − MIXR-K1-B (7.17)| ≤ 1.26. The schema is inert where the
  target already varied. Failure means the schema change is not a targeted fix and P-FIX
  cannot be attributed.
- **P-B-TI:** |TI-PCT-B − NIR-B-equivalent| — reported, not scored: Borehole is where the
  label was already neutral (h207 P-NIR +1.23), so this arm measures whether the fix costs
  anything where nothing was broken. Registered expectation: inside the band of CTRL-K1-B.
- **Ordering, so it can be wrong:** TI-PCT-H < K1-TI-H, and MIXRK1-PCT-B ≈ MIXR-K1-B.

## What each outcome RETRACTS

- **P-FIX supported** → `terminal_improvement` is not harmful on Hartmann; the **target
  schema** was, and h207–h210's Hartmann numbers under that label were measuring a
  conditioning defect. h211's "the Hartmann gap is the LABEL" narrows to "the gap is the
  label's *interaction with the schema*".
- **P-FIX failed** → retracts the loop audit's inference that the pinned target caused the
  penalty; the label is simply worse on Hartmann and the pinning was a coincidence.
- **P-NEUTRAL failed** → the schema is a global intervention, not a fix; both readings above
  are void and the arm set is re-designed.

## Stage 0 (smoke)

(a) each arm builds and runs for BUDGET=30; (b) `mf.schemas.rtg_target_schema == 'percentile'`
on the new arms and `'floored'` by default; (c) the recorded `rtg_target` array from a
percentile run has **> 20 distinct values** where the floored run had ≤ 9 — the pinning is
visibly gone. A silent default is the failure mode.

## Compute

15 runs, cap 15, one wave. Hartmann first. 1 thread/worker.

## Evaluation

Frozen: final simple regret, rel% of |optimum| @ cost 200, imported from h83's `grid`.
Finals only. Every run reported.
