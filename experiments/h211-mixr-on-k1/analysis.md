# h211 analysis — CONFIRMATORY. 15/15 finals, 0 tracebacks.

Frozen metric imported from h83's `grid`; finals only; endpoint only.

## Result

| arm | bench | regret | vs CTRL-K1 | LF |
|---|---|---|---|---|
| **MIXR-K1-B** | Borehole | **7.17** | **−4.42** (se 1.67, **5/5**) | 0.125 |
| MIXR-K1-H | Hartmann | 11.19 | +5.26 (se 3.09, 1/5) | 0.649 |
| K1-TI-H | Hartmann | 11.73 | **+5.80** (se 2.30, **0/5**) | 0.779 |
| CTRL-K1 (reference) | B / H | 11.59 / 5.93 | — | |
| MF-MES | B / H | 6.40 / 6.62 | | |

## Scoring

**P-B — SUPPORTED.** MIXR-K1-B = 7.17, better than CTRL-K1 on **5/5**. The random-rollout
effect is **not** a K=8 interaction: it transfers to the plain K=1 base, where it is
measured against the real reference rather than a handicapped one. This is the cleanest
form of the effect to date — ROI-Q10 plus 20 unselected random rollouts per member, no
window, no positional-embedding change, no label change.

**P-H — SUPPORTED.** MIXR-K1-H = 11.19, worse than CTRL-K1-H on 4/5 (+5.26, se 3.09;
per-seed +2.8, +0.5, +14.4, +10.5, −1.9). The Hartmann harm is the random half, not the
window: it persists on the better base. Spread is wide — seed 44 carries much of it — but
the sign is consistent.

**P-LABEL — FAILED, and it overturns h210's attribution.** `terminal_improvement` **alone,
at K=1**, costs **+5.80 on 0/5 seeds** on Hartmann. h210 concluded "the ~3.8-point Hartmann
gap is the K=8 configuration"; that is now **RETRACTED**. The gap splits, and not evenly:

| Hartmann arm | regret |
|---|---|
| CTRL-K1 (K=1, `mes_entropy`) | 5.93 |
| K1-TI (K=1, `terminal_improvement`) | 11.73 |
| NIR (K=8, `terminal_improvement`) | 9.75 |

The **label** costs +5.80; adding the window on top *recovers* 1.98. So the window is not
the Hartmann problem — the label is, and the window partially masks it.

This is exactly what the loop audit (`loop-audit.md`, same day) predicted from code and
measurement: under `terminal_improvement` the inference RTG target is **pinned at 0.5**,
above everything in the training distribution, because the `max(batch_max, alpha·running_max)`
schema is a floor designed for the non-negative RTG of `Old_dro.py`, and
`terminal_improvement` drops that clamp. On Borehole that miscalibration was neutral
(h207 P-NIR, +1.23, inside band); on Hartmann it costs 5.8 points. **A pinned conditioning
target is not harmless — it is benchmark-dependent.**

## The decision rule, applied

Registered: *keep a single configuration only if it is ≤ CTRL-K1 on BOTH benchmarks.*
MIXR-K1 is −4.42 on Borehole and +5.26 on Hartmann. **The rule is not met; no single
configuration is kept.** Reported as per-benchmark numbers, with no method claim.

## What stands

- **A deployable Borehole improvement, now on the clean base:** CTRL-K1 11.59 → MIXR-K1
  7.17 (5/5), close to MF-MES's 6.40, on the benchmark where MF-DRO previously lost. The
  effect has now held across 20/20 seeds and four configurations (MIXR 42–46, MIXR 47–51,
  MIXR-P72, MIXR-K1). Mechanism still unidentified: not volume (h209), not boundary
  geometry (h210), not fidelity composition (h210), not RTG conditioning (audit — the
  channel is a constant input), not the window (h211).
- **A second, separable finding:** `terminal_improvement` is harmful on Hartmann for a
  reason traced to the target schema, not to the reward definition. Every h207–h210 arm
  that used it on Hartmann was carrying that penalty.

## Next (proposed, NOT registered)

1. **Fix the target schema** for signed RTG (batch percentile, or per-batch min-max into
   [0,1]) and re-measure `terminal_improvement` on Hartmann. If the +5.80 disappears, the
   label is fine and the schema was the defect — and h207–h210's Hartmann numbers need
   re-reading.
2. **MIXR-K1 on Currin and Ackley** — genuinely held-out, since everything has been
   developed on Borehole and Hartmann. Decides whether Borehole is the exception or
   Hartmann is.
