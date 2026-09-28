# h217 — Is MIXR's gain Borehole-specific? Currin and Ackley as held-out benchmarks

Registered BEFORE any h217 run. CONFIRMATORY. No code change: the h211 MIXR-K1 worker,
run on two benchmarks it has never seen.

## Why these two, and what the existing numbers already say

| bench | d | HF:LF | ROI-Q10 K=1 | MF-MES | gap | MIXR-K1 |
|---|---|---|---|---|---|---|
| Currin_2D | 2 | 3:1 | 0.13 | 0.35 | −0.22 (MF-DRO wins) | **?** |
| Hartmann_6D | 6 | 8:1 | 5.93 | 6.62 | −0.68 (MF-DRO wins) | 11.19 (**hurts +5.26**) |
| **Borehole_8D** | 8 | **2:1** | 11.59 | 6.40 | **+5.19 (MF-DRO BROKEN)** | 7.17 (**helps −4.42**) |
| Ackley_10D | 10 | 5:1 | — | — | tie | **?** |

**Borehole is the only benchmark where MF-DRO loses to the strongest baseline**, and it is
the only one where MIXR helps. Three properties co-vary there and no experiment so far can
separate them: a broken baseline, a **corner optimum** (7 of 8 dims within 2% of a box
face, vs 0 of 6 on Hartmann), and the **cheapest HF** (2:1).

Currin breaks two of the three: `d`=2, 3:1 cost, and MF-DRO is already near-saturated
(0.13% regret). Ackley adds a third point at d=10, 5:1.

## Arm — 10 runs

MIXR-K1 (CTRL-K1 + 20 random rollouts/member, `mes_entropy`, K=1, ROI-Q10) on
**Currin_2D** and **Ackley_10D**, seeds 42–46. Controls already on disk at the same seeds:
h84/h86 ROI-Q10.

## Predictions, committed now (band ±1.26 rel%; endpoint only; no p-values at n=5)

- **P-CURRIN:** MIXR-K1 is **worse** than ROI-Q10 on Currin (> +1.26 in relative terms, or
  clearly worse on ≥4/5 given the tiny absolute scale). Rationale: MIXR has helped only
  where the baseline is broken, and Currin's baseline is near-optimal. **Near-saturation
  caveat, stated up front:** at 0.13% regret the headroom is ~nil, so a "no change" reading
  is weakly informative and only a clear *harm* is decisive.
- **P-ACKLEY:** MIXR-K1 worse than ROI-Q10 on ≥4/5.
- **The synthesis being tested:** *MIXR is a repair for a failure mode only Borehole
  exhibits, not a general improvement.* Confirmed if it hurts or is neutral on both.
  **Refuted if it helps on either** — which would make it a general method and would be the
  most important result of this thread.

## Metric note

Ackley's known optimum is 0, so rel% of |optimum| is undefined; h83's convention reports
**absolute** simple regret there. The readout follows h83 per benchmark rather than forcing
one scale.

## Compute

10 runs, cap 15, queued behind h214. Ackley first (d=10, longer). 1 thread/worker.

## Evaluation

Frozen: final simple regret at cost_curve 200, metric imported from h83's `sr_curve`/`grid`.
Finals only. Every run reported.
