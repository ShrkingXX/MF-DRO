# h217 analysis — CONFIRMATORY. 10/10 finals, 0 tracebacks. Both predictions FAILED.

Metric imported from h83's `grid`; finals only; endpoint only. Ackley reports **absolute**
regret (its known optimum is 0), per h83's convention and as flagged in the protocol.

## Result

| bench | MIXR-K1 | ROI-Q10 K=1 | difference |
|---|---|---|---|
| Currin_2D (rel%) | **0.0107** | 0.1255 | −0.1148 (se 0.1057, MIXR better 3/5) |
| Ackley_10D (abs) | **4.0013** | 3.7425 | +0.2589 (se 0.3722, MIXR better 2/5) |

**P-CURRIN — FAILED.** I predicted MIXR would be clearly *worse*. It is nominally better,
but the entire mean difference is one seed (43: −0.5365); the other four are
−0.0385, +0.0087, +0.0029, −0.0104. At 0.01–0.13% regret this is noise, exactly as the
protocol's near-saturation caveat anticipated. **Read as a null.**

**P-ACKLEY — FAILED.** I predicted worse on ≥4/5; it is worse on 3/5 with a difference
(+0.26) well inside its own standard error (0.37). **Read as a null.**

## What this does to the synthesis

The reading under test was *"MIXR repairs a failure mode only Borehole exhibits."* Half of
it survives and half is retracted:

- **Survives:** MIXR helps only on Borehole, the one benchmark where MF-DRO loses to the
  strongest baseline.
- **RETRACTED:** "it hurts everywhere else." It is **neutral** on Currin and Ackley.
  Hartmann is the sole benchmark where it harms, not the rule.

So MIXR is neutral-or-better on 3 of 4 benchmarks and harmful on 1 — a better standing than
I predicted.

## The pattern that emerges, ordered by cost ratio

| bench | HF:LF | MF-DRO vs best baseline | MIXR effect |
|---|---|---|---|
| Borehole | **2:1** | broken (+5.19) | **helps −4.42** |
| Currin | 3:1 | wins | neutral |
| Ackley | 5:1 | tie | neutral |
| Hartmann | **8:1** | wins | **hurts +5.26** |

**The effect is monotone in the cost ratio**: helps where HF is cheapest, neutral in the
middle, hurts where HF is most expensive. That is exactly what the fidelity-economics
account predicts — MIXR shifts real queries toward HF (LF −0.136 Borehole, −0.095
Hartmann), and the value of that shift is set by what HF costs.

EXPLORATORY: four points, one per benchmark, and the cost ratio is confounded with
dimension (2, 2, 10, 6) and with whether the baseline is broken. It is a pattern worth
acting on, not an established mechanism.

## What it points at

A `random_p_hf` matched to the MES half's own HF fraction, on **both** Borehole and
Hartmann. h210's MIXR-P72 already showed mix-matching **preserves** the Borehole gain
(−4.72, 5/5) on the K=8 base; it has never been run on Hartmann or on CTRL-K1. If the
monotone pattern is fidelity economics, matching should remove the Hartmann harm while
leaving Borehole intact — which would meet the registered both-benchmarks rule.
