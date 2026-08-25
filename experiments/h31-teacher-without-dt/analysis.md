# H31 — the transformer is NOT a net negative. H32's locked expectation confirmed.

10/10 runs, **8900.1 s wall** (148 min).

## Per-seed final simple regret

| seed | teacher-only | MF-DRO | diff | t:n_HF | m:n_HF | t:iters | m:iters |
|---|---|---|---|---|---|---|---|
| 42 | 0.2445 | 0.4491 | −0.2047 | 10 | 17 | 130 | 81 |
| 43 | 0.4766 | 0.4483 | +0.0283 | 16 | 20 | 93 | 62 |
| 44 | 0.3713 | 0.2507 | +0.1206 | 17 | 23 | 81 | 45 |
| 45 | 0.5084 | 0.6018 | −0.0934 | 12 | 17 | 117 | 81 |
| 46 | 0.7511 | 0.5673 | +0.1838 | 13 | 18 | 109 | 76 |
| 47 | 0.4600 | 0.4307 | +0.0293 | 10 | 14 | 136 | 103 |
| 48 | 0.5746 | 0.2141 | +0.3605 | 13 | 20 | 109 | 67 |
| 49 | 0.4356 | 0.1996 | +0.2360 | 7 | 21 | 151 | 58 |
| 50 | 0.5068 | 0.5459 | −0.0391 | 10 | 18 | 130 | 80 |
| 51 | 0.4520 | 0.2992 | +0.1528 | 20 | 21 | 64 | 54 |

| method | final simple regret | sd |
|---|---|---|
| MF-MES teacher, **no DT** | 0.4781 ± 0.0414 | 0.1310 |
| MF-DRO / joint MES | **0.4007 ± 0.0475** | 0.1501 |
| MF-MI-Greedy | 0.5091 ± 0.1266 | 0.4004 |

Paired teacher − MF-DRO: **+0.0774**, teacher better on only **3/10** seeds,
Wilcoxon **p = 0.2324**.

## H32's pre-landing expectation is CONFIRMED

H32 predicted near-parity, reasoning from the faithful location distillation
(teacher-rank of the student's pick: median 2 of 200). That is what happened
(p = 0.2324).

**The "transformer is a net negative" hypothesis is refuted.** If anything MF-DRO
is slightly ahead (better on 7/10 seeds), but not significantly. The apparatus
does not cost performance; it also does not buy any.

## The fidelity divergence is confirmed and quantified

H33 predicted the two policies would differ on fidelity. Measured over full runs:

| | mean n_HF | mean iterations | HF share |
|---|---|---|---|
| teacher | 12.8 | 112.0 | **11.4%** |
| MF-DRO | 18.9 | 70.7 | **26.7%** |

MF-DRO queries high fidelity **2.3x more often** and therefore gets **37% fewer
queries** for the same budget (70.7 vs 112.0).

**This forces a correction to how H33 framed the miscalibration.** H33 called the
head's level a defect. On this benchmark the "defect" points in the direction
that is, if anything, mildly *helpful* — MF-DRO's higher HF rate coincides with
its slightly lower regret. What survives from H33 is the **uninformativeness**
(`p` sd = 2.4e-4, `corr` = 0.155 with the teacher's choice): the head does not
respond to state. Its *level* being wrong relative to the teacher is not
demonstrably harmful, and we should not have implied otherwise.

## The most consequential number here

**The teacher alone also fails the frozen success test**: mean+SE = 0.5195
against the bar of 0.3825.

So it is not the transformer that fails to clear the bar — the **entire
MF-MES-based approach** fails to clear it on this benchmark, distilled or not.
That relocates the negative result one level up: the ceiling is set by the
acquisition, not by the machinery wrapped around it.

## ETA record

Estimated 15--40 min; actual **148 min**. Wrong by ~4-10x. Cause identified
mid-run: the binding constraint is the *LF-heavy* seed, and cost-budgeted runs
have wall-times set by the fidelity mix, which varies by seed. Third miss on this
class of estimate.
