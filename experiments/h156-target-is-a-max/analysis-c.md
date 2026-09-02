# h156c -- Hartmann generality. Gate: **R2, PARTIAL**, as pre-named.

9 Hartmann states, N=100 per condition, harness unchanged.

| condition | MAX | real arm | observed | err |
|---|---|---|---|---|
| C1 closed-loop | 0.7860 | control | 0.8844 | −11.1% |
| C2 own path frozen | 0.7148 | (no arm) | — | forecast |
| C3 random path | 0.2789 | RANDOM-POOL | 0.2924 | **−4.6%** |
| C4 oracle → x* | 0.2490 | ORACLE | 0.3622 | **−31.2%** |
| C5 diverse-good | 0.2679 | (no arm) | — | forecast |

## What holds

The **scale** separation replicates cleanly on a second benchmark: control
0.79-0.91, every failing arm 0.25-0.30, on both. And the quantity the h153
forecast rests on replicates **exactly**: C2 retains **90.9%** of C1's MAX on
Hartmann, against 90.9% on Borehole. Two benchmarks, same number.

## What does NOT hold -- R2 fires

The harness was required to track the NARROWING between benchmarks, not just
reproduce two extremes. It does not:

|  | Borehole | Hartmann | change |
|---|---|---|---|
| ORACLE observed | 0.3113 | 0.3622 | **+0.0509** |
| C4 harness | 0.3022 | 0.2490 | **−0.0532** |

**The harness gets the cross-benchmark direction WRONG for ORACLE.** Reported as
partial: it reproduces scale, not fine structure.

## The misfit has a shape, and it points at my own instrument

| condition | Borehole err | Hartmann err |
|---|---|---|
| C3 random path | −7.6% | −4.6% |
| C4 oracle (interpolating) | −2.9% | **−31.2%** |
| C5 diverse-good (interpolating) | **−19.3%** | — |

The random-path condition fits well on both. The two INTERPOLATING conditions
fit worst and inconsistently. That is a suspicious pattern, and the likely cause
is a shortcut in my harness, not a fact about the arms: C4/C5 assign fidelity by
a hardcoded `1 if rand() < 0.75 else 0` (documented in run.py) instead of the
cost-normalised info-gain criterion the real arms use at the forced point. The
random condition is insensitive to this because its locations carry no
information either way; an interpolating path walking into a high-value region
is exactly where the HF/LF choice should matter most.

Registered as h156d, which tests that explanation directly.

## Effect on the h153 forecast

Unchanged and slightly strengthened: C2/C1 = 90.9% on both benchmarks. The
Hartmann misfit is in ORACLE, a condition the forecast does not depend on.
