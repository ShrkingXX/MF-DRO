# H65 — does the teacher's local refinement transfer off Borehole?

**CONFIRMATORY. Protocol committed before any run.**

## Why this and not h64

h61 produced the project's first met prediction and its first durable gain:

| arm | mean | rel | wins | sd | spread |
|---|---|---|---|---|---|
| BASE | 73.40 | 23.7% | — | 4.73 | 8.62 |
| POOL600 | 60.23 | 19.5% | 3/3 | 13.34 | 25.33 |
| **REFINE** | 59.75 | 19.3% | 3/3 | **1.41** | **2.57** |

The two arms have near-identical means; **REFINE's distinguishing property is a
10x tighter spread**, and variance is the one failure signature that has
persisted through this entire project.

**h64 is testing the wrong arm for this question.** It generalises POOL600 (pool
size 200 -> 600) to Hartmann and Currin, and its pre-registered prediction is
NULL, on the measured grounds that widening the pool buys Hartmann **1.00x**
additional acquisition value. That prediction concerns pool coverage.

REFINE is not a pool-size change. h61's liveness check found it **widening** the
query spread (0.2907 vs BASE's 0.2534) while matching POOL600's mean, so its
channel is demonstrably not coverage. **h64's null prediction therefore says
nothing about REFINE**, and REFINE outside Borehole is untested and unpredicted.

## Design

Hartmann 6D, seeds 44/46/48, cost budget 200, `teacher_refine_samples=100`,
`teacher_refine_noise=0.05`. Everything else identical to h57's MF-DRO. **BASE
reuses h57's cells** (policy code verified byte-identical at the h61 regression
gate; commit hash recorded per result). 3 jobs.

Currin is deliberately excluded: it is saturated (every non-degenerate method
inside 0.6%) and can show neither a mean nor a variance effect worth reading.

## Locked predictions

1. **PRIMARY (variance)**: REFINE's spread across the three Hartmann seeds is
   **smaller than BASE's**. BASE Hartmann is 0.7531 / 0.2875 / 0.4228 — spread
   **0.4656**, which is 95% of its own mean (0.4878). If refinement stabilises
   the method rather than the benchmark, that spread should contract.
2. **SECONDARY (mean)**: REFINE beats BASE on >= 2/3 paired seeds.
3. **BOREHOLE-SPECIFIC**: neither spread nor mean moves. Then h61's result is a
   property of Borehole — plausibly its near-deterministic LF/HF relation
   (corr 1.000, OLS residual sd 0.0001), where a local pass can reliably find
   the same optimum every time.
4. **HARMFUL**: live. h60 showed Hartmann's fidelity split is teacher-driven and
   unstable (81%/4%/28% across seeds) and h58 showed both extremes cost regret
   there. A sharper teacher could amplify that instability, which would show up
   as spread INCREASING — the opposite of prediction 1 and the outcome that
   would most cleanly refute the stabilisation story.

**Prediction 1 is the primary, not prediction 2.** h61's mean gain and variance
gain came apart (POOL600 matched the mean with 10x the spread), so mean is the
weaker signal and I am committing to variance as the test before seeing data.

## What this cannot settle

n = 3. A spread computed from three points is a crude statistic and cannot carry
a significance claim; it can only show a direction. Cost ~3.3x BASE per seed
(~4.5 h for the wave), the measured REFINE multiplier from h61.
