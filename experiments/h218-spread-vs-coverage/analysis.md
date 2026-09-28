# h218 analysis — CONFIRMATORY. 15/15 finals, 0 tracebacks. Both hypotheses eliminated.

Frozen metric imported from h83's `grid`; finals only; endpoint only. Borehole 42–46,
CTRL-K1 base.

## P-COV — the coverage hypothesis is REFUTED (by direction, not by the band)

| arm | regret |
|---|---|
| R-ROI (20 random) | 40.51 |
| R-ROI-120 (**6x** the rollouts) | **41.90** |

R-ROI-120 − R-ROI = **+1.38** (se 0.52, better on **0/5**).

Six times the random rollouts makes it **slightly worse**, never better on any seed. The
coverage account predicted a clear improvement; it is refuted by the **sign**. The
conditional-mean account predicted no change and got +1.38 — just outside the ±1.26 band,
so **strict P-COV fails on magnitude while its prediction is the one that survives
directionally.** Reported as such rather than claimed as a win.

## P-SPREAD — target spread is NOT the mechanism

| arm | action-target dispersion | regret | vs CTRL-K1 |
|---|---|---|---|
| CTRL-K1 | ~0.44 | 11.59 | — |
| JIT10 | 0.526 | 13.20 | +1.61 (se 1.07, 1/5) |
| JIT25 | **0.700** | 22.76 | **+11.17** (se 1.74, **0/5**) |
| MIXR-K1 (target) | — | 7.17 | −4.42 |

Both jitter levels are **worse**, monotonically in the dose. And JIT25's action-target
dispersion is **0.700 — exactly the all-random batch's** (Stage 0), so this is not
under-dosed: the spread was matched and it hurt.

**P-SPREAD FAILED.** Adding spread to the regression targets around the MES action is
actively harmful. MIXR's gain is not "a wider distribution of action targets".

## What this leaves

MIXR's mechanism, eliminated so far:

| candidate | ruled out by |
|---|---|
| data volume | h209 NIR120 (+1.42) |
| boundary geometry | h210 L1 + MIXR-P72 dims-on-face |
| fidelity composition | h210 MIXR-P72 keeps the gain mix-matched |
| RTG conditioning | audit (constant input) + the zero action-R² measurement |
| the K=8 window | h211 P-B (transfers to K=1) |
| **more random data / coverage** | **h214 dose + h218 R-ROI-120** |
| **action-target spread** | **h218 JIT10/JIT25** |

The JIT arms are the informative elimination: they keep the MES rollouts' **states** and
**outcomes** and change only the recorded **action**. That fails. So whatever the random
half contributes needs **new rollouts** — states MES never visits, and the outcomes
observed there — not a different action attached to the same rollout.

**Surviving candidate: state coverage.** The natural next arm is the DAgger-shaped one —
random rollouts for the **states** they visit, but relabelled with the action MES *would*
take at those states, so the DT learns good behaviour on a wider state distribution. That
separates "which states are in the training set" from "which actions are labelled there",
which nothing so far has done.
