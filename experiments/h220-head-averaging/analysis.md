# h220 analysis — CONFIRMATORY. 15/15 finals, 0 tracebacks. P-HEAD FAILED.

Borehole 42–46, CTRL-K1 base. Primary readout is the probe, not regret.

## P-HEAD — FAILED. The loss is not the blocker.

|x − x\*|∞ of the emitted query, first 10 iterations (before the GP finds x\* on its own):

| arm | at rtg=min | at rtg=max |
|---|---|---|
| **MIXO-MSE** (20 MES + 10 oracle) | **0.342** | 0.437 |
| **MIXO-L1** (same mixture, median head) | **0.359** | 0.479 |
| reference: h207 MIXO (MSE, 50/50 mixture) | 0.182 | 0.156 |
| reference: CTRL-K1 (no oracle half) | 0.669 | 0.666 |

Registered: MIXO-L1 ≥ 0.45, i.e. the median tracks the MES **majority** instead of the
midpoint. Measured **0.359** against MSE's **0.342** — a difference of 0.017 where the
MSE-to-control span is 0.33. **The median head averages just as the mean head does.**

The mixture ratio behaves as expected and confirms the readout is measuring what it should:
h207's 50/50 mixture pulled to 0.18, this 2:1 mixture pulls to ~0.34, CTRL-K1 with no
oracle half sits at 0.67. The output moves with the *proportion* of oracle data and not
with the *loss function*.

**So "the DT cannot select" is not repairable by the loss.** Both L1 and MSE interpolate
between the modes in proportion to their mass. The next suspect is the architecture — a
head with no mechanism to make its output depend on RTG at all (gating, mixture density) —
which is a build, not a flag.

## Secondary — P-MIXR-L1 SUPPORTED, and L1 is not the drag h210 measured

| arm | regret |
|---|---|
| MIXR-L1 | **6.43** |
| MIXR-K1 (mse) | 7.17 |
| CTRL-K1 | 11.59 |
| MIXO-L1 (ceiling) | 10.26 |
| MIXO-MSE (ceiling) | 11.97 |

MIXR-L1 − MIXR-K1 = **−0.75** (se 1.21, better on 3/5), inside the band — **P-MIXR-L1
supported**. Worth recording against h210, which measured L1 as a **+3.88 drag** on the
K=8 base: **that drag was window-specific.** On CTRL-K1 the median head is free, and
nominally the best deployable number this project has produced (6.43, against MF-MES's
6.40) — though the difference from MIXR-K1 is inside noise and should not be read as an
improvement.

Notable: MIXR-L1's per-seed spread is far tighter than MIXR-K1's (5.28–7.73 vs 2.04–10.96).

## The oracle arms are not a ceiling here

MIXO-MSE 11.97 and MIXO-L1 10.26 are both **worse than CTRL-K1**. Under `mes_entropy` the
oracle half is labelled *worse* than the MES half — the smoke measured `rtg[0]` at
mes +0.129 / oracle −0.106 — because re-querying x\* is information-poor. So this mixture
degrades the training set rather than enriching it, as Amendment 1 anticipated. It is a
valid probe of averaging and not a performance ceiling; h207's 0.00 ceiling used
`terminal_improvement`, where the oracle half was labelled +50 better.
