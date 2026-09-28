# h212 analysis — CONFIRMATORY. 15/15 finals, 0 tracebacks.

Frozen metric imported from h83's `grid`; finals only; endpoint only.

## Result

| arm | bench | regret | comparison |
|---|---|---|---|
| **TI-PCT-H** (`terminal_improvement`, percentile) | Hartmann | **6.46** | **−5.27** vs K1-TI-H (se 2.00, **5/5**) |
| K1-TI-H (same label, floored/pinned) | Hartmann | 11.73 | — |
| CTRL-K1-H (`mes_entropy`, floored) | Hartmann | 5.93 | TI-PCT-H is +0.53 above it |
| **MIXRK1-PCT-B** (negative control) | Borehole | **10.70** | **+3.52** vs MIXR-K1 (se 0.69, **0/5**) |
| MIXR-K1-B (floored) | Borehole | 7.17 | — |
| TI-PCT-B | Borehole | 13.48 | +1.89 vs CTRL-K1-B (1/5) |

## Scoring

**P-FIX — SUPPORTED, and it is the larger of the two effects.** Un-pinning the inference
RTG target recovers **5.27 points on 5/5 seeds**, and brings `terminal_improvement` on
Hartmann from 11.73 to **6.46 — within 0.53 of the `mes_entropy` control**. The defect was
real and it was costly: conditioning on a value the training data no longer contains
(`max(batch_max, alpha·running_max)` becoming a ceiling once signed RTG goes negative).

**P-NEUTRAL — FAILED, as registered at-risk.** The percentile schema **hurts** where the
target already varied: MIXRK1-PCT is **+3.52 on 0/5** against MIXR-K1. This was predicted
before results (Amendment 1): the 90th percentile of a batch sits systematically *below*
`max(batch_max, alpha·running_max)`, so q=90 is a **less ambitious target**, and asking for
less gets less. The reading is not "the schema is uninterpretable" but "q=90 is wrong for
an unsigned label".

**So the percentile schema is not a universal replacement.** It repairs the signed-label
pinning (+5.27) and regresses the unsigned case (−3.52). A sign-aware rule — floored when
RTG is non-negative, percentile (or a higher q) when it is signed — is what the evidence
supports, and is not yet implemented or tested.

## Decomposing h211's unattributed +5.80 on Hartmann

| configuration | regret |
|---|---|
| CTRL-K1-H — `mes_entropy`, floored | 5.93 |
| K1-TI-H — `terminal_improvement`, floored, RNG unmatched | 11.73 |
| TI-PAR-H — `terminal_improvement`, floored, **RNG matched** (h213) | 8.99 |
| TI-PCT-H — `terminal_improvement`, **percentile**, RNG unmatched | 6.46 |

Both single-factor fixes are large and neither alone reaches the control. **The arm that is
missing is `terminal_improvement` + percentile + RNG parity**, which would say whether the
label has any residual cost once both defects are removed. Registered prediction for it,
made here before it is run: it lands within the band of CTRL-K1-H (5.93), i.e. the label's
apparent Hartmann penalty was entirely its *deployment* — the target schema and the
unmatched stream — and not the reward definition.
