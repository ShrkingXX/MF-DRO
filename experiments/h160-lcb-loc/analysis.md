# h160 — screen returns **NO-GO**. Arm NOT launched. And the screens say something.

| replicate | C8 (β=−2) | C1 control | C8/C1 |
|---|---|---|---|
| 0 | 0.842 | 1.020 | 82.5% |
| 1 | 0.614 | 0.901 | 68.2% |
| 2 | 0.924 | 0.972 | 95.0% |
| 3 | 0.838 | 0.950 | 88.2% |
| **mean** | **0.8046** | 0.9608 | **83.7% (sd 11.4)** |

Failing band 32.5%. Gate was ">70% = does not discriminate, do not launch".
**83.7% → NO-GO. The pipeline arm was not launched.** Threshold never moved;
round 1 returned 75.8% on two replicates spread 31% apart, so two more
replicates were added to make the comparison against the threshold meaningful.

## Why it does not discriminate — my design error, again

LCB-LOC picks argmax(μ − 2σ): confidently mediocre points. I locked it as
"model-selected but anti-informative". But **LCB is still model-selected**, so
the model-selected account predicts it works — and the harness says it earns
83.7% of the tail, so the information account predicts it works too. Both
predict success. Non-discriminating by construction.

This is the second arm in two ticks I locked as a discriminator that is not one
(h159, β=0, forecast 91.5%). In both cases the screen caught it **before** the
pipeline arm was funded. Two screens cost ~40 minutes; the two arms they
prevented would have cost ~20 worker-hours to produce results both accounts
already predicted.

## What the three screens establish on their own

| teacher | β | tail as % of control |
|---|---|---|
| UCB-LOC | +2 | 102.0% |
| EXPLOIT-LOC | 0 | 91.5% |
| LCB-LOC | −2 | 83.7% |
| failing arms (frozen, externally chosen) | — | ~30% |

Monotone in β, and **every closed-loop teacher stays far above the failing
band** — even one steered deliberately at low-value, low-uncertainty points.
The conditioning target is a function of whether the teacher **adapts**, almost
regardless of what rule it adapts by. Adaptive conditioning keeps moving any
rule onto new ground.

Combined with h153 (frozen, target collapsed to 0.323, performance fine), the
separation is now clean: **rtg_target tracks adaptivity; performance does not
track rtg_target.** Two different things, and only the second one matters.

## Caveat carried from h153

C8 is an unvalidated harness condition, exactly like C2 — which was wrong by
2.7×. This screen is therefore used only as go/no-go on discrimination, never as
a prediction of the arm's outcome. That restriction was recorded in the protocol
before the screen ran.
