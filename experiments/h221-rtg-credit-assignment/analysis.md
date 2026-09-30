# h221 — GATE MISS. Common random numbers do NOT restore the action signal. Arms not launched.

CONFIRMATORY on the gate. No Stage 1 run. Identity gate PASS at 122.29066752728207.

## The gate

Adjusted R² of `rtg[0]` on the first action `(x_0, ell_0)`, within member, n=100 rollouts:

| setting | per member | mean |
|---|---|---|
| baseline | 0.027, 0.005, 0.041 | +0.024 |
| **CRN** | −0.015, −0.010, −0.036 | **−0.020** |
| ADV | 0.083, 0.107, 0.011 | +0.067 |
| CRN + ADV | 0.051, −0.015, −0.053 | −0.005 |

**SC1 FAILED** (registered threshold: CRN raises adj R² above 0.10 on ≥2 of 3 members).
The supervisor launched nothing.

## What the registered retraction says, and it fires

Protocol: *"SC1 fails → fantasy noise is not what decouples the label from the action."*

Sharing the fantasy noise across a member's rollouts step-for-step leaves the first
action explaining **zero** of the rollout return. So the decoupling is not sampling
variance that CRN can cancel — it is structural. Two candidates remain, and this gate
separates them from the noise account without deciding between them:

1. **Credit horizon.** `rtg[0]` aggregates eight steps; the first action is one-eighth of
   the plan and the remaining seven re-optimise around whatever it did. h172 measured
   `rollout_length=1` at 13.69 vs 15.82 (better, 4/5) and h208 measured L1 ≈ L8 on the
   original — both consistent with the first action's value only being visible when the
   horizon is one step.
2. **Near-optimality of the first action.** MES's τ=0 actions vary by 0.35 in position
   but all sit near the acquisition argmax, so they vary little in *value*. CRN removes
   noise; it cannot create signal that the action distribution does not contain.

The advantage baseline alone lifts adj R² to +0.067 — a real but small effect, and one that
SC2 could not evaluate (the baseline is applied inside the batch generator, so the
before/after ordering comparison the check assumed was not available to it). Recorded as
a diagnostic, not a result.

## Consequence for the RTG thread

This closes the "noise" branch of the credit-assignment question. The label does not
encode the first action, and it is not because of the fantasy draws. The remaining
levers are the horizon (`rollout_length=1`, already measured favourably) and the action
distribution's value spread (the multi-teacher / low-quality-contrast line, which is
where the user's proposals sit). h220 — the head-averaging probe — is the other half of
the same question and completed the same day; see its analysis.
