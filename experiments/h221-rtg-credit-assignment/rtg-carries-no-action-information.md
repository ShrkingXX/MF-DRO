# The RTG label is uncorrelated with the first action — for BOTH teachers

EXPLORATORY. Direct measurement on a live policy (Borehole, seed 42, cost 40), 100
rollouts per ensemble member per arm. No new runs of the pipeline.

## The measurement

How much of the rollout outcome `rtg[0] = log b_0 − log b_T` is explained by the **first
action** `(x_0, ell_0)`? Within an ensemble member (so `b_0` is shared), **adjusted** R²:

| rollouts | adj R² per member | mean | τ=0 action spread |
|---|---|---|---|
| MES | 0.001, 0.041, −0.022 | **+0.007** | 0.35, 0.33, 0.37 |
| random | −0.076, −0.030, −0.030 | **−0.045** | 0.69, 0.70, 0.71 |

**Both are zero.** The first action varies plenty — 20 distinct actions out of 20 under
MES, spread 0.35; 0.70 under the random teacher — and the return varies too
(`rtg[0]` sd 0.10–0.16). They just do not vary *together*.

## RETRACTION: h207's SC4 figure was a fitting artefact

h207's Stage 0 reported "the τ=0 action explains **14%** of the 8-step outcome in random
rollouts (R² 0.147 / 0.108 / 0.172)" and I have quoted it since. That was **raw** R² with
**10 parameters on n=100**, whose expected value under the null is (p−1)/(n−1) ≈ **0.09**.
Almost the whole figure was the parameter count. Adjusted, it is a few percent at most, and
in this measurement it is zero. **The claim that the first action explains a meaningful
share of the rollout outcome is withdrawn.**

## Why this matters more than the head or the schema

`rtg[0]` is the *only* channel by which rollout steps τ>0 can reach the real query —
h208 measured the other one (positions 1–7 as training supervision) as inert, and measured
the label horizon as inert too. So the design intent is: the plan's outcome labels the
first action, and conditioning on a high outcome selects a good first action.

**That cannot work if the outcome does not depend on the first action.** And it does not:
`rtg[0]` is dominated by the *other seven* steps and by fantasy-sampling luck. The label
tells the DT *"this rollout got lucky"*, not *"this action was better"*.

This sits **upstream** of everything else on the RTG thread. A non-averaging head (h220), a
sign-aware target schema (h212), RNG parity (h213) — none of them can extract selection
from a label that does not encode it. It also explains why adding low-quality contrast did
not make RTG selective: the random half's returns are *equally* uncorrelated with its
actions (−0.045).

## Three ways to make the first action's value show up in `rtg[0]`

1. **Common random numbers across a member's rollouts.** Today every rollout draws its own
   fantasy `y` at every step, so the noise is independent and swamps the action signal.
   Seeding the fantasy sampler identically per step index across a member's rollouts makes
   two rollouts that differ only in their first action directly comparable. CRN is already
   used in this codebase (`regret_lookahead_teacher`) for exactly this reason. Cheapest of
   the three and targets the measured defect directly.
2. **Shorten the credit horizon.** With `rollout_length=1`, `rtg[0]` is a pure function of
   step 0. h172 measured L=1 at **13.69 vs the control's 15.82** on MF-DRO (better on 4/5)
   and h208 measured L1 ≈ L8 on the original. Consistent with this diagnosis, and already
   half-tested.
3. **Advantage-style credit.** Label the first action by outcome(rollout) minus the mean
   outcome of the member's other rollouts — a baseline subtraction that removes the shared
   luck. Cheap to compute from an existing batch, no extra rollouts.

(1) and (3) compose and neither needs new rollouts.
