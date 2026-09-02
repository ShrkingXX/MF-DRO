# h153 MES-FROZEN — PRELIMINARY, n=1 of 5. The forecast SPLIT.

CONFIRMATORY. Forecast committed blind, several ticks before any result.

| | forecast | observed (seed 43) | verdict |
|---|---|---|---|
| rtg_target | **0.83–0.94** (82–96% of control) | **0.3361** (34%) | **FAILED** |
| rel% | near control, not 43.94 | **20.50** | **HELD** |
| improves | ~5/5, not 0/5 | **1/1** | **HELD** |

Sanity checks all pass: SC1 path reproduction error **0.0** (exact), SC2
open-loop penalty **+0.4005** (positive, as required), SC3 fidelity flip
fraction 0.153, over 6060 wrapped rollouts.

## The harness's central forecast failed, and it is the one I have been quoting

For several ticks I reported "C2 retains 90.9% of the control's tail" across four
harness measurements (90.9, 95.5, 95.9, 87.3%), and built the h153 forecast on
it. **The real number is 34%.** That is the largest instrument error yet — a
factor of ~2.7, far outside the 8–13% noise floor and outside every other
condition's error (C1, C3, C4, C5, C6 were all within ~30%, most within noise).

SC2 shows why, directly: the **real** open-loop penalty is **+0.4005**, against
the **+0.16** the same harness measured offline. The harness understates the cost
of freezing by ~2.5x. Its C2 condition replays a frozen path with fresh fantasies
from the SAME start model; the pipeline derives the path from a pass-1 rollout
and then trains the DT on those trajectories across ~60 real iterations, so the
states drift in a way the offline replay never sees.

**C2 was the one condition with no real-arm counterpart to check against.** Every
other harness condition was validated against an arm that had already run. C2
was validated against nothing, and it is the one that broke. That is a general
lesson about the instrument, not a detail.

## The substantive result is more interesting than the forecast miss

rtg_target collapsed to **0.3361** — squarely in the failing arms' band
(ORACLE 0.3113, RANDOM-POOL 0.2965, DIVERSE-GOOD 0.3285). **And performance did
not fail**: 20.50 rel% improving 1/1, against those arms' 43.94 improving 0/5.

**A collapsed conditioning target does NOT by itself cause the failure.** The
account's causal half — "the failing arms collapse the target, and that is why
they fail" — is contradicted by this seed. The descriptive half (the failing arms
DO have no informative tail, which h157/h158 established against four real arms)
is untouched.

## Named consequence if this holds at n=5

RETRACTS the causal reading of the tail account from findings.md and from the
published report. What survives is: target collapse is a *correlate* of the
failing arms, not their cause, and the actual cause is still unidentified. The
open-loop penalty would then be real (+0.40, larger than measured) but still not
the mechanism.

## Status

**n=1. No conclusion is being drawn at n=1** beyond recording that the forecast
split and which half failed. Seeds 42/44/45/46 are running.
