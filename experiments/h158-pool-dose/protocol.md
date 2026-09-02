# h158 -- the registered POOL dose, answered on the tail axis

STATUS: protocol locked, nothing run. TYPE: CONFIRMATORY.

## Why this is being run now, having been declined before

The /loop prompt has carried "Registered follow-up: a POOL dose {16, 256, 4096}
tracing quality up and diversity down on one axis" for many ticks. I declined it
in the real pipeline and recorded the reason: h146/h149 showed the outcome is
flat in both quality and diversity, so every dose point would return 43.94 at a
cost of ~15 worker-hours. That reasoning was an ASSERTION from two endpoints.

It is now a PREDICTION from a mechanism, and it can be tested for minutes on the
offline harness instead of hours in the pipeline. Declining an expensive
experiment is only defensible if the cheap version gets run.

## The manipulation

h146's DIVERSE-GOOD endpoint is `argmax of POOL true-objective draws`
(worker.py:23, POOL=256). Sweeping POOL moves two things in opposite directions
on ONE axis:
  POOL=16    lower-quality endpoints, HIGH endpoint diversity
  POOL=256   h146's actual setting
  POOL=4096  higher-quality endpoints, LOW endpoint diversity

## Manipulation check (must pass before the outcome is read)

MC1 mean endpoint true-objective value RISES monotonically with POOL.
MC2 endpoint spread (mean per-dimension s.d. across trajectories) FALLS with POOL.
If either fails, the dose did not move what it is supposed to move and no
conclusion about quality or diversity may be drawn from it.

## Prediction

Under the tail account the MAX is FLAT across all three POOL values, inside the
failing band (25-34% of the control), because interpolating toward an
already-good point earns almost no information REGARDLESS of how good that point
is. Quality and diversity are both orthogonal to the reward.

## What this can RETRACT

R1 tail RISES with POOL -> higher-quality endpoints DO earn more information.
   The account's central claim (quality is orthogonal to the reward) is WRONG,
   and h149's reinstated mechanism goes with it. This is the costly outcome and
   it is named first.
R2 tail FALLS with POOL -> endpoint DIVERSITY is load-bearing after all, and
   h146's reading ("diversity does not rescue it") needs revisiting.
R3 tail FLAT inside the failing band -> confirms the account on an axis not yet
   tested, and retrospectively justifies not spending ~15 worker-hours on the
   pipeline version. The justification then rests on measurement, not assertion.

A flat result is NOT a null: it is the prediction, and it is the only one of the
three that leaves the account standing.

## Design

3 POOL values x 6 states (Borehole seeds 42/43) x N=100 x 10-model ensemble,
2 replicates. Noise floor from the same harness: 8-13%, so "flat" means the
three points sit inside one band, not that they are equal.

2 workers. h153 (5) + h155 (5) + 2 = 12 <= 15.
