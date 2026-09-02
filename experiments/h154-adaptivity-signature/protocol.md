# h154 -- does the DT's own policy show an ADAPTIVITY SIGNATURE?

STATUS: protocol locked, nothing run.
TYPE: **EXPLORATORY**, and weak by construction. See the caveat.

## Why now

h153 (MES-FROZEN) costs ~4 hours of 5 cores. It tests whether removing
ADAPTIVITY from a teacher whose QUALITY equals the control's reproduces the
43.94 failure. If the adaptivity hypothesis is already contradicted by data I
ALREADY HAVE, h153 is a waste and should be killed early.

So: test a PREDICTION of the hypothesis against the completed runs, cheaply.

## The prediction

If a frozen (state-independent) teacher teaches a state-independent POLICY,
then the DT's own real queries in the frozen arms should not respond to what it
observes, while the control's should.

Two measures, on the real post-init query trace only (x_t_trace, y_t_trace,
both already serialised -- checked with tools/check_fields.py before locking):

M1  RESPONSE-TO-OUTCOME. Pearson r between the standardised observed value y_t
    and the next step size ||x_{t+1} - x_t|| (domain-normalised).
    Adaptive: NEGATIVE (a good outcome keeps the policy nearby).
    Non-adaptive: ~0.

M2  SEQUENCE PREDICTABILITY. Lag-1 autocorrelation of the query sequence. A
    policy driven only by the slowly-varying RTG/BTG conditioning tokens, with
    the state ignored, should be SMOOTHER than one reacting to observations.
    Non-adaptive: HIGHER.

Arms compared, all Borehole seeds 42-46, all already complete:
  control MES (closed-loop) | ORACLE | DIVERSE-GOOD | RANDOM-POOL

## Predicted pattern if the adaptivity hypothesis holds

  M1: control clearly negative; the three frozen arms near zero.
  M2: control lowest; the three frozen arms higher and similar to each other.

## GATE (pre-stated)

CONSISTENT   the predicted pattern appears in both measures -> h153 continues,
             and h154 is corroborating (NOT confirming) evidence.
CONTRADICTED control is indistinguishable from the frozen arms on BOTH measures
             -> the adaptivity hypothesis loses its main observable prediction.
             h153 still runs to completion (it is the direct test and is
             already 20% through) but its prior drops sharply, and that must be
             written down BEFORE its result lands, not after.
MIXED        one measure fits, one does not -> reported as MIXED. Named now so a
             half-fit is not read as support.

## CAVEAT, stated before running, because I have been burned here exactly once

h150 tested "policy distillation of MES" with query-level statistics and the
result RETRACTED a finding I had already published. Query-level statistics on
n=5 are weak evidence about what a network internally represents. h154 can
lower h153's prior; it CANNOT substitute for h153, and no h154 outcome will be
reported as having established the mechanism.

Additional known confound: on Borehole the frozen arms improve 0/5, so any
measure conditioned on "after an improvement" has an empty bucket. M1/M2 were
chosen specifically to avoid conditioning on improvement.

## What this can RETRACT

- CONTRADICTED would retract the adaptivity hypothesis's status as the leading
  explanation in findings.md (currently written as h153's motivation), demoting
  it to "direct test pending" before that test reports.
- It cannot retract h152's measured open-loop penalty (+0.1594), which is a
  separate, directly measured quantity.

## Companion cheap check (same commit, separate result)

h154b: re-run h152's open-loop penalty harness on HARTMANN traces. The +0.1594
penalty is currently a Borehole-only number. 1 worker, offline harness, minutes.
Registered outcome: if Hartmann shows no penalty, the penalty is
benchmark-specific and h152's generality claim must be narrowed.
