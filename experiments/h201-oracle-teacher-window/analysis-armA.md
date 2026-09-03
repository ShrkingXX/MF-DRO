# h201 arm A — **P1, and decisively.** oracle teacher + K=8 window reaches ~x* on 5/5 seeds

**CONFIRMATORY**, 5/5 finals. CEILING/DIAGNOSTIC, NOT A METHOD -- x* is not available
at run time.

## Result

| arm | final regret |
|---|---|
| **h201A oracle teacher, K=8 window** | **0.00** |
| h194 CTRL-K1 (MES teacher, no window) | 11.59 |

Paired **−11.59** (se 0.19), better on **5/5**, essentially the largest possible margin
against this control.

## Verified real, not a metric artifact

Best HF value observed: 309.5752-309.5753 across seeds, against the known optimum
309.5751 -- an actual query, not a rounding quirk of the readout. The DT's proposed x is
NOT literally x* (up to 6.8% relative deviation in the worst coordinate at the best
query), but Borehole is flat enough near its optimum that this closes the regret to
0.00 at reporting precision.

It happens EARLY: query 23-49 of ~105-108, cost 85-130 of the 200 budget. Every seed
found it well before the budget ran out.

## This is the mechanism's sharpest confirmation to date

The prediction was explicit and pre-registered: the mechanism says the DT emits its
teacher's action MEAN AT THE READ POSITION. Under this teacher, tau=7 (the K=8 read
position) is x* with zero across-rollout variance (tau7_max_dev_from_xstar=0.0,
confirmed in-run over 6360 rollout paths). If the mechanism is right, the DT should
emit ~x*. It does, on 5/5 seeds.

This is NOT the same claim as "a good teacher helps" -- SC-DIVERSITY and this session's
prior caveats apply: the teacher's tau=7 action is a CONSTANT, and the DT is shown here
to be very good at reproducing a constant when it sits at the position actually read.
That is exactly the mechanism, demonstrated positively for the first time rather than
only through negative/null results.

## What is NOT yet resolved

**Arm B (K=1, the matched control for the SAME oracle teacher) is still running.**
Without it this is oracle-teacher+window vs MES-teacher+no-window -- TWO things differ
at once. Arm B isolates whether the WINDOW specifically is doing the work (the DT
reading position 7 instead of 0) or whether ANY readout of this teacher's action would
have worked (e.g. if K=1 also does well because tau=0's... no, tau=0 for this teacher is
UNIFORM RANDOM by construction, so arm B should reproduce something closer to h145's
+28.13 failure). That prediction is registered now, before arm B's result is read.

This file will be updated (not overwritten) once arm B completes.


## Reconciliation with h196/h197 (human challenge, 2026-09-03): why doesn't K=8 win generally?

Measured directly, not argued: h201A's fidelity head saturates IDENTICALLY to h196/h197.

| arm | lf_frac | n_queries |
|---|---|---|
| h201A oracle + K=8 | 0.107 | 106 |
| h196 MES + K=8 | 0.085 | 105 |
| h197 MES + K=8 + spec | 0.092 | 105 |
| CTRL-K1 (MES, K=1) | 0.261 | 116 |

The window's COST (h202's sequence-length fidelity saturation, ~9% fewer queries per
budget) is paid by every K=8 arm regardless of teacher. What differs is the BENEFIT: MES's
tau=7 action is not meaningfully more converged than its tau=0 (HF target 0.731 vs 0.909,
still an exploratory pick under the same criterion), so h196/h197 pay the tax for nothing
and net worse. The oracle's tau=7 is the answer, zero variance, so the benefit swamps the
identical tax by orders of magnitude.

CORRECTED FRAMING: "the window works" is imprecise. The window is necessary
infrastructure (it must move the readout to expose a good late-step target) but NOT
sufficient -- it is neutral-to-harmful with an ordinary teacher and only pays off paired
with a teacher whose late-step target is worth reading. This was h201's own registered
purpose (arm A vs arm B isolates this) and requires no new hypothesis to explain h196/h197.