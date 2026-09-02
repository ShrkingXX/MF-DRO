# h156 -- the target is a MAX, so the upper TAIL is what matters, not the mean

STATUS: protocol locked, nothing run.
TYPE: CONFIRMATORY, and it makes a PRE-REGISTERED PREDICTION about h153, which
is still running. That prediction is recorded here before h153 reports.

## The gap this addresses

findings.md currently carries an honest hole: the measured open-loop penalty is
+0.0845 to +0.1594 rtg units (h152/h154b), which is "real but NOT on its own
large enough to explain the 0.976 -> 0.311 rtg_target collapse". I have been
treating that as a magnitude problem for the adaptivity story.

It may not be one, because I was comparing the wrong statistics.
mf_dro.py:2056-2060:

    rtg0_list  = [traj['rtg'][0] for traj in rollout_batch]
    batch_max  = max(rtg0_list)
    running_max_rtg = max(running_max_rtg, batch_max)
    return max(batch_max, alpha_rtg * running_max_rtg)          # alpha_rtg=0.5

**rtg_target is a MAXIMUM over the batch (floored by half the running max), not
a mean.** Every penalty I have measured is a penalty on the MEAN. A mean shift
of -0.16 and a max shift of -0.67 are perfectly compatible if the intervention
also compresses the ACROSS-TRAJECTORY SPREAD: the max of N draws is a function
of the upper tail, and collapsing the tail moves the max far more than the mean.

## The measurement

From one fixed state, generate N=100 trajectories under each condition and
compare mean, s.d. AND max of rtg[0]:

  C1 CLOSED-LOOP    N independent greedy MES rollouts (the control).
  C2 OPEN-LOOP-OWN  for each trajectory: derive its own greedy path (adaptively,
                    with its own fantasy draws), FREEZE it, replay with FRESH
                    fantasies. This is h153's design exactly.
  C3 OPEN-LOOP-RAND for each trajectory: a uniformly random path from the same
                    pool, replayed. This is h149's design.

9 states (Borehole seeds 42/43/44 x 3 cuts). Offline harness, ~2 workers.

## Statistic of record

  tail ratio  = max(rtg[0]) / mean(rtg[0])   within a condition
  max gap     = max_C1 - max_Cx
  mean gap    = mean_C1 - mean_Cx

The claim under test: **max gap >> mean gap**, i.e. the tail ratio is much
higher for C1 than for C2/C3.

## PRE-REGISTERED PREDICTION ABOUT h153 (recorded before h153 reports)

h153 freezes each rollout's own adaptively-derived path -- exactly C2. So:

  If C2's MAX is close to C1's MAX, then h153 should NOT collapse rtg_target,
  and therefore h153 should NOT reproduce the 43.94 failure. h153 would come
  out near the control, and the adaptivity hypothesis would be REFUTED by its
  own direct test.

  If C2's MAX collapses toward C3's, h153 SHOULD reproduce the failure.

This is a genuine forecast, not a post-hoc reading: h153 is ~25% through and
its result is not yet visible in any form.

## What this can RETRACT

R1 max gap ~= mean gap (no tail collapse) -> RETRACTS the reading in
   findings.md that the open-loop penalty could explain the target collapse at
   all. The magnitude hole would be real and the adaptivity story would need a
   different bridge, or none.
R2 tail collapse confirmed for C3 but NOT C2 -> h153 is predicted to SUCCEED,
   which would mean the frozen-path mechanism is not what h145/h146/h149 share,
   and the h152 confound, while real, is not the operative cause. I would have
   to withdraw the framing that currently motivates h153 in findings.md.
R3 tail collapse for both -> the bridge holds and h153 is predicted to fail;
   the magnitude caveat in findings.md and in the published report can be
   removed rather than merely restated.

All three are informative. R2 is the one that would cost me the most, and it is
named first-class so it cannot be quietly reread as R3 later.
