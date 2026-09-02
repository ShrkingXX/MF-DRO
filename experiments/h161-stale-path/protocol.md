# h161 -- STALE-PATH: the discriminator that h159 and h160 were not

STATUS: protocol locked, nothing run. TYPE: CONFIRMATORY.
This is the test registered in findings.md when the model-selected hypothesis
was recorded as post-hoc. It is the arm that actually separates the accounts.

## Why h159 and h160 failed to discriminate, and this does not

h159 (beta=0) and h160 (beta=-2) were both locked as discriminators and neither
is one: both are still MODEL-SELECTED, so both accounts predict they work, and
both screens confirmed they stay far above the failing band (91.5%, 83.7%). Two
screens caught this for ~40 minutes before ~20 worker-hours were spent.

The real contrast is not WHICH rule selects the location. It is whether the
location was selected **for the state it is being fed into**.

Among the FROZEN arms the pattern is already suggestive:
  h153 frozen path, model-selected for THIS state      19.36, 5/5
  RANDOM-POOL       uniform from the same pool         43.94, 0/5
  ORACLE / DIVERSE  chosen by criteria outside the model 43.94, 0/5
h161 breaks the remaining tie: model-selected, but for a DIFFERENT state.

## The arm

Identical to h153's two-pass wrapper, except pass-2 replays a path taken from a
call **LAG=2000 rollouts earlier** (~10 real iterations back, since
rollouts_per_iter=200). The path is genuinely model-selected -- an ordinary MES
rollout -- just selected against a model that has since seen ~10 iterations of
new data. Before the buffer fills, the current path is used, and the fraction of
calls actually served stale is recorded.

Everything else is h153: state extraction, conditioning, b_tau, rtg, btg, costs
and the fidelity rule all run through identical code. src/policy/mf_dro.py is
NOT modified.

## Predictions (opposed -- this is the point)

MODEL-SELECTED-FOR-THIS-STATE account: STALE-PATH **FAILS**, ~43.94, ~0/5.
MODEL-SELECTED-AT-ALL account:         STALE-PATH **WORKS**, ~19-20, ~5/5.

Note both accounts predicted success for h159 and h160. Here they split. That
is the entire reason to spend the arm.

## Sanity checks, read before the number

SC1 fraction of calls served a stale path (target: >0.9 after warmup)
SC2 mean lag actually applied, in rollouts
SC3 realised HF fraction against the control's 0.88 -- a collapse voids the arm
SC4 path reproduction error in pass 2 (must be exact, as in h153)

## What this can RETRACT

R1 STALE-PATH FAILS -> the location must be model-selected FOR THE CURRENT
   STATE. This is the first positive mechanism to survive since "the MES rule"
   (retracted by h155) and "target collapse" (refuted by h153). It would also
   mean h153 works for a reason narrower than "model-selected".
R2 STALE-PATH WORKS -> "model-selected at all" suffices, and staleness is free.
   Then what separates h153 from ORACLE/DIVERSE/RANDOM is something ELSE again,
   and the post-hoc paragraph in findings.md must be rewritten a third time.
R3 Intermediate (25-35 rel%) -> inconclusive at n=5, reported as such.

## Design

Borehole_8D seeds 42-46, n=5, frozen metric, no p-values. 5 workers alongside
h159's 5 = 10 <= 15. h159 is being allowed to finish rather than killed: it is
non-discriminating for the accounts but it DOES validate the harness's C7
forecast (91.5%), and h153 proved unvalidated harness conditions can be wrong by
2.7x. That is worth one arm.

## Design error caught by the smoke test, before the arm ran

LAG was locked at 2000 rollouts on the assumption that a batch is
`rollouts_per_iter=200`. The smoke test's STATE-DIAG line shows the batch is
**n_traj=60**. LAG=2000 is therefore ~**33** of the ~60 real iterations, which
would have left **more than half the run in warmup**, using the CURRENT path and
being byte-identical to h153 over that stretch. The manipulation would have been
diluted to roughly half strength and the arm would have been uninterpretable.

**LAG corrected to 600** (10 iterations at 60 trajectories each), giving ~83%
of iterations genuinely stale. h161 was killed ~4 iterations in and relaunched.
SC1 (stale fraction, target >0.9 after warmup) now has a realistic target of
~0.83 and is recorded as such.

This is exactly what the smoke test was for. It cost ~5 minutes and one worker;
discovering it from SC1 after the fact would have cost ~10 worker-hours and an
uninterpretable arm.
