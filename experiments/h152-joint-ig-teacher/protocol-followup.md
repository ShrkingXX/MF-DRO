# h152b -- debiased selection for the joint teacher

STATUS: protocol locked, nothing run.
TYPE: **EXPLORATORY**. This is a method change made AFTER seeing Stage 0's
result, so it does not inherit Stage 0's confirmatory standing. Its own gate is
pre-stated below and will be honoured whichever way it lands.

## Diagnosis being acted on

Stage 0: planned lift +0.5461 (21/21), realised lift -0.1219 (6/21), winner's
curse +0.6680 (21/21). The beam scores each candidate path on ONE fantasy
sample path and argmins over B of them, so it selects lucky DRAWS rather than
informative DESIGNS. Only the (x, ell) sequence transfers to the rollout; the
y's are redrawn. So the quantity that actually transfers is

    E_y [ log b_T | (x, ell) path ]

and the beam was optimising a single-sample estimate of it.

## Two changes

C1  SELECTION (the fix that matters). Among the B survivors, choose by MEAN
    log b_T over `select_M` FRESH fantasy replays from the original model,
    instead of by each survivor's own single realised b_T. This is an unbiased
    estimate of the transferable quantity, so the curse should vanish.

C2  PRUNING. Revert intermediate pruning from single-path b_tau to accumulated
    cost-normalised MES. MES is an analytic acquisition value (an expectation),
    not a single realisation, so it is a far less curse-prone proposal
    heuristic. Elitism is RETAINED, which is what made this safe -- accumulated
    MES was the v1 pruner and it failed SC2 by pruning the greedy path away
    before elitism existed.

Elitism, the cost cap, and the forced_x delivery path are all unchanged.

## Sanity checks (must pass before the gate is read)

SC1b  B=1,k=1,select_M=1 still reproduces the greedy teacher step for step.
SC2b  realised-b_T of the SELECTED path <= realised-b_T of the elite path,
      measured on the SAME M replays (selection is now on the honest metric,
      so this must hold by construction).
SC3b  cost cap still respected.

## GATE (pre-stated, binding)

Re-run Stage 0 unchanged: 21 states, seeds 42/43/44, R=8 independent replays.

  PASS  realised lift over greedy > 0 AND > the 0.2218 noise floor
        -> the joint teacher is alive; Stage 1 (Borehole seeds 42-46) runs.
  FAIL  otherwise -> the joint-IG teacher is dead ON THE MERITS, not on a
        scoring bug, and h152's finding stands as: the joint optimum exists in
        plan and is unreachable in realisation because b_T cannot be estimated
        precisely enough to select on.

Intermediate outcome to name now so it cannot be spun later: a realised lift
that is POSITIVE but INSIDE the noise floor is a FAIL, not a "trend". At n=21
states with s.d. 0.22 that is the honest reading, and no p-value will be
computed to dress it up.

## What this can RETRACT

- If it PASSES, it retracts h152 Stage 0's headline ("unreachable in
  realisation") -- the advantage would be reachable, and Stage 0's failure was
  my scoring bug, not a property of the problem. Stage 0's analysis.md would
  need correcting, not just supplementing.
- If it FAILS, it retracts nothing new but CONFIRMS Stage 0 was not an
  artefact of single-sample scoring, which is currently its main vulnerability.
