# h204 — a curriculum-annealed exploit teacher: a DEPLOYABLE analog of h145's oracle

**CONFIRMATORY.** Locked before any code is written and before any result exists.
**Human-proposed**, developed through discussion (curriculum-learning framing,
web-researched precedent, then narrowed to a concrete design by three follow-up
questions: generation order, delta vs default MES, fidelity handling).

## The question

h201 showed that pairing a teacher whose LATE-step action is a converged, low-variance
answer with a K=8 window (which moves the DT's readout to that late step) produces a
near-perfect result (0.00 rel%, 5/5). But that teacher needs x*, unavailable at run time
-- it is a ceiling, not a method.

**h204 asks whether the SAME property -- a converged, low-noise target sitting at the
read position -- can be manufactured WITHOUT an oracle**, by having the acquisition
RULE itself change shape across the rollout: exploratory early, exploitative late.

## Why this is not the same question as h198/h199

h198/h199 change the DECISION CRITERION per step (choose the action that maximises
expected future value, via lookahead simulation). h204 changes NEITHER the criterion's
INPUTS nor adds simulation -- it changes WHICH ACQUISITION FAMILY MEMBER is used at each
step, via a single scalar schedule, reusing h155's existing UCB-on-HF-posterior location
mechanism (`rollout_policy="ucb_loc"`) rather than inventing new machinery.

**Delta vs the default MES teacher**, stated precisely (from discussion): the default
teacher's OBJECTIVE never changes across tau -- `compute_joint_mf_mes` always asks
"where would querying teach the GP the most?", at every step, regardless of how much
conditioning has already happened. h204 asks that question early and switches to
"where does the GP currently believe the optimum is?" late. Measured this session:
under MES, the tau=7 HF-fraction target (0.731) is not meaningfully more converged than
tau=0's (0.909) -- same rule, same character of noise. h204 is built specifically to
break that: an information-seeking criterion keeps probing wherever the GP is
UNCERTAIN, which need not settle down over a rollout; a shrinking-beta UCB converges
toward the CURRENT posterior mode, a comparatively more stable quantity BY DESIGN. This
is stated as a design rationale, not a proven fact -- SC-DIVERSITY below is what checks
it.

## Design

Single pass, same `for tau in range(rollout_length)` loop every rollout_policy branch
already uses (verified: no `rollout_policy` branch generates two rollouts or
post-processes one).

**Location**, generalising h155's `ucb_loc` branch (mf_dro.py:1755) from a fixed
`ucb_loc_beta` to a per-step schedule:

    beta(tau) = beta_max * (1 - tau / max(rollout_length - 1, 1))    # beta_max -> 0
    x_tau = argmax_x [ mu_H(x) + beta(tau) * sigma_H(x) ]            # over roi_candidates

`beta_max = ucb_loc_beta`'s existing default (2.0), reused rather than a new constant.
At tau = rollout_length-1, beta=0 and this IS argmax_x mu_H(x) -- pure exploitation of
the current HF posterior mean, computable from the GP alone.

**Fidelity is UNTOUCHED** -- reuses h155's/h145's existing "info gain of HF vs LF AT
the chosen point" block VERBATIM (mf_dro.py:1763-1786: build the HF proxy, Thompson-
sample y* over the SAME roi_candidates pool, score HF/LF at the one chosen x_tau, cost-
normalise, argmax). This is the same reasoning h155 already documented: h60 showed a
different-teacher-RULE change can silently collapse fidelity to ~99% one class and
confound "teacher rule" with "fidelity mix"; keeping fidelity on the untouched channel
means any h204 result is attributable to the LOCATION schedule alone. Given h202 found
the WINDOW independently saturates the fidelity head by sequence length, touching
fidelity a second time here would tangle three channels instead of two.

**RTG label: `mes_entropy`, UNCHANGED.** One-change-at-a-time against every existing
control (CTRL-K1, h201). This also has a specific reason to be safe here that it was
NOT safe for h198: h180 established (bit-level) that the RTG TARGET is architecturally
near-inert on which action the DT emits -- MES-FROZEN and STALE-PATH, two different
target manipulations, produced BIT-IDENTICAL actions on 5/5 seeds. h198's label fork
existed because its TEACHER explicitly optimises the labelled quantity by construction
(argmin b_T), so a label mismatch was a real confound for THAT teacher. h204's teacher
does not optimise entropy at any point -- exactly like h145's oracle, which also kept
the `mes_entropy` label throughout and still produced h201's result. No factorial is
registered for this reason, stated rather than assumed.

## Arms

| arm | teacher | window |
|---|---|---|
| **A** | curriculum-anneal | K=8 |
| **B** | curriculum-anneal | K=1 (matched control) |
| CTRL-K1 | MES (existing) | K=1 -- **already in hand, 11.59** |

B is mandatory, mirroring h201's own structure: without it, a result in A cannot
distinguish "the annealed trajectory-shape helps" from "the window helps regardless of
teacher" (h202's reconciliation is precisely why this distinction cannot be skipped).

## SCs, registered before running

1. **Reduction identity.** With `beta_max = beta_min` (no annealing, i.e. constant
   `ucb_loc_beta`), the teacher must reproduce h155's `ucb_loc` EXACTLY, step for step,
   same seeds. A generalisation that doesn't collapse to its own special case is a bug.
2. **SC-DIVERSITY (GATE).** tau=(rollout_length-1) action SD across rollouts, normalised,
   against the MES teacher's own SD on the same states -- same measurement and same
   >=25% bar h199 used. Below it: degenerate, GATE MISS, Stage 1 does not run (h146's
   lesson). This is the check for whether "posterior-mean argmax should vary rollout to
   rollout" (the design's central claim) is true rather than assumed.
3. **SC-FIDELITY.** Realised LF fraction against CTRL-K1's 0.261 and against h196/h197's
   0.085-0.092, measured under BOTH K=1 and K=8 (h202 showed window and teacher can
   interact on this channel independently). Not gated -- reported, since the window's
   fidelity tax is now an established, teacher-independent cost (h201A/h196/h197 table)
   and this SC exists to confirm h204 pays the SAME tax, not a different one.

## Prediction, committed now

- **P1** -- Arm A beats CTRL-K1 (< -1.26) AND beats Arm B. The manufactured convergence
  works: a deployable teacher can reproduce a meaningful fraction of h201's ceiling
  without an oracle. This is the result that would matter for the paper.
- **P2** -- Arm A ~= Arm B, both ~= CTRL-K1. The annealed trajectory does not produce a
  low-variance target the way h145's forced convergence did -- SC-DIVERSITY likely also
  shows it, and the design's central bet failed.
- **P3** -- Arm A worse than both B and CTRL-K1. The window's fidelity tax (paid
  identically regardless of teacher, per the table above) is not offset by this
  teacher's benefit, unlike h201A's oracle. Would show the ceiling is much closer to
  h201A's oracle-specific mechanism than to any GP-only teacher tried so far.

## What each outcome RETRACTS

P1 retracts nothing already claimed but would be the FIRST teacher-design intervention
this session to beat CTRL-K1 without an oracle -- upgrading the front from "we understand
why teachers fail" to "we have a fix". P2 retracts the specific mechanistic claim in the
Design section (that beta-annealing produces a materially more converged target than
MES's constant rule) -- stated as retractable because it is a genuine guess, not a
measured fact, going in. P3 retracts nothing but sharpens h201's reconciliation further:
it would mean NO GP-only rule tried so far (MES, regret-lookahead h198/h199, now this)
reproduces the oracle's benefit, only the oracle itself does.

## Cost

Cheap relative to h198/h199: one extra Thompson-sampled fidelity-at-a-point call per
step (K=10, the SAME call h145's forced_x and h155's ucb_loc already make), no extra
fantasies, no lookahead simulation. Same order of cost as the default MES teacher --
h197 measured ~80 min/seed for a K=8 MES-teacher run. 10 workers for both arms.

**QUEUED.** Compute is at the 15-worker cap (h198a, h198b, h201B). Runs when slots free,
via a supervisor mirroring h201's (SC1 + SC-DIVERSITY gate first; Stage 1 launches only
on PASS).
