# h152 -- The joint-information-gain teacher

STATUS: protocol locked, nothing run.
TYPE: CONFIRMATORY (Stage 0 gate + Stage 1 arm both pre-specified below).

## Motivation

Every "better teacher" tested so far (h145 ORACLE, h146 DIVERSE-GOOD, h149
RANDOM-POOL) was better or worse in a currency the reward does NOT measure --
proximity to x*. All of them collapsed rtg_target (0.976 -> ~0.29-0.32) and all
landed at 43.94 rel% on Borehole vs the MES control's 15.82.

This is the first teacher that is better in the reward's OWN currency.

## The objective, exactly

mf_dro.py:1953 labels the return as

    rtg[tau] = log(b_tau) - log(b_T)

so the total return is rtg[0] = log(b_0) - log(b_T). b_0 is fixed at rollout
start (it is _rollout_gumbel_b(ko_model) before any conditioning). Therefore

    argmax total information gain  ==  argmin b_T

The per-step rewards telescope: the total is a function of the TERMINAL
posterior only and is PATH-INDEPENDENT. Maximizing joint information gain is a
SET-selection problem over the T query points. The current teacher --
per-step argmax of cost-normalised MES (mf_dro.py:~1590) -- is the GREEDY
approximation to that set problem. Information gain about the max is
approximately submodular, so greedy carries a (1-1/e)-type optimality gap.
That gap is the headroom this experiment probes.

## Cost constraint (non-negotiable, else the arm is confounded)

rollout_length=8 is fixed and no budget is enforced inside the rollout, so
unconstrained argmin b_T degenerates to "8 HF queries" -- more information
bought with more money. The joint teacher is therefore constrained to

    total rollout cost <= cost of the greedy trajectory on the SAME state

computed by running greedy first on that state (needed for the comparison
anyway). Same budget, better allocation.

## Algorithm: cost-constrained beam search, scored by terminal b_T

Beam width B, branch factor k.
  nodes := [(ko_model, [], cost=0)]
  for tau in 1..T:
      for each node:  vectorised cost-normalised MES over all n_roi_candidates
                      -> take top-k (x, ell) pairs        [1 vectorised call/node]
      children := B*k expansions; drop any exceeding the cost cap
      condition each surviving child (sample_fantasy + make_fantasy_ko)
      prune to best B by ACCUMULATED MES (cheap surrogate; b is not computed
      per-node mid-search, only at the end)
  for each of the B survivors: compute true b_T via _rollout_gumbel_b
  return argmin b_T

At B=k=1 this reduces EXACTLY to the current greedy teacher. That identity is
a required sanity check (SC1 below).

## Delivery into the rollout

Via the EXISTING forced_x hook (mf_dro.py:1618, built for h145). The beam
emits x_1..x_T; fidelity is then re-chosen by the same info-gain criterion the
real path uses. This guarantees state extraction, make_fantasy_ko
conditioning, b_tau, rtg, btg and costs all run through identical code -- the
same property that licensed h145/h146's comparisons.

CONSEQUENCE, stated up front: forced_x replay re-samples its own fantasies, so
the REALISED b_T differs from the b_T the beam optimised. This is not a bug to
hide -- a planning advantage that does not survive fantasy resampling is not an
advantage the DT can be trained on. Stage 0 therefore gates on REALISED
rtg[0] under replay, not on the beam's internal b_T.

## Sanity checks (all must pass before Stage 1)

SC1  B=k=1 beam reproduces the greedy trajectory point-for-point.
SC2  beam's internal b_T <= greedy's internal b_T on every state (it is an
     argmin over a superset containing the greedy path; a violation means the
     search or the cost cap is wrong).
SC3  every emitted trajectory satisfies the cost cap.
SC4  use_roi=False path stays bit-identical (standing constraint).
SC5  emitted x are inside bounds and drawn from roi_candidates.

## Stage 0 -- the cheap gate (RUN FIRST)

On ~20 real mid-run states x 3 Borehole seeds, replay greedy and beam through
simulate_mf_trajectory and record REALISED rtg[0].

  lift := mean( rtg[0]_beam - rtg[0]_greedy )

GATE: proceed to Stage 1 only if lift > 0 and exceeds the replay noise floor,
where the noise floor is the s.d. of rtg[0] across repeated greedy replays of
the SAME state. If lift is inside the noise floor, STOP -- greedy is already
jointly near-optimal, and that is itself the result.

## Stage 1 -- the arm (only if Stage 0 passes)

Borehole_8D, seeds 42-46, n=5, frozen metric (rel% of |optimum| @cost_curve
200 via h83 sr_curve+grid). Compare against the h83 MF-DRO control
(15.82 rel%, improves 5/5, rtg_target 0.9761). No p-values at n=5.

## What this can RETRACT -- named before running

R1  Beam BEATS control  -> retracts "the MES selection rule specifically is
    what the DT needs". Replaced by "teacher reward-optimality is what
    matters, and greedy MES is a suboptimal approximation to it". This would
    be a METHOD IMPROVEMENT, not just a diagnosis.

R2  Beam achieves HIGHER rtg[0] but performs WORSE or EQUAL -> retracts
    "rtg_target collapse explains the failure", the current leading
    explanation. Sharpest available negative result: it decouples reward from
    outcome using a teacher better in the reward's own currency, and would
    mean the reward is not the channel at all.

R3  Stage 0 finds no lift -> retracts the PREMISE of this experiment, and
    says greedy MES is already effectively the joint optimum. That would
    explain why no substitute teacher has ever worked: there is nothing
    better to substitute. Cheap to establish, and worth establishing.

All three outcomes are informative. R3 is checked first because it is cheapest.
