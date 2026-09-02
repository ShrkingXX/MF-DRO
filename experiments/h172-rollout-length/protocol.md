# h172 -- if only the first step matters, how short can the rollout be?

STATUS: protocol locked, nothing run. TYPE: CONFIRMATORY.

## The implication being tested

h171 established by intervention that the teacher's FIRST step is what reaches
inference: a teacher consulting the acquisition on 1 of 8 steps works (16.96,
5/5) while one consulting it on 7 of 8 fails (43.94, 0/5). The seven later steps
affected only cost -- TAIL took 1.82x HEAD's wall time for a worse result.

The direct consequence: **rollout_length=8 may be seven times more rollout than
the method needs.** This tests it as a dose.

## The dose

`rollout_length` in {1, 2, 4, 8}, everything else the control's configuration
(MES teacher throughout). 8 is the control and is already measured.

## Why this is NOT simply h171 restated

Shortening the rollout changes more than the wasted steps. It changes:
  - the RTG label. rtg[0] = log(b_0) - log(b_T), so a shorter rollout measures
    information gain over fewer steps and rtg[0] shrinks mechanically.
  - the BTG label, similarly.
  - the DT's training context length (positions 0..T-1).
  - Bayesian early stopping has fewer steps to act on.

So a null here would NOT refute h171 -- it would say the savings are not
realisable by this particular lever. That asymmetry is stated up front so a null
is not read as evidence against the mechanism.

## Predictions

P1 rollout_length=1 and 2 land near the control (within ~3 rel%, improving ~5/5).
P2 wall-clock falls roughly in proportion to rollout_length.
P3 If P1 fails, it fails at the SHORT end (length 1 worst), because the RTG label
   degrades most there -- not because the teacher got worse.

## What this can RETRACT

R1 P1 holds -> the actionable claim is real: the method can run at a fraction of
   its rollout cost. This is the first change to what the CODE should do to come
   out of this front.
R2 P1 fails -> the savings are not realisable by shortening rollouts. h171's
   mechanism is untouched (see the asymmetry above), but the practical payoff
   is withdrawn and must not be quoted from h171's wall-clock numbers alone.
R3 A partial dose (2 or 4 works, 1 does not) -> locates the floor, and P3 says
   where to look for why.

## Named confound, checked before the numbers

Realised HF fraction against the control's 0.883, per h60's precedent and
h171's own SC1 firing on TAIL.

## Design

Borehole_8D, seeds 42-46, lengths {1, 2, 4}; length 8 is the existing control.
15 workers is 3 lengths x 5 seeds -- at the cap, so launched only while nothing
else is running.

## SC passed before launch

Smoke test at `rollout_length=2`: 180 trajectories generated, **all of length
exactly 2** (min 2, max 2). The parameter reaches `simulate_mf_trajectory` and is
not silently ignored — the failure mode that cost h169 an arm.
