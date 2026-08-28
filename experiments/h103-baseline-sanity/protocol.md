# H103 — are the registered baselines broken, or faithfully weak?

ZERO COMPUTE so far: a source audit against the author's MATLAB reference.
Written after the audit; labelled EXPLORATORY and not presented as pre-registered.

## Why this became load-bearing

The registered success test passes against MF-MI-Greedy and MF-GP-UCB, and those
two are the weakest optimisers in the comparison: 2.0 and 3.0 queries above their
own initial best out of ~25 and ~21 HF evaluations, against MF-MES's 21.0 of
22.0. If that is OUR defect, the whole h83 comparison rests on broken baselines
and every conclusion drawn from it -- including "MF-DRO loses to MF-MES" -- is
compromised.

## What the audit found

`src/baselines/additive_mfgp.py:109` sets, unconditionally:

    self.mean_val = allY.max() + 2.0 * rangeY

The reference, `mfBO/mfboPreProcessParams.m:229-238`:

    if strcmp(params.acqStrategy, 'MF-GP-UCB') | ...
       strcmp(params.acqStrategy, 'Illus-MF-GP-UCB') | ...
       strcmp(params.acqStrategy, 'GP-UCB') | ...
       strcmp(params.acqStrategy, 'MF-MI-Greedy')
      priorMeanVal = maxY + 2*rangeY;   % works best for UCB
    else
      priorMeanVal = meanY;             % works best for EI and the rest
    end

**MF-MI-Greedy is explicitly in the inflated-prior branch.** So is GP-UCB. The
port is FAITHFUL for both registered baselines, and the unconditional form is
correct because both methods our code implements fall in that branch.

## I nearly reported a porting bug, from partial evidence

Reading only lines 234-236 -- the two assignment lines -- shows an inflated prior
"for UCB" and a normal one "for EI and the rest", which reads as though a
non-UCB method like MI-Greedy should get `meanY`. That is the wrong conclusion
and I was one step from writing it up as a defect that invalidated the
comparison. **The branch CONDITION, four lines above, names MF-MI-Greedy
explicitly.**

Recorded because it is the same failure that produced the sd-vs-MAD
misdiagnosis, the invented dim-7 split, and the h100 override: reading part of
the evidence and concluding. The difference here is only that I checked before
publishing rather than after.

## What this means

The registered baselines are **faithfully ported and genuinely weak at this
budget.** The inflated prior is the reference author's deliberate choice, and its
own comment says it suits UCB; it makes the posterior mean high everywhere data
is absent, which drives exploration and suppresses exploitation. At ~20-25 HF
queries that shows up as almost no improvement over the initial design.

  - NOT our bug. The h83 comparison is not invalidated.
  - The pass in the previous entry remains real and remains uninformative about
    competitiveness, for the reason the concurrent session gave.
  - The open "two baselines under-optimise" question, carried in
    research-state.yaml for hours, is now ANSWERED: faithful port, weak method
    at this budget, deliberate prior choice upstream.

## What is NOT settled

Whether the reference's own budget regime differs enough that this prior is
reasonable there and not here. That would need the reference paper's budgets and
is a literature question, not a code one. It does not change any number in this
project, only how charitably the baselines should be described.
