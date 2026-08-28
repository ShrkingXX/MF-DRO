# H123 — The paper's beta_t, faithfully: a WIDENING ROI

STATUS: LOCKED. Registered before the code fix is applied and before launch.
TYPE: CONFIRMATORY. **The locked prediction is a NULL** (below), stated with
      grounds, so that a null is a registered outcome rather than a shrug.
COMPUTE: not yet launched. Requires the fix in section 2. Machine currently
         holds h117 (4), h120 (3), h122 (3); this waits for slots.

## 1. Why this has never been tested

The DRO paper writes `beta_t` with a subscript t, implying a schedule; the
implementation used a constant. h84's `ROI-ANN` arm appeared to close this. It
did not: today's bug finding showed its `_prog = n_real_iter / T_real` runs
against `T_real = bo_iterations = 4000` while these runs terminate on a COST
budget of 200, reaching ~104 HF observations on Borehole and ~18 on Hartmann.
Realized q moved 1.1 points on Borehole and 0.13 on Hartmann out of a configured
45. **ROI-ANN is a constant q~0.49 arm.** No schedule has ever run here.

The peer session independently confirmed the defect and sharpened the diagnosis:
the bug is not that 4000 was the wrong constant, it is that **the denominator was
a quantity the termination condition does not use.** HF-count/bo_iterations
cannot reach 1.0 by construction on any configuration this project runs, so any
schedule written against it is dead code.

## 2. The fix (to be applied when the in-flight runs drain)

`_prog` becomes cost-consumed / cost-budget — the only progress variable that
traverses [0,1] under a cost-terminated run. Gated so that `use_roi=False` stays
bit-identical and so that omitting the new schedule config reproduces current
behaviour exactly. A bit-identity check against a stored trace is run before
h123 launches, as with h117's gate G0.

## 3. The direction, and the fact that it inverts our prior

Acceptance is monotone INCREASING in beta — `mf_dro.py:1258` says so and the
bisection depends on it. GP-UCB's `beta_t` GROWS with t. Therefore **a faithful
`beta_t` schedule WIDENS the ROI as the run proceeds.**

This contradicts the direction of every ROI experiment either session has run:
q = 0.10 (h84/h90), q = 0.05 (h97/h107/h110/h111), ROI-FIX2 at realized q ~ 0.21,
ROI-ANN's *intended* 0.50 -> 0.05 tightening, and a tightening claim the peer
published and later withdrew. Every one of those tightens.

Registering the widening direction is therefore not a menu choice made after
seeing results — it follows from the monotonicity plus the paper's notation, and
it is recorded here BEFORE the run so that a reader can see the direction was not
selected post hoc. Both sessions agreed on it in advance.

## 4. Arms (Borehole_8D, seeds 42-46, paired)

  ROI-WIDEN   use_roi=True, quantile mode, q_t = 0.05 -> 0.50 by COST progress.
              The faithful direction: tight early, loose late.
  ROI-Q10     use_roi=True, constant q = 0.10. The established comparator.
  (control)   h83 `MF-DRO`, measured bit-identical to h84 `ROI-OFF` (137 and 132
              queries, 0 differing, three commits) -- see h120 Amendment 3.

Borehole because it is the ONLY benchmark where the ROI has any demonstrated
effect; a schedule that cannot beat a constant there will not do so elsewhere.

## 5. Prediction (locked): NULL

P1. ROI-WIDEN does NOT beat ROI-Q10 on final simple regret at >= 4/5 seeds with
    |mean|/sd >= 1.0. **I expect this to be null.**

Grounds, and they are strong: h97/h107/h110 established that q = 0.05 vs q = 0.10
is not separable across three seed sets, and h111 found no tightness effect off
Borehole. **Tightness has been a null axis everywhere it has been measured
properly.** A schedule is a time-varying tightness, so the prior expectation is
that it too is null.

P2. If ROI-WIDEN DOES clear the bar, that is a positive result against both
    sessions' stated expectation and must be reported at least as prominently as
    the null, and confirmed at fresh seeds before it is believed.

## 6. Why run a predicted null

Because "the paper's beta_t was never tested" is a hole in the write-up that
cannot be filled by argument, and because the arm that claimed to fill it did not
run as configured. A registered null converts "we used a constant where the paper
wrote a schedule" into "we tested the paper's schedule, in the faithful
direction, and it did not help" — which is the only version of that sentence
worth putting in a paper.

## 7. Limitations

- n=5, one benchmark, no p-values.
- A single widening profile (0.05 -> 0.50 linear in cost) stands in for a family.
  A null bounds this profile, not every schedule.
- Cost progress is a proxy for the paper's t. The paper's beta_t is a per-round
  confidence parameter with a specific log form; a linear q-ramp is a
  behavioural analogue, not that formula. Named here so no one later reads this
  as having implemented GP-UCB's beta_t literally.
