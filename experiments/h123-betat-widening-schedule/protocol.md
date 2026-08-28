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

---

## AMENDMENT 1 (2026-08-28) — the grounds for the locked null are REFUTED.
## Filed before launch; h123 has not run and has no result files.

Section 5 justified the locked null with: "tightness has been a null axis
everywhere it has been measured properly", citing h97/h107/h110 (q=0.05 vs 0.10)
and h111 (two settings spanning 2x).

**h125 refuted that.** Every cited study is a 2x contrast or narrower. At 5x —
ROI-Q10's realized q=0.100 against ROI-ANN's realized q~0.495, paired, n=5 —
Borehole final_regret moves **+9.018 with effect 5.69 in 5/5 seeds**, the largest
effect measured anywhere in this project. Hartmann regret separates too (1.01,
4/5). The axis was never null; the lever had not been moved far enough.

TWO CONSEQUENCES, and I am recording both rather than quietly editing the
prediction:

1. **The stated grounds are gone.** Leaving "tightness is a null axis" in a
   locked protocol that a reader will consult after h125 would misrepresent what
   was known at launch. Struck.

2. **The direction this protocol tests is the one h125 shows is harmful.**
   Widening the ROI is exactly the 0.10 -> 0.495 move that costs 9 regret points.
   A faithful `beta_t` grows, which widens the acceptance set, so the prediction
   for h123 should now be that ROI-WIDEN is WORSE than constant q=0.10 — not
   merely null.

REVISED PREDICTION (P1'), replacing P1. ROI-WIDEN performs WORSE than ROI-Q10 on
final_regret, in >= 4/5 seeds with |mean|/sd >= 1.0. This is a direction change
made BEFORE any h123 run, on the basis of a measurement external to h123, and
the original P1 stays visible above so the change is auditable.

P2 and P3 stand. If ROI-WIDEN beats ROI-Q10, that is a result against both the
original and revised predictions and must be reported as such.

## Does h123 still deserve compute?

Yes, and for a better reason than before. There is now a stated tension:
GP-UCB's theory says `beta_t` GROWS, which widens this ROI; h125 measures that
widening costs 9 regret points on the one benchmark where the ROI works. Both
can be true — the theory governs the validity of a confidence bound, not the
usefulness of the induced acceptance set as a filter on a teacher's training
distribution. h123 puts a number on the size of that gap, which is the honest
thing to report about a paper whose notation we did not follow.

What h123 is NOT: a test of whether the paper is wrong. It tests one linear
q-ramp as a behavioural analogue of `beta_t`, which section 7 already flags.

---

## AMENDMENT 2 (2026-08-28) — the revised direction was ALSO unfounded. Withdrawn.
## Filed before launch; h123 still has no result files.

Amendment 1 replaced the original null with P1' ("ROI-WIDEN performs WORSE than
ROI-Q10"), justified by h125. **That inference does not hold**, and the peer
session identified why:

**h125 compared two CONSTANT tightness settings** — realized q=0.100 against
q=0.493, each fixed for a whole run. It establishes that being *uniformly* wider
is worse. **A schedule is neither of those things.** The contrast h125 supports
is tight-vs-wide; the contrast h123 tests is tight-then-wide. Nothing in h125
speaks to the second.

So Amendment 1 committed the same class of error as h128: taking a number off
our own record and applying it past the frame it was measured in. Amendment 1's
P1' is **WITHDRAWN**. The original P1 (a null) was withdrawn earlier and is not
reinstated — its grounds were refuted too.

### REVISED PREDICTION (P1''): NO DIRECTION REGISTERED.

h123 is launched with the direction genuinely open. I have no valid ground for
either sign:
  - h125 licenses nothing about schedules (above).
  - The theory (GP-UCB's beta_t grows) argues for widening but governs a
    confidence bound's validity, not an acceptance set's usefulness as a filter.
  - The peer's erosion analysis (below) argues the other way but is post-hoc.

Registering no direction is the honest state. The gate remains: separation
requires >= 4/5 seeds and |mean|/sd >= 1.0, in whichever direction it falls.

### The peer's prediction, recorded as THEIRS and not adopted

They predict a widening schedule **beats** constant q=0.10 on Borehole by
recovering part of an eroding advantage, and does **not** beat it before cost 100.

Their supporting measurement, which I verified independently and reproduce
exactly (Borehole, seeds 42-46, paired, rel% of optimum):

  ROI advantage @cost 100:  -6.75, -10.59, -4.11, -8.36, -9.25
  ROI advantage @cost 200:  -3.78,  -2.49, -1.56, -5.71, -7.57
  erosion: mean **+3.588**, sd 2.563, effect **1.40**, eroded in **5/5** seeds
  and between cost 100 and 200 the control's regret falls **5.193%** while the
  ROI's falls **1.605%** — the restricted method stalls while the control keeps
  improving.

**I am deliberately NOT adopting this as my prediction.** They labelled it
exploratory and post-hoc — it was run after seeing the curve that suggested it —
and importing a post-hoc hypothesis into my locked protocol would launder it into
a confirmatory one. It is recorded here so that h123's result adjudicates it as
an independent prediction, with its provenance visible.

### DESIGN CHANGE ADOPTED (this is measurement, not direction)

A schedule mixes regimes, so a single read point cannot characterise it.
**h123 will report at post-init cost 25, 50, 100 and 200**, not 200 alone. This
is adopted from the peer's point because it is about how to measure, not about
what to expect, and it would be needed under any hypothesis. It also lets the
result speak to their before/after-cost-100 prediction rather than averaging
across it.

---

## AMENDMENT 3 (2026-08-28) — manipulation check. Filed before launch.

Adopted from the peer's h132 design; I should have had it and did not.

**GATE M1 (blocking, and it can void the headline regardless of direction).**
ROI-WIDEN ramps q from 0.05 to 0.50 by cost progress, so at low cost it is
TIGHTER than ROI-Q10's constant 0.10, and at high cost looser. The check:

  - At post-init cost 25, ROI-WIDEN's realized `roi_summary.accept_frac`
    trajectory must be BELOW ROI-Q10's, and its regret must not differ from
    ROI-Q10's by more than the seed-to-seed spread. Early on, a tight ramp and a
    constant q=0.10 should behave similarly, with the ramp marginally tighter.
  - Realized acceptance must track the configured ramp. Given the quantile
    calibration hits its target to three decimals at q=0.05, 0.10 and 0.493
    (measured), a ramp that does not move acceptance is not a ramp.

**If M1 fails, the implementation is wrong and h123's comparison is void
regardless of which way it fell.** Recorded now so that a favourable result
cannot be kept while an unfavourable one is blamed on implementation.

This is the check that would have caught ROI-ANN's bug on its first run: that
arm's realized acceptance never moved from ~0.49 across a configured 0.50->0.05
anneal, and nobody compared configured against realized until today.

---

## AMENDMENT 4 (2026-08-28) — M1 replaced with a SHARP check. Filed before launch.

The peer proposed strengthening M1 to **byte-identity** between ROI-WIDEN and
ROI-Q10 early in the run, and asked which of byte-identity or indistinguishability
should be the registered gate, since only one can be.

**Neither, and for a structural reason.** Byte-identity is unavailable to a RAMP
at any start value. Their h132 is a STEP — q=0.10 until cost 100, then off — so
its arms are genuinely identical before the switch and byte-identity is exactly
right there. A ramp's q differs from the first iteration onward, so the two arms
diverge immediately by construction, whether the ramp starts at 0.05 or at 0.10.
This is not a design choice I can make differently; it follows from ramp vs step.

But their underlying objection to M1 as written was correct: **"regret must not
differ by more than the seed-to-seed spread" is a bar with a hole in it** — the
same defect as h131's indeterminate P1 and the h113 gate. Vague criteria decide
nothing, and the hole only becomes visible when a result lands in it.

### M1 REPLACED. New gate, sharp and one-sided:

ROI-WIDEN's **run-mean realized `roi_summary.accept_frac` must fall in
[0.25, 0.30]**.

Grounds: the ramp is q_t = 0.05 -> 0.50, linear in cost progress, so its
cost-weighted mean is (0.05 + 0.50)/2 = **0.275**. The quantile calibration is
measured to hit its target to three decimals at q = 0.05, 0.10 and 0.493, so
realized acceptance is a precise instrument here, not a noisy one. A +/- 0.025
window is wide relative to that precision and narrow relative to the failure
modes.

**This gate would have caught the ROI-ANN bug on its first run.** That arm was
configured 0.50 -> 0.05 — cost-weighted mean also 0.275 — and its realized
accept_frac is **0.4934**, which is its START value, not its mean. The schedule
never advanced. Checking configured-mean against realized-mean is exactly the
comparison nobody made for four months.

**If M1 fails, the schedule did not run and h123's comparison is void regardless
of direction.** Retained from Amendment 3, and now with a criterion that can
actually fail.

### Note on what M1 cannot check

`roi_summary` stores run-level aggregates only (`n_records`, `accept_frac`,
`beta_sqrt`, `n_distinct`, `n_draws`) — there is no stored acceptance
trajectory. So M1 tests the ramp's MEAN, not its SHAPE. A schedule that jumped
straight to 0.275 and stayed there would pass. That is a real limitation and it
is stated rather than papered over; catching shape would require logging the
per-iteration series, which is a code change I am not making mid-experiment.

---

## AMENDMENT 5 (2026-08-28) — comparator. Filed before launch, and deliberately
## NOT changing the registered primary.

The peer observes that **ROI-FIX2 is the best constant arm on record**, and I
verified it: vs the ROI-OFF control on Borehole, seeds 42-46, paired, rel% of
optimum @`cost_curve` 200 —

  ROI-Q10   -4.224%, sd 2.433, effect 1.74, better 5/5
  ROI-FIX2  -4.814%, sd 1.030, **effect 4.67**, better 5/5
  ROI-ANN   -1.311%, sd 2.440, effect 0.54, better 3/5

FIX2 carries the largest effect size in this project. But head-to-head it is
**tied** with Q10: -0.589%, sd 1.707, effect 0.35, better in 3/5, per-seed
-0.37/-1.35/-3.10/+0.80/+1.07. Its 4.67 comes from an sd 2.4x smaller, not from
a larger benefit. Both facts are recorded; neither alone is the story.

They suggest the informative contrast for an explicit `beta_t` is h123 vs FIX2
rather than vs Q10, since beating the incumbent best is a stronger result.

**DECISION: the registered primary comparator stays ROI-Q10. FIX2 is added as a
SECONDARY, reported alongside.**

Reasons, in order of weight:

1. **Changing a locked primary comparator after seeing which arm performs best
   is result-shaped**, however good the argument. The whole point of registering
   it was to fix it before the numbers were in view. That FIX2's superiority was
   measured by someone else does not make it external to the decision.
2. The two arms are **tied head-to-head**, so "the incumbent best" is a
   description of an effect size, not of a benefit. Swapping to chase a 4.67
   would be swapping to chase a smaller variance.
3. **FIX2 is not a constant-tightness arm at all.** Its `roi_beta_sqrt=2.0` is
   fixed; its acceptance is a run-mean OUTCOME of 0.2141 with a per-seed range of
   0.162-0.265 (measured, h125). A fixed beta gives a time-varying acceptance
   rate, because the accepted set depends on posterior width. So comparing a
   quantile ramp against FIX2 contrasts two things that differ in BOTH schedule
   and parameterisation — a worse-controlled contrast than against Q10, whose
   acceptance is pinned to 0.100 on every seed to three decimals.

Adding FIX2 as a secondary costs nothing (its runs exist) and lets h123 speak to
their point without the primary being chosen after the fact.

### The deeper observation, which I am NOT resolving here

If a fixed beta yields time-varying acceptance, **ROI-FIX2 may already be an
implicit schedule** — and it is the best arm on record. Its direction is
UNMEASURED: `roi_summary` stores run-level aggregates only, so nothing
distinguishes a rising from a falling acceptance trajectory. Analytically a fixed
beta should narrow as posterior width shrinks, which predicts a stall FIX2 does
not show (late-recovery 24.34% against Q10's 10.51%). Either that analysis is too
simple or sigma does not shrink monotonically here. **Not guessed at; it needs
per-iteration logging, below.**
