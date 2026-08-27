# H85 — teacher acquisition refinement (the falsification branch of H84)

LOCKED BEFORE ANY RUN, and before H84 completed. This is the experiment H84's
pre-registered falsification clause points to, written while H84 was still in
flight so it cannot be a reaction to H84's outcome.

## Why this, and why not the ROI

Three measurements taken during H84's run, all recorded in findings.md before
any H84 result landed, point away from the candidate DISTRIBUTION and at the
teacher's proposal MECHANISM:

1. At a matched model state on Hartmann, the ROI raises mean teacher-action
   score by **+0.010**; acquisition refinement raises it by **+0.046**. Not
   additive. The gap to close is 0.336 -> 0.747.
2. Tightening the ROI costs REACH: the single best teacher action falls
   0.974 -> 0.717 at q=0.10, and the closest-ever approach to x* degrades
   0.022 -> 0.110 at q=0.02.
3. On Borehole, MF-DRO lands within 5% of a domain bound on **8.86%** of
   coordinates against a **10% uniform null** -- boundary-INDIFFERENT, not
   boundary-averse. It saturates clamp(0,1) on 2.02% of coordinates, so the head
   is NOT representationally blocked. MF-MES reaches 34.68%, and its 25.58%
   exact-bound rate is the signature of running bounded L-BFGS-B.

MECHANISM. MF-DRO's rollout teacher takes a FLAT ARGMAX over uniform random
candidates. A uniform draw in 8-D is essentially never at a bound in several
sensitive dimensions AT ONCE, so the teacher can almost never propose Borehole's
optimum (whose four sensitive dims are all at bounds), and the DT never sees
such a training target. The ROI relocates a uniform draw; it does not make the
draw able to reach a corner. Local refinement can walk there.

## The change

`teacher_refine_samples = 100` (currently 0): after the teacher's broad argmax,
draw 100 points around the winner at `teacher_refine_noise = 0.05` of the span,
clip to bounds, and re-score the UNION. Already implemented and gated; the
latent bug in that branch (it concatenated into `roi_candidates`, a variable
defined OUTSIDE the rollout loop, permanently growing the shared pool) was fixed
on 2026-08-27 before this experiment.

DOES THIS VIOLATE THE STANDING CONSTRAINT? No, and the distinction matters.
`use_candidate_scoring=True` is barred because it puts a pool+argmax at
INFERENCE, replacing the contribution. This changes only how TRAINING DATA is
generated. At inference the DT still emits `x = action_head(h).clamp(0,1)`
directly, with no search and unchanged inference cost.

OBJECTION TO STATE PLAINLY, not to bury: if this works, a reader can argue the
gain comes from a better teacher that the DT imitates, not from the DT. The
honest answer is that a better rollout teacher producing a better policy IS the
DRO frame working as designed -- but the claim must then be about the
teacher-plus-policy pipeline, not about the policy alone. If H85 succeeds, the
write-up says that.

## Design

| | |
|---|---|
| arms | REFINE-0 (control, = h83 MF-DRO), REFINE-100 |
| benchmarks | Borehole_8D (boundary optimum), Hartmann_6D (interior optimum) |
| seeds | 42-46 |
| budget | 200 post-init; M=3, pool 600, use_roi=False |

Arm REFINE-0 reuses h83's MF-DRO runs under the same bit-identity gate H84 uses,
with 2 live re-runs per benchmark as a reproduction control that VOIDS the reuse
if it fails.

## Metrics (frozen)

- PRIMARY: mean HF query quality, `(y - best_init)/(y_opt - best_init)`.
- MECHANISM: fraction of post-init HF coordinates within 5% of a domain bound
  (uniform null = 10%).
- SECONDARY: final relative regret at cost 200.

## Predictions (pre-registered, independent)

- **P1 (PRIMARY, the disproportionality prediction).** Refinement improves mean
  HF query score MORE on Borehole than on Hartmann. This is the prediction that
  distinguishes the boundary mechanism from a generic "refinement helps
  everywhere" effect, and it is the reason both benchmarks are run.
- **P2 (MECHANISM).** On Borehole, REFINE-100's near-bound coordinate fraction
  rises above the 10% uniform null (REFINE-0 sits at 8.86%). If refinement helps
  regret WITHOUT moving this number, the boundary mechanism is wrong even if the
  method improves.
- **P3 (SECONDARY).** REFINE-100 lowers final relative regret on Borehole on
  >= 4/5 seeds.
- **P4.** Wall-clock cost of refinement is under 2x REFINE-0. Recorded because a
  fix that is 5x slower is not a usable default, whatever it does to regret.

No p-values at n=5. Every run reported including failures.

## Falsifier

If P1 and P2 both fail, the teacher's proposal mechanism is NOT the cause of
MF-DRO's budget waste, and the remaining candidates are the RTG/reward signal
and the state representation -- neither of which any measurement this session
has touched.
