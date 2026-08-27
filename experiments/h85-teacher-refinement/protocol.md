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

## Amendment 1 — third arm HF-FLOOR (added before ANY h85 run)

Human-proposed: force a high-fidelity query every n real queries. Added as a
third arm rather than a separate experiment so it is measured against the same
control and the same seeds.

`real_hf_every = 4` guarantees at least one HF query in every window of 4 real
queries. This mechanism did NOT previously exist: `minimum_hf_fraction` operates
on `actions_ell`/`tau` inside ROLLOUTS "to ensure training diversity"
(mf_dro.py:1589) and has no effect on real queries. Implemented at the same site
as the existing `real_hf_warmup` cold-start override, disabled at 0 by default,
so no existing configuration changes (the guard is `if hf_every > 1`).

WHAT THE EVIDENCE SAYS, stated before running so the bar is honest. Across
h83's Hartmann seeds, HF COUNT DOES NOT PREDICT OUTCOME:

  seed   LF%   HF n   mean score   max score   rel.regret
    42   94%      8        0.684       0.823       16.41%
    43   25%     24        0.808       0.993        0.67%
    44   90%     12       -0.942       0.648        7.98%
    45   96%      6        0.560       0.933        5.28%
    46   94%      8        0.571       0.892        7.42%

Six HF queries (seed 45) reach 0.933 while twelve (seed 44) reach 0.648, and
seed 45 beats seed 42 on regret with fewer HF evaluations. Across benchmarks the
same: on Borehole MF-DRO already makes MORE HF queries than MF-MES (94 vs 84)
and loses by 9.4 points; on Currin it makes more (27 vs 11) and wins. What binds
is query QUALITY -- seed 44's average HF query lands 0.942 BELOW its own
starting point.

So HF-FLOOR is registered as a WORST-CASE FLOOR, not a mean improver. Its cost
is real: on Hartmann each forced HF query consumes the budget of 8 LF queries
that would otherwise inform the KO surrogate, so it can degrade the model that
generates proposals.

A NOTE ON A DISCARDED ANALYSIS. A bootstrap of E[max score] against the number
of HF draws was run first and is NOT relied on here: resampling with replacement
from observed values can never exceed the observed maximum, so E[max]
asymptotes to it by construction and the test cannot detect a benefit from
additional draws finding NEW better points. It structurally understates the case
for this arm. The per-seed table above is assumption-free and is what the
predictions rest on.

### Predictions for HF-FLOOR (independent of P1-P4)

- **P5 (VARIANCE, primary for this arm).** HF-FLOOR reduces the ACROSS-SEED
  SPREAD of Hartmann relative regret (h83 control: 0.67%-16.41%, sd 5.79). This
  is the failure mode the floor targets and the only thing the evidence supports
  it improving.
- **P6 (MEAN, registered as NOT expected).** HF-FLOOR does NOT improve mean
  relative regret on Hartmann by >= 1 point. Registered as a negative prediction
  because HF count does not predict outcome across seeds; if it DOES improve the
  mean, the fidelity-allocation story is more important than this project's
  measurements have indicated and the diagnosis needs revisiting.
- **P7.** On Borehole, HF-FLOOR changes little either way -- it already runs at
  11.7% LF, so a 1-in-4 floor is close to non-binding there.

## Amendment 2 — any positive result here requires fresh-seed confirmation BEFORE announcement

Added before launch, in light of h87.

h84 produced a Hartmann result that met h83's own bar at 4/5 seeds, survived
four attached caveats, was announced in findings.md, research-state.yaml, the
research log and a published report -- and then failed to replicate at fresh
seeds (2/5). The caveats were all correct and none of them prevented the
announcement.

THEREFORE, for h85: **no arm's result is announced as a finding until it has
been re-tested at seeds never used for it, with the configuration fixed in
advance.** Positive results from these 24 runs are provisional by default and
are to be written up as "pending confirmation", not as findings. The h87
template applies -- one arm, fixed config, fresh seeds, analysis script
committed before the treatment arm finishes.

This does not change P1-P7. It changes what may be said about them.

Corollary on seed count: h87 showed that a paired sd of 0.45 on one seed set can
become 7.45 on another, because the two methods fail on DIFFERENT seeds. n=5 is
not enough to characterise a paired difference in this project. Treat every
n=5 margin here as a hypothesis.

## Review note (written by a CONCURRENT session while h85 was already running)

Two sessions have been working this repo in parallel. This note is a review of
the launched configuration, not a change to it -- the launcher is running and
killing it would kill its 15 workers.

FINDING: **the reproduction control is queued LAST, violating Lesson 21.**
`run_all.py` builds 20 treatment jobs and then appends the 4 REFINE-0 control
jobs, so with `max_workers=15` the controls dispatch only after most treatment
runs finish -- roughly two hours in, on Borehole timings. Lesson 21 was recorded
in findings.md precisely because h84 did this: a control that can VOID the
experiment must run FIRST, since its failure invalidates every arm-relative
number. h87's launcher got this right (MF-MES first, explicitly commented);
h85's launcher predates the lesson and was not updated.

NOT FIXED, deliberately: restarting the launcher would kill 15 in-flight
workers, and the cost of a late control here is bounded -- Amendment 2 already
forbids announcing any h85 result before fresh-seed confirmation, so a control
that lands late cannot cause a premature announcement. The defect is recorded
rather than repaired.

ACTION FOR WHOEVER ANALYSES h85: do not report any arm-vs-REFINE-0 number until
all four control runs have completed and reproduced h83 bit-identically. The
analysis must print "reuse unverified" until then, as h84's did.

CONCURRENCY HAZARD, for the record: this session and another both wrote the h87
withdrawal to findings.md ~50 seconds apart, producing a duplicate that had to
be consolidated. Two sessions launching experiments independently can also
exceed the 15-worker cap. Compute was checked before writing this note: 15
workers, all h85, at the cap and not over.

### Live verification of the HF-FLOOR mechanism (read-only, concurrent session)

The pre-launch smoke test did NOT exercise the floor: at budget 40 the run was
all-HF anyway, so "longest LF run = 0" proved only that the code did not crash.
Checked against the live checkpoints instead, where `real_hf_every=4` guarantees
no window of 4 real queries is all-LF, i.e. the longest run of consecutive LF
queries must never exceed 3:

  arm         bench         seeds   longest LF run   verdict
  HF-FLOOR    Borehole_8D   42-46        1 (all)     OK, 0 violations

The mechanism works. But note what is NOT yet tested:

  1. THE FLOOR IS BARELY BINDING ON BOREHOLE. Its LF fraction there is 5.9-11.8%,
     and MF-DRO's Borehole baseline was already ~12% LF -- the policy chooses
     high fidelity there on its own. This is consistent with P7, registered
     before launch: "On Borehole, HF-FLOOR changes little either way."
  2. THE ARM THAT MATTERS HAS NOT STARTED. HF-FLOOR on Hartmann is still queued.
     Hartmann is where MF-DRO runs 80-96% LF and takes only 6-12 real HF
     evaluations -- the fidelity-head collapse the floor exists to bound. Until
     those five runs land, the floor is verified as CORRECT but not as USEFUL.

An incidental illustration from the same check: REFINE-100 on Hartmann seed 45
is currently at 80% LF with a run of EIGHT consecutive low-fidelity queries.
That is exactly the failure mode HF-FLOOR targets, observed live in a different
arm.

## Mechanism check (in-flight, no performance claim)

Verifying `real_hf_every=4` does what it says, from live checkpoints:

  arm          bench          post-init q    HF%   longest LF run
  HF-FLOOR     Borehole_8D             36   94.4%        1
  REFINE-100   Borehole_8D             21   92.4%        2
  REFINE-100   Hartmann_6D             21   49.5%       19

1. **The floor binds correctly.** HF-FLOOR's longest consecutive-LF run is 1,
   inside its guarantee of <= 3. No violations.
2. **It is NON-BINDING on Borehole**, which already runs at ~94% high fidelity.
   P7 registered exactly this ("close to non-binding there") and it holds. The
   Borehole HF-FLOOR arm is therefore expected to be near-identical to the
   control, and should be read as a null-by-construction rather than as
   evidence the floor does nothing.
3. **Hartmann is where it can bind.** The un-floored trace there shows a
   **19-query consecutive low-fidelity run** -- the fidelity-head collapse the
   floor exists to bound. Those runs had not started at the time of this check.

NO PERFORMANCE CLAIM IS MADE HERE. This records only that the intervention
fires as specified and where it can possibly matter, so that a null on Borehole
is not later mistaken for a null on the mechanism.

## In-flight cost measurement (provisional; P4 is on the boundary)

Wall-clock minutes per post-init query, against h83's MF-DRO runs as the
REFINE-0 control:

  arm          bench          min/query   vs control
  REFINE-0     Hartmann_6D        0.783       1.00x
  REFINE-0     Borehole_8D        0.773       1.00x
  REFINE-100   Hartmann_6D        1.562       1.99x
  REFINE-100   Borehole_8D        1.565       2.02x
  HF-FLOOR     Hartmann_6D        0.938       1.20x
  HF-FLOOR     Borehole_8D        0.935       1.21x

TWO READINGS.

1. **Refinement is firing.** A ~2x cost is the expected signature of 100 extra
   acquisition evaluations per rollout step. The arm is not silently inert --
   worth confirming, since the teacher_refine_samples branch had a latent bug
   (it grew the shared candidate pool across rollout steps) that was only fixed
   on 2026-08-27 and had never been exercised at teacher_refine_samples > 0.
2. **P4 is on the boundary and likely to FAIL.** It required wall-clock under
   2x. Borehole already reads 2.02x. Whatever the regret result, a fix that
   doubles wall-clock is a different proposition from one that is free, and P4
   exists precisely to keep that visible.

PROVISIONAL: these are in-flight rates from partial runs. MF-DRO slows as data
accumulates, so the final ratio could move in either direction depending on
whether refinement's overhead scales with the data or stays constant. No bar is
being called; the verdict comes from analyse.py at 5/5.

### P4 (wall-clock cost) — in-flight measurement, concurrent session

P4 registered that refinement must cost under 2x the no-refinement control.
First data, with h83's MF-DRO runs as the REFINE-0 control:

  arm / benchmark                 wall     control   ratio
  Hartmann REFINE-100 s43       46.4m      25.1m     1.85x   COMPLETED RUN
  Borehole REFINE-100 (proj)      165m      82.4m    2.00x
  Hartmann REFINE-100 (proj)      154m      94.0m    1.64x
  Borehole HF-FLOOR   (proj)       97m      82.4m    1.18x
  Hartmann HF-FLOOR   (proj)       49m      94.0m    0.52x

TWO OBSERVATIONS.

1. REFINEMENT SITS RIGHT ON THE BAR, and the projections probably UNDERSTATE it.
   These are linear extrapolations (elapsed / fraction-of-budget-spent), but
   MF-DRO's progress is SUBLINEAR -- GP fits get more expensive as data
   accumulates, which is why every ETA in this project has run optimistic.
   Borehole REFINE-100 projects to exactly 2.00x and should be expected to
   finish above it. P4 is therefore likely to FAIL on Borehole. Recorded before
   the runs land.

2. THE HF FLOOR IS FREE, AND ON HARTMANN IT IS FASTER THAN THE CONTROL (0.52x).
   That is not an anomaly: forcing high-fidelity queries makes a run consume its
   cost budget faster (c_H=8 vs c_L=1 on Hartmann), so the run TERMINATES IN
   FEWER ITERATIONS. Half the wall-clock for the same cost budget.

   This matters for how the two interventions compare. If HF-FLOOR and
   REFINE-100 produce similar regret improvements, the floor wins outright on
   cost -- it is 3-4x cheaper in wall-clock than refinement on both benchmarks.
   And it reframes P6, which registered that HF-FLOOR is NOT expected to improve
   the mean: an intervention that is free and bounds a known failure mode
   (fidelity-head collapse at 94-96% LF) is worth having even at zero mean
   benefit, provided it does not HURT.

STATUS: EXPLORATORY, one completed run plus linear projections that are known to
be optimistic. P4's verdict waits on completed arms.
