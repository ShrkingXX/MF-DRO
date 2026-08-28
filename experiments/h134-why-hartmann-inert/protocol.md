# h134 — Does the DT actually LEARN on Hartmann? If not, the ROI has nothing to shape.

STATUS: LOCKED before computing. ZERO NEW COMPUTE.
TYPE: CONFIRMATORY.
STATISTIC / READ POINT: `L_loc_per_iter`, mean over the first third vs the last
third of each run's iterations, expressed as **fractional decline**
`(first - last)/first` per seed. Borehole and Hartmann, seeds 42-46, control arm
(ROI-OFF) and ROI-Q10 reported separately.

## Why

The primary question states the causal path: the DT regression head emits x and
is trained on rollout-teacher actions drawn from `roi_candidates`, **so the ROI
acts by shaping a training distribution.** An intervention on a training
distribution can only matter if training does something.

Four mechanisms x four benchmarks have exactly one positive cell each, always
Borehole. h131 could not decide whether Hartmann differs in kind or is merely
truncated (P1 indeterminate). This asks a different and more basic question that
neither has: **does the head learn on Hartmann at all?**

Note a fact that cuts against the obvious guess: Hartmann runs ~120 post-init
iterations, similar to Borehole's ~107 — it is starved of HF queries (11.2 vs
93.4), not of iterations. So "the DT barely trains" is NOT automatic and has to
be measured.

## Predictions (locked)

**P1 (PRIMARY).** Borehole's control shows a substantial fractional decline in
`L_loc` (>= 0.40) while Hartmann's is markedly smaller, with the between-benchmark
difference separable at effect >= 1.0 and >= 4/5 seeds.

**P2.** If P1 holds, the ROI's inertness on Hartmann has a sufficient explanation
that does not require any claim about the region's *placement*: there is little
learning for a training-distribution intervention to redirect.

**Falsified if** Hartmann's decline matches Borehole's. That would mean the head
learns comparably well on both, the ROI still does nothing there, and the failure
is about *where* the region sits rather than *whether* training responds — which
would send the question back to h100's containment framing with its defective
instrument.

## Confounds, stated before looking

1. **`L_loc` is a loss on a moving target.** A flat loss can mean "no learning" or
   "the target moves as fast as the head follows". These are not distinguishable
   from this statistic alone, and I will not claim the first if the data shows the
   flat case.
2. **Scale.** `L_loc` is not comparable across benchmarks in absolute terms, which
   is why the registered statistic is a **fractional** decline within each run.
   The seventh scale error today would be comparing raw losses across benchmarks
   whose objectives differ 100x in range.
3. This is correlational. It can show the signature is absent, not that absence
   causes inertness.

---

## AMENDMENT 1 — P4, replicating the unregistered lead across independent ROI arms

The h134 result carried an EXPLORATORY observation I did not register: on
Hartmann the ROI arm's `L_loc` degrades less than the control's (-0.579 vs
-1.003). If real it would be **the first non-null Hartmann cell in this project**,
so it needs a test rather than a mention.

Three further Hartmann ROI arms exist at seeds 42-46 and were not used above:
h84 `ROI-FIX2`, h84 `ROI-ANN`, and h111 `ROI-Q05`. They span realized acceptance
from ~0.05 to ~0.49 and were run for unrelated purposes.

**P4 (locked, registered before looking at any of the three).** All three arms
show *less* `L_loc` degradation than the ROI-OFF control on Hartmann — 3 of 3
arms in the same direction — and the pooled ROI-vs-control contrast across all
four ROI arms is separable at effect >= 0.5 with >= 4/5 seeds.

**Falsified if** two or more of the three go the other way. With four arms
spanning a 10x range of acceptance rates, a real effect of the ROI should not be
confined to the one arm that suggested it.

**Stated before looking:** this is a replication of a post-hoc observation on the
*same seeds*, so it is not independent evidence — the seeds are shared and only
the arms differ. It can show the observation is not an artefact of one arm; it
cannot show it is not an artefact of these five seeds. A fresh-seed test would be
needed for that and is not run here.
