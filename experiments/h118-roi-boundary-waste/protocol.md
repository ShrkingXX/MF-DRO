# H118 — Does the ROI reduce the HF budget MF-DRO wastes off the boundary?

STATUS: LOCKED before any statistic was computed.
TYPE: CONFIRMATORY (prediction below, stated before computation).
COMPUTE: zero new runs. Re-analysis of h90-borehole-confirm.
DATA: h90 ONLY — Borehole_8D, seeds 47-51, arms NO-ROI / ROI-Q10 / REFINE-100,
      one experiment, matched seeds. No cross-experiment pairing.

## Why this is the primary question, not a side quest

h116 established (EXPLORATORY, h83 seeds 42-46) that MF-DRO spends ~8.9% of its
HF budget off Borehole's boundary optimum in dim 0 (r_w, 86% of variance), on
queries that could never become the incumbent. That is a concrete, priced
instance of "wasting HF budget on low-value regions" — the thing the ROI is
supposed to prevent.

MECHANISM THE ROI COULD ACT THROUGH. The teacher's candidates come from
`_draw_raw()` (mf_dro.py) — `torch.rand`, i.i.d. uniform over the FULL box —
which are then filtered by the ROI, survivors accumulated until `_N_POOL`
(default `n_roi_candidates=600`) are collected, over at most 40 draws. So at
acceptance rate q, those 600 points occupy the ROI's volume instead of the
box's. **The ROI is a resolution amplifier for the teacher: effective sampling
density inside it scales as 1/q.** The DT's regression head is trained on those
teacher actions, so pool resolution propagates to the policy's proposals.

If MF-DRO's boundary failure is a pool-resolution problem, tightening the ROI
should let the teacher place candidates nearer the boundary optimum and reduce
the wasted fraction. If the ROI does not act through resolution, it should not.

## Measure (identical to h116; no changes)

Per arm x seed, over non-init HF queries (`fid==1 and not is_init`):
  z0 = (x[0] - domain_min[0]) / (domain_max[0] - domain_min[0])
  waste = fraction of those queries with z0 < 0.9

The 0.9 cut was fixed in h116 from the h83 distributions and is carried over
UNCHANGED. Declared now, not post hoc: the same statistic is also reported at
a 0.95 cut as a sensitivity check. Both are reported whatever they show.
Amendment-2 floor of 15 non-init HF queries applies; exclusions are reported.

## Prediction (locked)

1. Paired per seed, waste(ROI-Q10) < waste(NO-ROI) in at least 4 of 5 seeds.
2. |mean paired difference| / sd >= 1.0.
3. Direction: the ROI REDUCES wasted budget.

Failing 1 or 2 refutes the resolution-amplifier account for this metric.
A result with waste(ROI-Q10) > waste(NO-ROI) refutes it outright and would mean
the ROI's known Borehole regret benefit (3.5-4.2 pts, sd 0.37) operates through
some channel other than boundary resolution.

REFINE-100 is reported alongside but does NOT gate: it is a different
intervention (teacher refinement) and was not part of forming this hypothesis.

## Secondary, EXPLORATORY and labelled as such

Across all arms x seeds available here, the association between per-run waste
and that run's final simple regret. This is 15 points at most, from 5 seeds and
3 non-independent arms; it is a description, not an inference, and no gate
attaches to it.

## Limitations

- n=5 seeds. No p-values (project rule).
- One benchmark. Every ROI effect in this project has been Borehole-specific,
  and Borehole is the most anisotropic of the four (PR/d = 0.168). This tests
  the MECHANISM on the one benchmark where the effect exists; it does NOT test
  generality and must not be reported as if it did.
- The h116 waste measure is itself EXPLORATORY and awaiting h117. If h117 fails
  to replicate it, h118 inherits that failure regardless of its own result.
- MF-MES is deliberately ABSENT here. Per the h117 amendment-2 confound, it
  refines with box-constrained L-BFGS-B and lands on active constraints by
  construction, so it is not a fair reference for a boundary metric. All three
  arms here are MF-DRO variants and differ only in the ROI treatment.
