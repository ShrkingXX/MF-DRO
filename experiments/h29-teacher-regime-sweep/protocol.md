# H29 — is the teacher's exploitativeness regime-dependent or intrinsic?

## Why

H28 showed the MF-MES teacher selects candidates at the **2.9th percentile** of
`sigma_H`, and that the student's uncertainty aversion is inherited from that.
The consequence depends entirely on whether this is a property of the *regime* or
of the *acquisition*:

- **Regime-dependent** (e.g. only at `c_H/c_L = 8`, or only with few `y*`
  samples): a tuning finding. MF-DRO may be rescuable by changing the teacher's
  operating point.
- **Intrinsic**: cost-normalised MF-MES behaves exploitatively as a demonstrator
  in general, and no student trained on its choices can explore. That is a
  limitation of the method family, not of this implementation.

## Design

Two-factor diagnostic sweep, measuring **teacher behaviour only**:

- cost ratio `c_H/c_L` in {2, 4, 8, 16} (frozen protocol uses 8)
- `y*` Thompson samples `K` in {5, 10, 50} (`compute_joint_mf_mes` default 10)

For each of the 12 cells, on a real 200-trajectory rollout batch, report the mean
percentile of the chosen candidate's `sigma_H` within its own pool, plus the
`mu_H` percentile as a control.

## SCOPE — this is not an evaluation change

`PROTOCOL.md` is untouched. **No regret is measured or reported here**, and no
result from this sweep may be used to alter the frozen evaluation, whose cost
ratio remains 8. This measures a property of the teacher, nothing else. Stating
it explicitly because sweeping a quantity the frozen protocol fixes would
otherwise look like protocol drift.

## Locked predictions

1. **PRIMARY (intrinsic)**: the chosen-`sigma_H` percentile stays below 40% in
   **all 12 cells**. Cost-normalised MF-MES is then exploitative as a
   demonstrator across the swept regime.
2. **ALTERNATIVE (regime-dependent)**: at least one cell exceeds 50%, in which
   case the operating point matters and the paper must say so --- along with the
   caveat that the frozen protocol's cell (`c_H/c_L=8`, `K=10`) is the one that
   was actually run.
3. **CONTROL**: `mu_H` percentile must exceed 50% in every cell, else the
   extraction is broken and nothing is interpretable.

## Compute

12 jobs, `num_workers=12 x threads_per_worker=1` (within the 15 limit).
