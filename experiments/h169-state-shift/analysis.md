# h169 — killed as designed, and replaced by what killing it revealed

## The arm was a no-op, and it was my error

The probe sampled `_last_batch_tau0_states[:4]`. The batch is laid out
`[member0 × rollouts_per_model, member1 × …]` with `rollouts_per_model=6`, so
the first four entries are **four rollouts of the same ensemble member** and
their τ=0 states are identical by construction. Measured: pairwise distance
**0.0000** across all five probed states.

h169 as launched was therefore a rerun of h168. **Killed at ~60% rather than
allowed to finish and be written up as a state manipulation.** Fixed by striding
`[::rollouts_per_model]`; the corrected states differ by 0.39–1.41.

## What the failure exposed, which is worth more than the arm

With the stride fix, the pairwise distances are:

```
        real   train0   train1   train2   train3
real  0.0000   0.0000   1.0338   0.3934   0.5797
```

**The real inference state is bit-identical to a τ=0 training state.** There is
**no state-distribution shift at τ=0**. And `[STATE-DIAG]` — a line printing in
every log of this project, which I have never followed up — reports
`uniq_tau0_states=3` out of 60 trajectories, in every seed.

The codebase's own docstring (mf_dro.py:1018) already says what follows:

> *"made every trajectory's tau=0 state bit-for-bit identical — the DT could only
> ever learn the conditional mean of that timestep's targets, independent of
> anything real inference later provides."*

That describes a bug since partially fixed (3 unique states, not 1). **The
consequence was never revisited after the partial fix.**

## The mechanism this gives — EXPLORATORY, post-hoc

Inference always queries `timestep=0` (mf_dro.py:3224). At τ=0 the training
states are near-degenerate. So the DT can only emit **the conditional mean of the
teacher's τ=0 action** — and that mean is a property of the teacher, computable
in advance:

| teacher | τ=0 action | its mean | predicted query |
|---|---|---|---|
| MES / UCB / EXPLOIT | acquisition argmax on the current model | that argmax | a specific informative point |
| frozen MES (h153/h161) | first point of a model-selected path | that point | a specific informative point |
| RANDOM-POOL | uniform draw from the pool | **box centre** | box centre |
| ORACLE | `x_start ~ Uniform` | **box centre** | box centre |
| DIVERSE-GOOD | `x_start ~ Uniform` | **box centre** | box centre |

Against the measured centroid distance from the box centre (Borehole):

```
control 0.7604 | UCB 0.7788 | FROZEN 0.7426 | EXPLOIT 0.7375
RANDOM  0.0239 | ORACLE 0.0394 | DIVERSE-GOOD 0.0409
```

**All seven arms match.** Every failing teacher's τ=0 action is an independent
draw whose mean is the box centre; every working teacher's is an acquisition
argmax.

## It also explains h167's P2 failure

h167 predicted ORACLE's collapse would land at the mean of its action
distribution **over all τ** — the midpoint of centre and x*, 0.66 away — and P2
failed because it landed at the centre. The marginal was wrong: **inference only
ever sees τ=0**, where ORACLE's action is `x_start ~ Uniform`, whose mean *is*
the box centre. Right idea, wrong marginal.

## Why this is not yet a sixth explanation to add to the pile

It is **post-hoc**: derived after seeing the seven d(centre) numbers it explains.
Five accounts have already fallen on this front by fitting the available evidence
and outrunning it. A quantitative, non-post-hoc test is registered as h170 and
this will not be reported as the mechanism until that test reports.

Supporting but not decisive: h168 (357 iterations, full length) shows the output
is independent of RTG; the corrected h169 smoke shows it moves 0.005 across
states differing by up to 1.41 (ratio 1.030, 5 iterations, not a verdict).
