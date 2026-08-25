# H47 — is the DT useful, and does the head choice change the answer?

## The question

"Does MF-DRO's Decision Transformer contribute anything?" cannot be answered by
comparing MF-DRO to MF-DRO. It needs the no-DT arm at the same settings, and it
needs a per-iteration measure of what the DT did differently from the teacher it
imitates. h47 supplies both.

## Why existing runs cannot be reused

- **h17** (scoring head, 0.4007) and **h31** (teacher, 0.4781) ran to a **cost
  budget of 200**, not to a fixed iteration count, with `n_HF=36 / n_LF=60`.
- **h42** (regression head, 0/3 frozen) ran 50 iterations but with a different
  initial design and logged no teacher comparison.

Mixing an iteration-capped run with a cost-capped one is exactly the error that
produced the retracted `cost_curve` checkpoints earlier in this project. All
three arms here are therefore run **fresh, at identical settings**.

## Design

Hartmann 6D, **50 iterations**, seeds **42, 43, 44**, `initial_hf=6`,
`initial_lf=45`, `real_hf_warmup=2`, `rollout_reward="mes_entropy"`. Only the
proposal mechanism varies:

| arm | proposal | pool + argmax |
|---|---|---|
| **S** scoring head | `argmax_k <w(h), cf_k>` over 200 fresh candidates | yes |
| **R** regression head | `x = action_head(h).clamp(0,1)` | no |
| **T** teacher, no DT | `argmax_{k,l} MES(cf_k,l)/c(l)` | yes |

9 jobs, 4 concurrent workers, 1 thread each (H45's 10 workers are still live;
4 + 10 + parent = 15, at the compute cap).

## The DT-responsibility measure

At every iteration, **after** the proposal is made and **before** it is
evaluated, draw a fresh diagnostic pool of 200 uniform points, append the
proposed `x` as row 201, and score all 201 in a **single**
`compute_joint_mf_mes` call so every row shares the same Thompson `y*` draws
(two calls would inject Monte-Carlo noise into the comparison). Then record

    delta_acq  =  max_l MES(x_DT, l)/c(l)  -  max_{k<=200, l} MES(cand_k, l)/c(l)

- `delta_acq < 0` — the DT picked a point a fresh 200-point myopic search would
  have beaten. Magnitude = how much acquisition value it left on the table.
- `delta_acq > 0` — the DT found a point better than that search.

Also recorded per iteration: `||x_DT - x_teacher||` normalised by the domain
span, the chosen fidelity, whether it matches the teacher's, simple regret,
inference regret, and cumulative cost.

**Arm T calibrates the measure.** The teacher's own pool and the diagnostic pool
are different 200-point draws, so arm T's `delta_acq` is the noise floor of this
statistic, not zero. Arms S and R are only interpretable against that floor.

## Locked predictions

1. **PRIMARY (usefulness)**: DT is useful iff an arm's mean final HF simple
   regret is **below arm T's**, paired across the 3 shared seeds, on **>= 2/3**
   seeds. With n = 3 no p-value is reportable; direction and per-seed win counts
   only.
2. **HEAD**: S and R are compared to each other on the same criterion. h45's
   locked prediction (arm B >= arm C at cost-budget settings) predicts **R does
   not beat T** here either.
3. **MECHANISM**: if an arm beats T while its mean `delta_acq` is **negative
   beyond arm T's noise floor**, that is evidence the DT wins by *deviating*
   from myopic MF-MES rather than by imitating it better — the only form of
   "usefulness" that would support the non-myopia claim the method was built on.
4. **NULL**: `delta_acq` for S and R lies inside arm T's floor and regret
   matches T. Then the DT is a distillation of its teacher and adds nothing,
   under both heads.

## What this cannot settle

n = 3. This sizes an effect and fixes a direction; it does not establish
significance. The h17-vs-h31 contrast needed 82 seeds for 80% power.
