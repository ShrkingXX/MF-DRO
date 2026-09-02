# h185 — the DT is a PER-TIMESTEP CONSTANT PREDICTOR. The whole front, quantified.

**EXPLORATORY.** No new runs; re-analysis of `L_loc_per_iter` against
`teacher_action_stats` (both saved on h171/h172 only). Borehole, seeds 42–46.

## The identity being used

If a predictor outputs the conditional mean of its target, its MSE **equals** the
target's variance. `L_loc` is the DT's location MSE; `var_total` is the variance of
the teacher's `actions_x`. Both per-coordinate, both averaged over iterations, both
from the same runs — so the ratio is directly interpretable.

| arm | rollout length | rel% | DT loss | teacher var | **loss / var** | variance explained |
|---|---|---|---|---|---|---|
| TAIL-MES | 8 | 43.94 | 0.0389 | 0.0489 | **0.795** | **20.5%** |
| HEAD-MES | 8 | 16.96 | 0.0782 | 0.0854 | **0.916** | **8.4%** |
| ROLLOUT4 | 4 | 15.14 | 0.0424 | 0.0430 | **0.985** | **1.5%** |
| ROLLOUT2 | 2 | 13.97 | 0.0385 | 0.0375 | **1.026** | **0.0%** |
| ROLLOUT1 | 1 | 13.69 | 0.0350 | 0.0342 | **1.022** | **0.0%** |

**Every arm sits at the best-constant MSE**, across a 2.5× range of teacher
variance. The DT explains between 0% and 20.5% of the variance in its teacher's
actions, and on three of five arms it explains **none**.

## The variance it *does* explain is exactly the τ-structure

The ordering is not noise — it tracks how much timestep-structure each arm's
teacher has:

| rollout length | τ-structure available | variance explained |
|---|---|---|
| 1 (ROLLOUT1) | none — only τ=0 exists | **0.0%** |
| 2 (ROLLOUT2) | minimal | **0.0%** |
| 4 (ROLLOUT4) | some | **1.5%** |
| 8, sharp τ=0 vs τ>0 split (HEAD, TAIL) | maximal | **8.4%, 20.5%** |

A **per-timestep constant** predictor explains exactly the between-τ variance and
none of the within-τ variance. That is the observed pattern, monotone in the amount
of τ-structure, including the two arms where the prediction is 0% by construction.

## This unifies the front

- **The DT is, to within 0–20%, a per-timestep constant predictor.** It does not
  learn a mapping from state to action; it learns one point per timestep.
- **Inference queries τ=0 only**, so what it emits is the teacher's **τ=0 mean**.
  This is why the teacher's *all-τ* average is anti-predictive (h182's HEAD/TAIL
  inversion) and why swapping the teacher's *rule* moves nothing (h180) — different
  rules share nearly the same τ=0 mean.
- **Performance is set entirely by WHERE that constant lands**, not by how much the
  DT learns. TAIL explains the **most** variance (20.5%) and fails **worst** (43.94);
  ROLLOUT1 explains **none** and does **best** (13.69). Learning more does not help;
  landing further from the box centre does.
- **"Fits better while performing worse"** — recorded as unexplained for weeks — is
  located: in TAIL the loss falls 20% over the run (0.0480 → 0.0384) while the
  queries contract to 31% of their starting distance from the centre (0.290 → 0.089)
  and **the teacher's own spread stays flat (ratio 1.03)**. The DT is fitting better
  *by predicting the mean harder*, and the mean is at the centre.

## Limits

- **Five arms, one failing, Borehole only.** `teacher_action_stats` was recorded
  only on h171 and h172, so this cannot be extended to the 28-arm set.
- `var_total` is over **all τ**, as is the training loss — so the loss/var identity
  is a statement about *training*. The τ=0 specificity comes from h180/h182, not
  from this table; what this adds is that the residual variance the DT captures is
  the right size and shape to be exactly the between-τ component.
- The best constant is computed post hoc on the same data, so ratios slightly above
  1.0 (1.022, 1.026) are expected and are not evidence of worse-than-constant fit.
- Correlational. Nothing here intervenes to force the DT off the constant solution.

## What could RETRACT it

- An arm with substantial τ-structure whose loss/var is well below ~0.8 — i.e. a DT
  that genuinely learns a state→action mapping — would break the constant-predictor
  reading.
- An arm that explains a large share of variance **and** performs well would break
  the "learning more does not help" claim. Currently the arm that explains the most
  performs the worst.
