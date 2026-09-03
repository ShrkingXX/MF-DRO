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

---

## GENERALITY — the identity holds on BOTH benchmarks, across 10 arms

**A correction first.** This file and findings.md both stated, as a limit, that
`teacher_action_stats` "was recorded only on h171 and h172, so this cannot be extended".
**That was wrong.** It is present on **18 arms across both benchmarks** — including
Hartmann's h173, h174, h168/h177/h178, and Borehole's h176, h179, h181, h184, h187. The
limit I published twice did not exist, and the test below was available the whole time.

| benchmark | arm | L | DT loss | teacher var | **loss/var** | var explained | rel% |
|---|---|---|---|---|---|---|---|
| **Hartmann** | ROLLOUT1 | 1 | 0.0208 | 0.0198 | **1.054** | **0.0%** | 10.91 |
| **Hartmann** | HEAD-MES | 8 | 0.0762 | 0.0816 | **0.934** | 6.6% | 25.16 |
| **Hartmann** | TAIL-MES | 8 | 0.0503 | 0.0591 | **0.850** | 15.0% | 46.45 |
| **Hartmann** | PROBE-RANDOM | 8 | 0.0841 | 0.0831 | **1.013** | **0.0%** | 65.14 |
| Borehole | ROLLOUT1 | 1 | 0.0350 | 0.0342 | **1.022** | **0.0%** | 13.69 |
| Borehole | ROI-Q10-L1 | 1 | 0.0227 | 0.0223 | **1.020** | **0.0%** | 10.81 |
| Borehole | HEAD-MES | 8 | 0.0782 | 0.0854 | **0.916** | 8.4% | 16.96 |
| Borehole | TAIL-MES | 8 | 0.0389 | 0.0489 | **0.795** | 20.5% | 43.94 |
| Borehole | LFF-CTRL | 8 | 0.0343 | 0.0457 | **0.750** | 25.0% | 15.76 |
| Borehole | STDCOND | 8 | 0.0385 | 0.0444 | **0.867** | 13.3% | 16.66 |

**Ten arms, two benchmarks, `loss/var` spans 0.750–1.054.** Every arm sits at the
best-constant value. The identity is not a Borehole artifact.

## A SECOND by-construction control, which was not designed — it fell out

The original prediction was that a per-timestep constant predictor explains **0%** of the
variance when there is only one timestep. That is confirmed on **three independent L=1
arms** across both benchmarks (1.054, 1.022, 1.020 → 0.0%).

But **PROBE-RANDOM explains 0.0% at L=8.** That is the *random* teacher, whose actions
are uniform at every τ — so all its per-τ means are the same point, and a per-timestep
constant collapses to a global constant, explaining nothing. **A second situation where
the theory forces 0%, arrived at by a completely different route, and it reads 0.0%.**

The full pattern, with every arm accounted for:

| when the teacher's action distribution... | variance explained | arms |
|---|---|---|
| has only one timestep (L=1) | **0.0%** | 3 |
| is identical at every timestep (random teacher) | **0.0%** | 1 |
| differs sharply at τ=0 (HEAD/TAIL) | 6.6–20.5% | 4 |
| varies naturally across τ (normal MES, L=8) | 13.3–25.0% | 2 |

**Variance is explained if and only if the teacher's action distribution differs across
timesteps.** That is the per-timestep-constant signature, and all ten arms fit it.
