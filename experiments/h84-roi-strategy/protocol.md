# H84 — ROI strategy: stop MF-DRO wasting HF budget on low-value regions

LOCKED BEFORE ANY RUN.

## The problem, as measured (h83, n=5)

MF-DRO's HF queries are worth far less on average than MF-MES's on the SAME KO
surrogate, initial design and budget. Scoring each post-init HF query as
`(y - best_init) / (y_opt - best_init)`:

| | mean query | worse than init | per-dim spread |
|---|---|---|---|
| Hartmann MF-DRO | **0.336** | **20.8%** | 0.097 |
| Hartmann MF-MES | 0.747 | 5.3% | 0.028 |
| Borehole MF-DRO | **0.381** | 7.9% | 0.078 |
| Borehole MF-MES | 0.669 | 0.5% | 0.074 |

MF-DRO's DT regression head emits x directly and is trained on rollout-teacher
actions, which are argmaxed over `roi_candidates`. With `use_roi=False` that
pool is UNIFORM over the whole domain, so the DT's training targets are spread
across the domain. The ROI is therefore the lever that shapes what the policy
learns to propose.

## Why a constant beta cannot work (measured, this experiment's motivation)

Acceptance |X_hat|/|X| under the paper's rule at sqrt(beta)=2, as HF data grows:

| benchmark | early | mid | late |
|---|---|---|---|
| Currin_2D | 20.0% | 8.0% | 1.7% |
| Hartmann_6D | 12.6% | 3.4% | 6.0% |
| Borehole_8D | **100.0%** | 29.6% | **0.4%** |
| Ackley_10D | **100.0%** | **100.0%** | 16.5% |

A 250x swing on Borehole. At a fixed beta the ROI is VACUOUS exactly when the
surrogate is worst (early), then collapses to 8 of 2000 points once data
accumulates. Both are failure modes, in opposite directions.

## The strategy

Control the ACCEPTANCE RATE and solve for beta_t. The ROI set remains exactly
the paper's `X_hat_{m,t} = {x | UCB_{m,t}(x) >= max_x' LCB_{m,t}(x')}`
(Sec 4.2); only beta_t -- which the paper leaves unspecified but explicitly
writes with a SUBSCRIPT t -- is chosen by a criterion rather than guessed.
Acceptance is monotone increasing in beta (raising it lifts every UCB and lowers
max LCB), so beta_t is found by bisection. Verified to hit a 10% target exactly
on all four benchmarks at two data sizes, with beta_t ranging 0.31 to 12.29 --
a 40x range no constant covers.

## Arms

| arm | ROI | beta |
|---|---|---|
| A `ROI-OFF` | off | -- (current shipped default) |
| B `ROI-FIX2` | on | fixed sqrt(beta)=2, the rule as literally implemented |
| C `ROI-Q10` | on | beta_t calibrated to constant acceptance q=0.10 |
| D `ROI-ANN` | on | beta_t calibrated to annealed q: 0.50 -> 0.05 over the run |

Benchmarks Hartmann_6D and Borehole_8D (MF-DRO's two losses; Borehole is the
decisive one). Seeds 42-46. Budget 200 post-init. M=3, pool 600, refinement off.

CONTROL REUSE: arm A is byte-for-byte the configuration h83 already ran, and the
`use_roi=False` path was gated bit-identical (max|dx| = 0.000e+00, fidelity
sequences identical) across the ROI-pool fix AND the beta-calibration commit.
h83's MF-DRO runs are therefore reused as arm A, and 2 seeds per benchmark are
RE-RUN as a live reproduction control. That control must reproduce h83 exactly;
if it does not, the reuse is void and all of arm A is re-run. This is an
explicit guard against the vacuous-control failure this project has already
committed once (claiming a control passed when it had never run).

## Metrics (frozen)

- PRIMARY: mean HF query quality score, `(y - best_init)/(y_opt - best_init)`,
  over post-init HF queries. This is the direct measure of budget waste.
- SECONDARY: final relative regret at cost 200.
- TERTIARY: fraction of post-init HF queries scoring < 0 (worse than the best
  initial-design point).

Diagnostics recorded per rollout: ROI acceptance, beta_t chosen, distinct
candidates, min distance to x*.

## Predictions (pre-registered, INDEPENDENT -- each reported pass or fail)

- **P1 (PRIMARY).** Arm C beats arm A on mean HF query score by >= +0.10
  absolute on BOTH benchmarks, on >= 4/5 seeds each. (Arm A is 0.336 Hartmann /
  0.381 Borehole; MF-MES sits at 0.747 / 0.669.)
- **P2.** Arm B does NOT beat arm A on Borehole. The fixed-beta ROI is vacuous
  at n_hf=10 (100% acceptance) and collapsed at n_hf=35 (0.4%), so it should
  buy nothing there. This is a NEGATIVE prediction about the paper's rule as
  literally parameterised.
- **P3 (SECONDARY).** Arm C's final relative regret is lower than arm A's on
  Borehole. Stated SEPARATELY from P1 because better average query quality does
  NOT imply a better final best-point -- a method can improve its mean and still
  miss the optimum.
- **P4.** Arm D is not worse than arm C on either benchmark. Direction only.

No p-values at n=5. Every run is reported including failures and gate misses.

## What would falsify the whole strategy

If C and D both fail P1, the ROI does not shape the DT's proposals in the way
the mechanism predicts, and the budget-waste problem is NOT a candidate-
distribution problem. That would redirect at the output parameterisation (an L2
head pulled toward the interior) instead -- see the boundary-aversion correction
in findings.md.
