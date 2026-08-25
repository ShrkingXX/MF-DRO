# H49 — does fixing the decision model's lengthscale anchor improve MF-DRO?

**CONFIRMATORY. Protocol committed before any run.**

## Where this came from

H48 built a standalone, validated Takeno MF-MES (V1–V6 all pass) and used it to
vary the surrogate with the acquisition held fixed. Decomposing the two knobs
that differ between the KO-GP defaults and mf_dro's ensemble member 0
(MF-MES, Hartmann 6D, 10 seeds, Bonferroni bar 0.05/3 = 0.0167):

| arm | mean regret | vs defaults | worse on | p | survives |
|---|---|---|---|---|---|
| KO-GP defaults | 0.2069 | — | — | — | — |
| member-0 `rho_init` only | 0.2742 | +0.0673 | 7/10 | 0.1934 | no |
| member-0 `initial_lengthscale` only | 0.3126 | **+0.1058** | **10/10** | **0.0020** | **yes** |
| both | 0.3408 | +0.1339 | 9/10 | 0.0039 | yes |

`initial_lengthscale = 0.1839` is the **shortest** point of the diversity grid,
and ko_gp.py's own docstring names that regime as near-interpolation
over-fitting. `ko_ensemble[0]` is simultaneously a rollout generator **and** the
model behind every real decision (mf_dro.py 2577, 2598, 2659, 2784, 3009).

## The intervention — one variable

`natural_decision_lengthscale=True` gives member 0 `initial_lengthscale=None`
(the KO-GP default) and spreads the diversity grid over members 1..M-1.
Member 0 stays in the rollout pool, so the train/inference state distributions
still match (preserving the earlier Bug A fix). `rho_init` is left alone — it
did not survive correction. Nothing about the acquisition or the DT changes.

## Arms

| arm | config | source |
|---|---|---|
| **control** | h45's `regression_direct` verbatim | **reused from h45** |
| **treatment** | identical + `natural_decision_lengthscale=True` | this experiment |

Reuse is sound because the flag defaults to `False`: with it unset the
construction is bit-identical to h45's, so h45's runs *are* the control. This
is stated up front rather than discovered afterwards.

**Head choice.** Both arms use the **regression head**
(`use_candidate_scoring=False`). h45 shows it is the stronger MF-DRO — 0.3711
vs 0.4523 on the 6 seeds available at protocol time, better on 5/6 — so the
scoring-head 0.4007 is no longer the reference configuration. Testing an
improvement against the weaker variant would inflate any gain.

Held fixed: Hartmann 6D, seeds 42–51, `cost_budget=200` post-init,
`initial_hf=36 / initial_lf=60`, `rollout_reward="mes_entropy"`,
`dkl_threshold=9999`, `num_epochs=10`, `rollout_length=8`.

## Prediction

**Primary: treatment beats control on paired final HF regret.** The mechanism is
measured — a better-conditioned decision GP gives better posterior means and
variances to every real decision.

**The null is live and I am stating why.** The H48 effect was measured with
MF-MES, where the surrogate feeds an acquisition that is then *properly
optimized*. In the regression-head MF-DRO the surrogate feeds the DT's state
features and `_refine_proposal` instead, and this project has repeatedly
measured the DT's decisions to be insensitive to its state input (0.13% of the
coefficient vector on the scoring head). A surrogate improvement can only help
if the downstream consumer is sensitive to it. If the effect vanishes here, that
is evidence the regression head is likewise state-insensitive — which is
informative, and is the reason to run it rather than assume the transfer.

## Analysis plan, fixed now

Single pre-declared endpoint: **final HF regret at the budget**, paired by seed,
Wilcoxon signed-rank over 10 pairs. One test, so alpha = 0.05 uncorrected.
Everything else I look at is labelled EXPLORATORY.
