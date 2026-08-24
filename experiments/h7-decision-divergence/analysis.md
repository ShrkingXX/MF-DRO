# H7 analysis — continued training changes ~18% of decisions

Instrument verified before interpretation (`instrument-verification.md`):
snapshot independent (0/93 params share storage), live model genuinely diverged
(77/93 params changed, `coef_head` by ~4.6e-03).

## Result — 377 paired decisions across 5 seeds

| seed | n decisions | argmax agreement | mean L2 dist | fidelity agreement |
|---|---|---|---|---|
| 42 | 97 | 0.763 | 0.150 | 0.804 |
| 43 | 53 | 0.906 | 0.069 | 0.679 |
| 44 | 82 | 0.780 | 0.188 | 0.817 |
| 45 | 83 | 0.807 | 0.110 | 0.843 |
| 46 | 62 | 0.887 | 0.048 | 0.742 |
| **pooled** | **377** | **0.817** | **0.121** | — |

- **Locked prediction 1 (agreement > 0.70): PASS** (0.817)
- **Locked prediction 2 (dist does not grow with t): FAIL** — slope
  +0.0012/iter, and agreement decays 0.860 (first half) -> 0.798 (second half).

## What this actually says — and a correction to my own earlier reading

The smoke test's `dist = 0.000000` on 4 consecutive early records led me to
describe the policies as producing "bit-identical" proposals. **At scale that is
wrong.** Over 377 decisions the live and snapshot policies agree on **82%** and
differ on **18%**, with mean displacement 0.121 in normalised space.

So continued training is **not** inert at the level of decisions. It changes
roughly one decision in five, and the divergence **grows with training** — which
is the expected direction, and which also demonstrates the instrument is
genuinely sensitive rather than reporting high agreement because it cannot
detect change.

## Reconciling H6 and H7 — this is the payoff

H6 (n=30) could not distinguish frozen from retrained: paired diff +0.075, CI
[-0.011, +0.161], p = 0.0795. H7 explains why, and it is *not* "because nothing
changes":

> Continued training changes ~18% of decisions, but those changes are worth
> little in regret — the paired regret difference is +0.075 with a CI containing
> zero.

That is a much more informative statement than either experiment alone, and it
is the opposite of what I was drifting toward at n=7 ("continued training
contributes ~nothing"). The correct claim is: **continued training does change
what the policy does; the changes just do not buy much.**

Combined with H5 (the score head ignores `h`) and H8 (RTG inert across its
realised band), the coherent picture is:

- the DT's **conditioning** pathway is genuinely inert (state, RTG, BTG),
- but its **weights** are not — retraining moves ~18% of decisions,
- and those weight-driven changes are close to regret-neutral.

So "MF-DRO re-fits rather than conditions" survives, but the re-fitting is doing
real work on the decisions; it simply is not *useful* work on this benchmark.

## Honest limitations

- 5 seeds, one benchmark, one snapshot point (k=5, fixed in advance).
- Agreement is measured against an iteration-5 snapshot, so it conflates "how
  fast the policy drifts" with "how much drift matters". A ladder of snapshot
  points would separate them.
- Prediction 2's failure means the k=5 choice matters: measured later, agreement
  would be lower still.
