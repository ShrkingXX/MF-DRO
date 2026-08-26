# H70b — WITHDRAWN, and the direction reverses

**CONFIRMATORY** against `protocol_b.md`, locked before any n=10 data existed.

## Verdict

| prediction | bar | result | verdict |
|---|---|---|---|
| PRIMARY | Hartmann: ALTGP >= 3.0 pts better **and** >= 7/10 wins | **−3.03 pts, 2/10** | **NOT MET** |
| SECONDARY | Borehole \|diff\| < 1.0 pt | **0.03 pts** | **MET** |
| NULL | < 3.0 pts or <= 6/10 wins | fired | **WITHDRAW** |
| REVERSED | KO-style better by >= 3.0 pts | **fired (3.03)** | — |

| Hartmann (n=10) | mean | sd | per seed |
|---|---|---|---|
| KO-style | **18.61%** | 6.75 | 16.2 / 11.8 / 22.7 / 13.2 / 29.5 / 25.2 / 7.5 / 23.8 / 17.7 / 18.5 |
| ALTGP (plain) | 21.63% | 10.81 | 16.2 / 11.8 / 15.7 / 21.1 / 17.2 / 27.3 / 7.5 / 23.8 / **45.2** / 30.4 |

| Borehole (n=10) | mean | sd |
|---|---|---|
| KO-style | 12.95% | 2.14 |
| ALTGP | 12.92% | 2.12 |

## The h70 finding is withdrawn

h70 reported, as an unpredicted exploratory observation, that the KO-style GP
construction **costs 6.41 points on Hartmann**. At n=10 the KO-style builder is
**better by 3.03 points and wins 8 of 10 seeds**. The direction reversed.

The built-in control settles what happened: seeds 44/46/48 reproduce h70 exactly
(19.89% vs 13.48%). The n=3 numbers were correct — they were simply the three
seeds where the plain builder happened to win. The full set contains seed 50,
where the plain builder posts **45.2%** against KO-style's 17.7%.

**SECONDARY held.** Borehole neutrality is real and now much stronger: 0.03 points
at n=10, with the two builders differing on only one seed out of ten despite
producing materially different models (verified max lengthscale difference 4.27).

## Third n=3 -> n=10 reversal in this project

| finding | at n=3 | at n=10 | outcome |
|---|---|---|---|
| h45 regression head | 5/6, then 7/8 favourable | worst-on-mean, p=1.0000 | withdrawn |
| h64 POOL600 Hartmann | 7.6% vs 8.5%, 2/3 | 5/10 wins, p=1.0000 | withdrawn |
| h70 KO-style GP | ALTGP +6.41 pts | ALTGP **−3.03** pts, 2/10 | withdrawn |

Three for three. Every exploratory n=3 direction this project has taken to n=10
has failed to survive, and one reversed sign. **n=3 on these benchmarks does not
estimate a direction**, and the honest operating rule is that no n=3 result is
reportable as a finding — only as a reason to run n=10.

The cost of getting this right was under two minutes of compute. The cost of
getting it wrong, twice before, was a shipped default and a retracted north-star
claim.

## What is now claimed, and how strongly

KO-style better on Hartmann by 3.03 points, 8/10 — an n=10 result that has not
itself been replicated, reported as a direction. It also has the lower variance
(6.75 vs 10.81). Borehole: the two builders are equivalent, 0.03 points at n=10.
