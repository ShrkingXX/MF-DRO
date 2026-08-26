# H72 result — the Hartmann column of the standings is not resolved at n=3

**CONFIRMATORY** against `protocol.md`; verdict script committed at 76/120 before
any number was seen. **Reproduction control PASSED**: h72's seeds 44/46/48
reproduce every published h57/h59 cell to **+0.00**, so the harness is identical
and the n=10 shifts below are real, not implementation drift.

## Verdict

| prediction | bar | result | verdict |
|---|---|---|---|
| PRIMARY | some cell's 3-seed range >= 5.0 pts | **51.83** (Hartmann/MF-GP-UCB) | **MET** |
| SECONDARY | some asserted ordering unresolved at n=3 | overlapping pairs found | **MET** |
| NULL | all < 5.0 and no overlap | — | did not fire |

## What a 3-seed draw could have said

All C(10,3) = 120 three-seed subsets, the exact estimator the standings used:

| benchmark | method | n=10 mean | 3-seed min | 3-seed max | range |
|---|---|---|---|---|---|
| Currin | MI-Greedy | 0.06% | 0.00% | 0.20% | 0.20 |
| Currin | MF-GP-UCB | 16.46% | 6.92% | 26.48% | **19.56** |
| Currin | SF-MES | 0.00% | 0.00% | 0.00% | 0.00 |
| Currin | SF-EI | 0.00% | 0.00% | 0.01% | 0.01 |
| Hartmann | MI-Greedy | 36.61% | 22.76% | 48.82% | **26.07** |
| Hartmann | MF-GP-UCB | 66.81% | 36.68% | 88.52% | **51.83** |
| Hartmann | SF-MES | 21.17% | 16.18% | 26.30% | 10.13 |
| Hartmann | SF-EI | 18.61% | 10.84% | 26.15% | 15.32 |
| Borehole | MI-Greedy | 9.29% | 7.15% | 11.29% | 4.13 |
| Borehole | MF-GP-UCB | 46.65% | 37.83% | 56.64% | **18.82** |
| Borehole | SF-MES | 12.76% | 9.86% | 15.52% | 5.65 |
| Borehole | SF-EI | 12.95% | 10.83% | 14.78% | 3.95 |

## The published n=3 entries were optimistic on Hartmann

| benchmark | method | published n=3 | n=10 | shift |
|---|---|---|---|---|
| Hartmann | **MI-Greedy** | 23.9% | **36.6%** | **+12.7** |
| Hartmann | **MF-GP-UCB** | 45.3% | **66.8%** | **+21.5** |
| Currin | MF-GP-UCB | 10.0% | 16.5% | +6.5 |
| Borehole | MI-Greedy | 8.3% | 9.3% | +1.0 |
| Borehole | MF-GP-UCB | 44.1% | 46.7% | +2.6 |
| Hartmann | SF-MES | 21.4% | 21.2% | −0.2 |
| Borehole | SF-MES | 13.3% | 12.8% | −0.5 |

Seeds 44/46/48 were a **favourable draw for MI-Greedy and MF-GP-UCB on Hartmann**
by 12.7 and 21.5 points. SF-MES was stable everywhere (|shift| <= 0.5), so the
noise is method-dependent, not a uniform property of the benchmark.

## Which orderings actually move

**Borehole is robust.** MI-Greedy stays best among the calibrated methods
(8.3% -> 9.3%) and its range [7.15, 11.29] does not reach SF-MES's [9.86, 15.52]
midpoint. h70's finding — that pool size alone reproduces MI-Greedy exactly — is
unaffected, being a per-seed identity rather than a mean comparison.

**Hartmann is not resolved.** At n=3, MI-Greedy's subset range [22.8, 48.8]
overlaps SF-MES's [16.2, 26.3] and SF-EI's [10.8, 26.2]: some three-seed draw
would have put MI-Greedy *ahead* of both, reversing the published ordering.

The other flagged overlaps are on **Currin and are not substantive** — SF-MES,
SF-EI and MI-Greedy all sit at 0.0-0.2%, so their ranges touch because every
method solves the problem, not because the ordering is uncertain. Reporting them
as "unresolved orderings" would overstate the result.

## The limitation that matters most

**MF-DRO and SF-DRO are not calibrated here** — at 82-473 min per run, n=10
across three benchmarks was not affordable. Their n=3 standings entries are
therefore of *unknown* reliability, and the per-seed spread already on record is
not reassuring: MF-DRO's Hartmann seeds are 22.7% / 8.7% / 12.7%, a spread
comparable to the calibrated methods whose means moved by >10 points.

So the honest position is not "the standings are wrong" but: **the Hartmann column
is unresolved at n=3 for every method measured here, and untested for the two
methods the project is about.**

## What this does not do

It does not re-rank anything. A wide subset range means an ordering is
*unresolved*, not reversed. The n=10 means are better estimates than the n=3 ones
they replace, but they are still n=10 and carry their own error.
