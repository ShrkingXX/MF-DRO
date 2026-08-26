# H75 result — MET. MF-DRO's published Borehole entry holds up.

**CONFIRMATORY** against `protocol.md`; verdict script committed at 0/7 with both
gates enforced in code. **Reproduction control PASS, bit-for-bit**: h75's worker
on Borehole seed 44 gives 75.6374736643, identical to h57's, **diff 0.000e+00**.
The worker is byte-identical to h57's, so the 7 new seeds share h57's code path
exactly.

## Verdict — both predictions met

| prediction | bar | result | verdict |
|---|---|---|---|
| PRIMARY | \|shift from published n=3\| < 3.0 pts | **0.82** | **MET** |
| SECONDARY | C(10,3) three-seed span < 8.0 pts | **5.89** | **MET** |
| NULL | shift >= 3.0 pts | — | did not fire |

| | |
|---|---|
| MF-DRO Borehole, n=10 | **22.89%** (sd 2.94) |
| per seed | 22.3 / 25.0 / 24.4 / 25.2 / 24.7 / 20.5 / 22.0 / 16.2 / 22.3 / 26.1 |
| published n=3 (44/46/48) | 23.71% |
| shift | −0.82 pts |
| three-seed range | [19.56, 25.45], span 5.89 |

## The calibration programme is now closed for Borehole

| method | n=10 | 3-seed span |
|---|---|---|
| MF-MI-Greedy | **9.29%** | 4.13 |
| SF-MES | 12.76% | 5.65 |
| SF-EI | 12.95% | 3.95 |
| SF-DRO | 14.60% | 3.11 |
| **MF-DRO** | **22.89%** | 5.89 |
| MF-GP-UCB | 46.65% | 18.82 |

Every Borehole entry is now n=10 with a passed reproduction control. **MF-DRO is
last among the non-degenerate methods**, and that was not an artifact of three
lucky or unlucky seeds — its published 23.71% was accurate to 0.82 points, and
the whole three-seed range [19.56, 25.45] sits well clear of SF-DRO's 14.60%.

Contrast with Hartmann, where the same check moved published baseline entries by
**+12.7** and **+21.5** points (h72). Borehole is the stable benchmark; Hartmann
is not. h75's PRIMARY was set from that measured fact rather than from intuition.

## A pattern in the prediction record worth noting

This project's met predictions are the ones derived from **prior measurements**;
the failed ones are **mechanism intuitions**.

| met | grounded in |
|---|---|
| h70 (all three bars) | h61's measured 1.44x acquisition-value gain from pool widening |
| h72 (both bars) | the observed n=3 spread in existing results |
| h75 (both bars) | h72's measurement that Borehole shifts little at n=10 |

| failed | grounded in |
|---|---|
| h63/h67 rho story | intuition that the sigmoid ceiling binds |
| h69 acquisition class | intuition that EI-vs-MES explained the Borehole gap |
| h76 trajectory shape | intuition that the advantage was early search |

Six mechanisms proposed and refuted; three measurement-derived predictions met.

## What this cannot settle

Borehole only. MF-DRO's **Hartmann and Currin entries remain n=3**, and Hartmann
is precisely the column h72 showed to be least resolved. It also re-ranks nothing
— a better estimate of MF-DRO's Borehole regret does not change that it loses
there by a wide margin.
