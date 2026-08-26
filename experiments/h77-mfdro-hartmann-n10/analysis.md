# H77 result — MET. MF-DRO's published Hartmann entry was wrong by 5.77 points.

**CONFIRMATORY** against `protocol.md`; verdict script committed at 0/7 with both
gates in code. **Reproduction control PASS bit-for-bit**: h77's worker on seed 44
gives 0.7531352462, identical to h57's, **diff 0.000e+00**. Worker byte-identical
to h57's.

## Verdict — both predictions met

| prediction | bar | result | verdict |
|---|---|---|---|
| PRIMARY | three-seed span >= 8.0 pts | **16.14** | **MET** |
| SECONDARY | \|shift from published 14.68%\| >= 2.0 pts | **5.77** | **MET** |
| NULL | span < 8.0 **and** \|shift\| < 2.0 | — | did not fire |

| | |
|---|---|
| MF-DRO Hartmann, n=10 | **8.91%** (sd 7.39) |
| per seed | 20.3 / 3.6 / 22.7 / 6.3 / 8.7 / 2.2 / 12.7 / 1.4 / 5.6 / 5.6 |
| published n=3 (44/46/48) | 14.68% |
| shift | **−5.77 pts** |
| three-seed range | [2.42, 18.56], span **16.14** |

Seeds 44/46/48 were the three worst-but-one draws available. A three-seed estimate
of this cell could have landed anywhere in **[2.42, 18.56]** — a 16-point window.

## The standings change materially

**Hartmann at n=10, every method with 10 seeds:**

| method | n=10 | published n=3 |
|---|---|---|
| MF-DRO + POOL600 | 6.64% | (advantage withdrawn by h66) |
| MF-MES | 8.24% | 8.52% |
| SF-DRO | 8.46% | 11.49% |
| **MF-DRO** | **8.91%** | **14.68%** |
| SF-EI | 18.61% | — |
| SF-MES | 21.17% | 21.39% |
| MF-MI-Greedy | 36.61% | 23.93% |
| MF-GP-UCB | 66.81% | 45.29% |

**Four methods now cluster at 6.6-8.9%, then a 10-point gap.** The published table
put MF-DRO at 14.7% — mid-table and clearly behind MF-MES's 8.5%. At n=10 the two
are **not separable**.

## POST HOC — MF-DRO vs MF-MES. Not a claim.

The means were seen before this comparison was posed, so it cannot be
confirmatory.

| | mean | sd | wins | Wilcoxon p |
|---|---|---|---|---|
| MF-DRO | 8.91% | 7.39 | **4/10** | 0.2754 |
| MF-MES | **8.24%** | 6.85 | 6/10 | |

**A tie, not a win.** MF-MES is marginally ahead on the mean and takes 6 of 10
seeds. Note seed 49, where MF-MES posts 25.25% against MF-DRO's 1.39% — the same
single-seed volatility that drove h64's withdrawn claim, now running the other way.

## The north star is unchanged

Still **not met**. It is a per-benchmark bar:
- **Hartmann**: MF-DRO ties MF-MES (4/10, p=0.2754). A tie is not "at least as
  good as the baselines" established — it is an absence of evidence either way.
- **Borehole**: MF-DRO 22.89% against MI-Greedy's 9.29% (h75, n=10). A clear loss,
  and h75 confirmed it is not a three-seed artifact.

What h77 changes is the *size* of MF-DRO's Hartmann deficit, not the verdict.

## What this cannot settle

Hartmann only; MF-DRO's Currin entry stays n=3 (Currin does not discriminate —
every non-degenerate method finishes inside 0.6%, so its span is bounded by
construction). The MF-DRO-vs-MF-MES comparison above is post hoc and would need
fresh seeds to be confirmatory.
