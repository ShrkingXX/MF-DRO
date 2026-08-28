# H119 — screen results

EXPLORATORY SCREEN. Nothing here is a result. Data: h90, Borehole, seeds 47-51,
NO-ROI vs ROI-Q10, paired, n=5. All seven pre-enumerated quantities reported.

| | quantity | NO-ROI | ROI-Q10 | paired | sd | \|m\|/sd | dir |
|---|---|---|---|---|---|---|---|
| C1 | HF share of cost budget | 0.775 | 0.704 | -0.071 | 0.062 | **1.15** | 5/5 |
| C2 | HF query count | 93.2 | 84.6 | -8.60 | 7.57 | **1.14** | 5/5 |
| C3 | LF query count | 14.0 | 31.0 | +17.00 | 14.73 | **1.15** | 5/5 |
| C4 | time-to-incumbent (frac budget) | 0.938 | 0.748 | -0.190 | 0.141 | **1.35** | 5/5 |
| C5 | HF quality (sd of init design) | 3.289 | 3.563 | +0.274 | 0.021 | **12.82** | 5/5 |
| C6 | frac HF worse than best init | 0.057 | 0.030 | -0.027 | 0.043 | 0.62 | 3/5 |
| C7 | early/late dispersion contraction | 3.069 | 3.298 | +0.229 | 0.457 | 0.50 | 3/5 |

Five of seven clear the descriptive bar. Per pre-committed rule 3, that is
roughly what correlated measures at n=5 can produce, and it is not resolved by
picking the largest. What the five say:

**C1, C2, C3 are one fact, not three.** They are mechanically linked by a fixed
cost budget: HF costs 2, LF costs 1, so fewer HF queries necessarily means more
LF ones and a smaller HF share. The single fact is that **the ROI reallocates
budget from HF to LF** — 93.2 -> 84.6 HF queries, 14 -> 31 LF, in 5/5 seeds.

**C4** says the ROI reaches its final best point after 75% of the budget instead
of 94% — it settles earlier.

**C5** says the ROI's individual HF queries are better, by 0.274 initial-design
standard deviations, with a paired sd of 0.021 across five seeds. The effect
size of 12.82 comes from that very small paired spread; the standardiser is the
seed's own initial design, which is IDENTICAL within a pair, which is why the
pairing is so tight. It is not an error, but the number to quote is +0.274 sd,
not "12.82".

**C6 did not separate (0.62)** — and C6 is the founding diagnosis's own
statistic, the fraction of HF queries worse than the initial design. On Borehole
it is already small in both arms (5.7% -> 3.0%).

## The hypothesis this generates (and it is only that)

> The ROI does not make MF-DRO search differently in space. It makes it **buy
> less high fidelity** — 9% fewer HF queries, redirected into more than twice as
> many LF queries — while each HF query it does buy is individually better, and
> it converges earlier in the budget.

That is a budget-allocation story, not a spatial-search story. It is consistent
with dispersion (h116) and boundary resolution (h118) both failing: those are
spatial channels, and this says the ROI acts on the fidelity mix instead.

**C5 has an obvious confound the screen cannot address**: quality is averaged
over 9% fewer queries in a run that also converges earlier (C4). A more
converged run has a better average by construction. Any confirmation must
count-match.

## Registered next step

Confirmatory test on **h84-roi-strategy, seeds 42-46** — which contains ROI-OFF
and ROI-Q10 in one experiment, at seeds disjoint from the 47-51 used here, and
so played no part in generating this hypothesis. Protocol to be committed before
those numbers are computed. Primary measure must be count-matched (compare the
first K HF queries, K = the smaller of the two arms' counts within each seed).

Pending that, the fidelity-reallocation account is a hypothesis with no
confirmatory support.
