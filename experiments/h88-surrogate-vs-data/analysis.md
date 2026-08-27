# H88 analysis — better data does NOT give a better model. And a correction.

Same KO GP (dkl_threshold=9999, matching h83), same 4096-point Sobol
recommender, only the DATA differs.

| | A: fit on MF-DRO's queries | B: fit on MF-MES's queries | A's own best | B's own best |
|---|---|---|---|---|
| Hartmann_6D | **13.15%** | 19.92% | 7.55% | 6.62% |
| Borehole_8D | 22.66% | 21.42% | 15.82% | 6.36% |

## P1 FAILED on both benchmarks — and the direction is the interesting part

P1 predicted that fitting on MF-MES's (better) data would give better
recommendations, which would have made the policy the lever. It does not:
1/5 seeds on Hartmann, 3/5 on Borehole. On Hartmann the model fit on MF-MES's
data is substantially WORSE (19.92% vs 13.15%).

MF-MES's queries are better POINTS but a worse DESIGN. We measured its
per-dimension query spread at 0.028 against MF-DRO's 0.097 on Hartmann -- it
concentrates, which is what finds good points and what makes a GP fit on it
extrapolate badly across a dense global pool.

**This reframes MF-DRO's dispersion.** Its 3x-wider queries have been described
throughout this project as waste. They are not only waste: they produce a
measurably BETTER GLOBAL SURROGATE than the method that beats it. MF-DRO's
failure is not that it learns a worse model of the function -- it is that it
does not convert a better model into better queries.

## P2 FAILED on Hartmann — the ratio-1.00 finding needs correcting

P2 was registered as a self-check: does "the model never recommends better than
its own best query" survive a stronger recommender? On Hartmann it does not.
Seed 42: the 4096-point recommendation scores **11.96%** against MF-DRO's own
best query of **16.41%** -- the model DOES know a better point there.

The earlier ratio-1.00 result used the live metric's 512-point `y_star_pool`
with a single ensemble member. That recommender was too weak to see this. The
corrected statement is:

  MF-DRO's surrogate recommends a better point than its own best query on
  **1 of 10 runs** (Hartmann seed 42) under a 4096-point recommender, and on
  **0 of 20** under the live 512-point one.

The qualitative conclusion survives -- the model almost never holds hidden
knowledge, so "good model, bad policy" remains a poor explanation of the bulk of
the gap. But "never" was wrong, and it was wrong because of an instrument I had
already flagged as weak and then reasoned from anyway.

## P3 MET on both

Neither model recommends a point better than MF-MES's own best query. Registered
negative; nothing gives the GP information beyond its observations, and nothing
surprising happened.

## The number that limits all of this

Every recommendation regret (11.96%-25.29%) is far worse than either method's
own best query (0.67%-19.19%). At these data sizes the KO surrogate's global
argmax is simply a poor point on both benchmarks. Whatever is wrong with MF-DRO,
a better recommender on top of this surrogate is not the fix.

## What it cannot settle

Both designs came from optimisers, so neither is a neutral probe of the
surrogate class. Condition C -- a space-filling design at matched cost -- was
pre-registered as the follow-up if A/B proved informative. They did.

## Condition C — P4 FAILED 0/5, and it corrects my reading of A/B

| | A: MF-DRO data | B: MF-MES data | C: Sobol data | C's best observation |
|---|---|---|---|---|
| Hartmann_6D | **13.15%** | 19.92% | 29.03% | 65.51% |
| Borehole_8D | 22.66% | 21.42% | 22.64% | 31.90% |

P4 predicted a space-filling design would beat both optimisers' designs, on the
reasoning that A/B had shown the MORE dispersed design giving the better model.
It loses on 5/5 seeds on both benchmarks.

**So recommendation quality is NOT monotone in dispersion.** The ordering is
MF-DRO (moderate spread) > MF-MES (concentrated) > Sobol (maximal spread). Both
extremes are bad. My A/B conclusion -- "MF-DRO's dispersion produces a better
global surrogate" -- was an extrapolation from two points that a third point
refutes. The corrected statement is that there is an INTERIOR optimum in design
dispersion for global model quality, and MF-DRO happens to sit nearer it than
MF-MES does. That is a much weaker and more accidental-sounding claim, and it is
the one the data supports.

The mechanism is visible in the last column: Sobol's best observed HF value is
65.51% relative regret on Hartmann. A GP fit on data that never approaches the
optimum has no information about the good region, however evenly it covers the
domain. Coverage does not substitute for having been somewhere good.

## The registered interpretation of P4's failure applies

The protocol stated, before running C: "If P4 FAILS -- if a design built purely
for coverage cannot beat two optimisers' designs -- then the limit is the
SURROGATE at this budget, not the data, and no policy change can fix the
recommendation gap."

It failed. Every recommendation across all three conditions and both benchmarks
lands in 11.96%-29.03%, while the best OBSERVED points reach 0.67%-19.19%. **The
KO surrogate's global argmax is a poor point regardless of what it is fit on.**

CONSEQUENCE FOR THE PROJECT: improving which points MF-DRO queries cannot close
the recommendation gap, because the gap is not about the data. This does not
excuse MF-DRO's query quality -- simple regret depends on the best QUERY, not on
the recommendation, and that is where MF-DRO loses to MF-MES. But it does close
off "recommend from the model instead of reporting the best query" as an avenue,
and it bounds how much any data-side intervention (ROI, refinement, HF floor)
can be expected to buy.

STATUS: EXPLORATORY. n=5, two benchmarks, one surrogate class. Condition C's
design is diagnostic only -- no optimiser could adopt it, since it never
exploits.
