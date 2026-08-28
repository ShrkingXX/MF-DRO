# H121 — Where does the founding diagnosis's wasted budget actually live?

STATUS: LOCKED before any statistic was computed.
TYPE: CONFIRMATORY (reproduction check of a number already on record, plus a
      pre-stated prediction about its distribution across benchmarks).
COMPUTE: zero new runs. Re-analysis of h83-main-comparison.
DATA: h83 ONLY — 4 benchmarks x MF-DRO and MF-MES x seeds 42-46.

## Why

The founding diagnosis states: "MF-DRO's mean HF query score is 0.336 vs
MF-MES's 0.747 on Hartmann, 20.8% of its HF queries land WORSE than the initial
design". That 20.8% is the quantity the ROI is supposed to reduce, and it is a
**Hartmann** number.

h119 computed the same statistic on Borehole and it DID NOT SEPARATE between
NO-ROI and ROI-Q10 (0.057 -> 0.030, effect 0.62) — because on Borehole it is
already small in both arms.

Meanwhile every ROI benefit in this project is Borehole-specific, and h111
showed the ROI's regret effect fails on Hartmann and Ackley at two tightness
settings spanning 2x.

If the waste lives on Hartmann and the ROI's benefit lives on Borehole, then the
ROI cannot be the fix for the founding diagnosis, and the primary question as
posed contains a mismatch that no ROI strategy can resolve. That is worth
establishing directly rather than inferring across three experiments.

## Measure

Per benchmark x method x seed, over non-init HF queries:
  waste_frac = fraction with y < max(y over the seed's INITIAL-DESIGN HF points)
  score      = (mean non-init HF y - mean init HF y) / (sd of init HF y)
Both are computed within seed, so the initial design (shared by both methods at
a given seed) is the reference. n=5 per cell.

## Predictions (locked)

P1. The recorded 20.8% reproduces on Hartmann for MF-DRO, within a tolerance of
    +/- 10 percentage points (i.e. lands in 10.8-30.8%). This is a reproduction
    check of a number on record, not a new claim; a miss means the recorded
    figure came from a different measurement and must be traced.
P2. MF-DRO's waste_frac on Hartmann EXCEEDS its waste_frac on Borehole, in the
    per-seed sense (Hartmann median > Borehole median).
P3. MF-DRO's waste_frac exceeds MF-MES's on Hartmann, in >= 4/5 seeds.

## What a pass would mean

That the waste the diagnosis names is a Hartmann phenomenon, that the ROI's
only demonstrated benefit is a Borehole phenomenon, and therefore that **the
ROI is not addressing the diagnosis it was introduced to address.** That is a
negative result about the primary question's own framing, and it should be
reported as such rather than buried.

## What a fail would mean

If P2 fails — if the waste is comparable on both benchmarks — then the ROI
operates on a benchmark where the waste exists, and the mismatch above
dissolves. If P1 fails, the 20.8% on record is not what this measurement
computes and the discrepancy must be resolved before anything is built on it.

## Limitations

- n=5 seeds, no p-values.
- "Worse than the best initial-design HF point" is a coarse notion of waste: it
  ignores how much worse, and a query can be informative without improving on
  the incumbent. It is used because it is the diagnosis's OWN statistic and the
  point is to check the diagnosis on its own terms.
- MF-MES is included only for P3. Per the h117 amendment-2 confound it refines
  with box-constrained L-BFGS-B, so it is not a neutral reference for spatial
  quantities; waste_frac is a value statistic, not a spatial one, so the
  confound is weaker here but not absent.
