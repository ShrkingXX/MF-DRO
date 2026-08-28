# H121 — analysis

Data: h83 only, 4 benchmarks x MF-DRO/MF-MES x seeds 42-46. Zero new runs.

## Verdicts

**P1 PASS — and it reproduces exactly.** Hartmann MF-DRO waste per seed:
[0.0, 0.0, 75.0, 16.7, 12.5]%. **Mean = 20.83%.** The recorded headline is
20.8%, so the number on record is the MEAN across seeds and it reproduces to
two significant figures.

**P2 PASS.** Hartmann waste (median 12.5%, mean 20.8%) exceeds Borehole's
(median 3.2%, mean 7.9%).

**P3 FAIL (gate miss).** MF-DRO's waste exceeds MF-MES's on Hartmann in only
**3 of 5** seeds, against a required 4. The paired mean is +15.6 points, so
MF-DRO is clearly worse on average — but two seeds are tied at exactly 0.0%
(both methods waste nothing), and a tie is not an exceedance. Reported as a
gate miss rather than rounded up.

Full table (waste_frac, non-init HF queries below the best initial-design point):

| benchmark | MF-DRO per seed | median | MF-MES median | MF-DRO mean n_HF |
|---|---|---|---|---|
| Hartmann_6D | 0.0, 0.0, 75.0, 16.7, 12.5 | 12.5% | 0.0% | 11.6 |
| Borehole_8D | 3.2, 17.2, 3.2, 2.2, 13.8 | 3.2% | 0.0% | 94.0 |
| Currin_2D | 0.0, 0.0, 0.0, 0.0, 0.0 | 0.0% | 0.0% | 27.2 |
| Ackley_10D | 2.5, 0.0, 2.5, 0.0, 2.5 | 2.5% | 0.0% | 40.0 |

## The headline number is fragile, and this is the first time that is on record

Hartmann MF-DRO per-seed HF query counts are **8, 24, 12, 6, 8**. So:

- The 75% seed is **9 of 12** queries.
- The 16.7% seed is **1 of 6**.
- The 12.5% seed is **1 of 8**.
- **Two of five seeds waste nothing at all.**

The 20.8% that launched this whole line of work is a mean over five fractions
computed on as few as six queries each, from a distribution with two zeros and
one extreme value. It is not wrong — it reproduces exactly — but it is one seed
away from being a very different number. Dropping seed 44 gives a mean of 7.3%.

That does not retract the diagnosis: MF-DRO does waste HF queries on Hartmann,
and MF-MES's median is 0.0% on every benchmark. It does mean the magnitude
should always be quoted as "20.8% mean, 12.5% median, over 6-24 queries per
seed" and never as a bare 20.8%.

## The statistic that did NOT reproduce

The diagnosis also records "mean HF query score 0.336 vs 0.747". My measure —
(mean non-init HF y − mean init HF y) / sd(init HF y) — gives **11.844 vs
15.276** on the same runs. Those are not the same scale, so the recorded scores
come from a DIFFERENT normalisation (they sit in [0,1]; mine does not).

This is a definition mismatch, NOT a discrepancy in the data, and I am not
claiming the recorded figure is wrong. But the definition behind 0.336/0.747 is
not recoverable from what is written down, and the ordering it implies
(MF-DRO worse) does reproduce under my definition. Anyone quoting 0.336/0.747
should state the normalisation.

## What this means for the PRIMARY QUESTION

The waste the founding diagnosis names is a **Hartmann** phenomenon: 12.5%
median there against 3.2% on Borehole, 2.5% on Ackley, 0.0% on Currin.

Every demonstrated ROI benefit in this project is a **Borehole** phenomenon:
the 3.5-4.2 pt regret gain is Borehole-only, and h111 showed it fails on
Hartmann and Ackley at two tightness settings spanning 2x.

**These are disjoint.** The ROI works where the waste is smallest, and does not
work where the waste is largest. So the ROI, as specified in DRO Sec 4.2 and as
implemented here, is not addressing the diagnosis it was introduced to address.

That is a negative answer to the primary question's framing, not to the ROI
itself: the ROI does something real and reproducible on Borehole (still the
project's most reproducible quantity), but "stops MF-DRO wasting HF budget on
low-value regions" is not what it is doing, because on the benchmark where that
waste is large the ROI has no effect.

## Limitations

- n=5 seeds, no p-values. Hartmann's fractions rest on 6-24 queries per seed.
- waste_frac is coarse: it ignores how much worse a query is, and treats an
  informative-but-non-improving query as waste. It is used because it is the
  diagnosis's own statistic.
- One experiment (h83). Hartmann MF-DRO waste at other seeds (h87 has 47-51) is
  NOT included here and would be the natural extension.
