# h214 analysis — CONFIRMATORY. 15/15 finals, 0 tracebacks.

Frozen metric imported from h83's `grid`; finals only; endpoint only. Borehole 42–46,
CTRL-K1 base.

## The dose

| random rollouts / member | arm | regret |
|---|---|---|
| 0 | CTRL-K1 | 11.59 |
| **20** | **MIXR-K1** | **7.17** |
| 60 | D60 | 9.82 |
| 120 | D120 | 13.47 |

| contrast | mean | se | better |
|---|---|---|---|
| D60 − MIXR-K1 | +2.65 | 2.42 | 2/5 |
| D120 − D60 | +3.64 | 3.02 | 2/5 |
| D120 − MIXR-K1 | +6.29 | 3.02 | 1/5 |

**P-DOSE (the user's hypothesis — monotone improvement) is NOT SUPPORTED.** More random
rollouts is monotonically **worse**: 7.17 → 9.82 → 13.47, and D120 is worse than using no
random rollouts at all.

**My registered lean was non-monotone** — "improvement from 0 to 20 then flattening or
reversing", with D60 ≈ MIXR-K1 and D120 ≥ MIXR-K1. The reversal is right and D120 ≥
MIXR-K1 holds; D60 ≈ MIXR-K1 does not (+2.65, outside the band). Scored as **partially
supported**: the direction was right, the shape was not.

**The effect has an interior optimum at ~20 random rollouts per member — a 1:1 ratio with
the MES half.** Standard errors are large (2.4–3.0) and seed 46 is an outlier in D60
(21.56), but three dose points move consistently.

## R-ROI: all-random with the ROI pool

| arm | regret |
|---|---|
| R-ROI (20 random, no MES) | **40.51** |
| h149 all-random, **uniform** pool | 43.94 (saturation floor) |
| CTRL-K1 | 11.59 |

All-random does not work even with the belief-tracking ROI pool. It sits just short of the
floor — the ROI pool buys ~3.4 points over a uniform pool, and nothing more. The MES half
is not optional.
