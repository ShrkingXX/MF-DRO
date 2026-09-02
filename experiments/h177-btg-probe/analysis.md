# h177 — **P1 holds, and the reason is architectural.** BTG is structurally inert.

CONFIRMATORY, n=5, 357 probed iterations, Hartmann RANDOM-POOL.

| BTG | 20.0 | 24.0 | 26.0 | 28.0 | 30.0 | 32.0 | 36.0 |
|---|---|---|---|---|---|---|---|
| mean d(box centre) | 0.0822 | 0.0822 | 0.0822 | 0.0822 | 0.0822 | 0.0822 | 0.0822 |

**Spread across the whole BTG sweep: 0.0000 (0.0% of mean).** Not small —
*nothing*, to four decimals, across a sweep spanning well outside the visited
range (realised `btg_now` 26.08–30.52). The RTG axis in the same runs reproduces
h168 exactly: **9.0%** against h168's 8.9%, which validates the measurement.

**P1 holds. R1 fires.** Both conditioning inputs are inert; the DT's inference
output is a function of the state alone.

## An exactly-zero effect demanded an explanation, and there is one

BTG *is* wired — `btg_embed = Linear(1, H)`, `btg_ln = LayerNorm(H)`, interleaved
as one of four tokens per step. So this is not a missing connection.

The scalars are fed **raw** into `Linear(1→H)` followed by `LayerNorm`. That
composition **saturates**: LayerNorm fixes the output norm, and the direction
converges as |v| grows. Measured over 200 random weight draws (so a structural
property, not one seed):

| scalar | operating range | relative embedding change |
|---|---|---|
| **RTG** | 0.30 – 1.00 | **0.4869** ± 0.0369 |
| **BTG** | 26.1 – 30.5 | **0.0056** ± 0.0003 |
| reference | 1 → 10 | 0.6720 |
| reference | 20 → 36 | 0.0222 |

**The BTG embedding is 87× less responsive over its operating range than RTG's.**
RTG sits near the origin where the response is live; BTG sits deep in the
saturated regime — because its values are ~30 rather than ~1.

That predicts exactly what was measured: RTG produces a small but nonzero action
change (9%), BTG produces none (0%).

## The defect, and the fix

**BTG conditioning is inert not because the network learned to ignore it, but
because the architecture cannot represent differences in the range the runs
actually use.** The fix is standardising the scalars before embedding — the
same treatment the state block already gets via `use_state_standardization`.

## Labelling and caveats

The structural analysis is **EXPLORATORY**: it was done after seeing the zero,
not before. What makes it more than a story is that it makes a *quantitative*
prediction — RTG live, BTG saturated, ~87× apart — and both measurements match.

It analyses the module with random weights. A trained bias could shift the
operating point, and I have not measured the trained embedding directly. The
observed exact zero in the trained network is consistent with saturation but does
not prove the mechanism. **Measuring the trained `btg_embed`/`btg_ln` response
directly is the check that would settle it, and it is registered, not done.**
