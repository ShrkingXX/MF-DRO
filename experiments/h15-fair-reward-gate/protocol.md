# H15 — the gate that disqualified joint MES could not have favoured it

## The finding that motivates this

The gate used to abandon `mes_entropy` and adopt `improvement` was:

> within-group **Spearman(`rtg[0]`, true `f_hf(x_0)`)**
> `mes_entropy` +0.129 (5/10 groups negative, p~0.32)
> `improvement` +0.191 (z=2.63, p=0.0085, 9/10 positive)

`x_0` is the trajectory's **first** query. So the gate asks: *does a high return
predict that the first query landed at a high true HF value?* That is a test of
**step-0 greediness**, not of trajectory quality.

- `improvement` is built from HF improvements, so it scores well **by
  construction**.
- A joint information-gain reward is *designed* to rate an exploratory first
  query highly when it shrinks `H[y*]`, even where `f_hf(x_0)` is mediocre.

The gate therefore penalises exactly the behaviour an information reward exists
to produce. It is the wrong axis for the comparison it was used to settle.

## Fair re-measurement (the intervention is the METRIC, not the method)

Same rollout batches, same seeds, same two rewards. Correlate `rtg[0]` against
**terminal trajectory quality** instead of step-0 value:

- **M1 (primary)**: `max_tau f_hf_true(x_tau)` over the rollout's HF queries —
  the best point the trajectory actually found.
- **M2**: true simple regret at the end of the rollout (`f(x*) − M1`), which is
  a monotone transform of M1 and is reported to make the sign explicit.
- **M3 (reference)**: the original `f_hf(x_0)` axis, re-run unchanged, so the
  old number is reproduced rather than merely cited.

10 groups, within-group Spearman, same aggregation (mean, SE, z) as the original
gate so the numbers are directly comparable.

## Locked predictions

1. **PRIMARY**: on M1, `mes_entropy` >= `improvement`. Specifically
   `mes_entropy`'s mean within-group Spearman on M1 is positive with p < 0.05.
2. **REPRODUCTION**: on M3, the original ordering reappears
   (`improvement` > `mes_entropy`). If it does **not**, the original gate is not
   reproducible and BOTH the old and new conclusions are suspect — report that
   and stop.
3. **NULL**: if `mes_entropy` fails M1 too, then it is genuinely the weaker
   conditioning signal despite its far healthier distribution, and
   `improvement` stands. That would be a real negative, not a void.

## Guards

- Prediction 2 is a **reproduction check on my own prior result** and is run
  first. A failure there invalidates the comparison regardless of M1.
- No regret comparison, no change to `PROTOCOL.md`, no method change. The only
  thing being changed is the yardstick.
- Both rewards are evaluated on identical seeds; `_rollout_gumbel_b` consumes
  RNG in `mes_entropy` mode, so fidelity draws diverge slightly — report the
  per-arm HF/LF counts so this is visible rather than assumed away.
