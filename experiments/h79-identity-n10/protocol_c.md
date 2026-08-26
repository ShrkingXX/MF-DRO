# H79c — are the coinciding endpoints literally the same point?

**CONFIRMATORY.** Locked before any h79c number exists.

## Why

h79b's addendum states that on the matching seeds both methods "plausibly" find
the same point, inferred from their final regrets agreeing to ten significant
figures. **That is an inference from equal value, not a check of equal location**
— and lesson 29, recorded one tick ago, says the strength of the word must match
the strength of the check. This converts the inference into a measurement.

The runs cost seconds to a minute. There is no reason to leave an inference in
the record when the check is this cheap.

## Design

Re-run MI-Greedy and SF-EI@1000 on Borehole seeds **44, 46** (endpoints match)
and **45, 49** (endpoints differ), recording **query coordinates**. Compare the
argmax-HF point each method finishes with.

**Built-in control:** the re-runs must reproduce the already-recorded
`final_regret` bit-for-bit, or the coordinates belong to different trajectories
than the ones analysed.

## Locked predictions

1. **PRIMARY.** On both matching seeds, the two methods' best-found `x` agree to
   **within 1e-9 in every coordinate**. Borehole is a smooth 8-D function; two
   different points producing identical `f` to ten significant figures would be a
   coincidence of a kind that does not happen by chance.
2. **SECONDARY.** On the divergent seeds the best-found `x` differ — confirming
   the regret difference reflects genuinely different points, not a bookkeeping
   artifact.
3. **NULL / SURPRISE.** The matching seeds have **different** best `x` with equal
   `f`. That would mean Borehole has exactly-equal-valued optima being found
   independently, which is far more interesting than the current account and
   would need its own investigation.

## What this cannot settle

Four seeds, one benchmark. It checks where each search *ended*, not the path it
took — h79b already established the paths differ from the first query.
