# H27 — does sliding-window inference improve REGRET?

## Why this is not a repeat of H11

H11 fed real history at inference and measured **argmax movement on a probe**
(0/12). It never measured regret. Two things have changed:

1. **The reward is now alive.** H11's DT-style RTG decrement arm **voided**
   because the improvement reward was zero everywhere. The default is now
   `mes_entropy`, whose returns are non-degenerate (0.0% dead vs 63.0%).
2. **Regret was never the measurement.** H17 showed the training-signal channel
   can move regret (0.5047 -> 0.4007) while conditioning stays 0/12, so a
   0/12 probe result does **not** entail "no regret effect". Inferring one from
   the other is exactly the error this project has repeatedly had to retract.

## Intervention (one variable)

`inference_context_k = 8`: at each real iteration the DT receives the **last 7
real (state, rtg, btg) triples plus the current one**, positions 0..7, readout at
the final state token. `inference_context_k = 1` (default) is the existing T=1
path, bit-for-bit unchanged.

Past RTG/BTG values are the ones **actually used** at those iterations --- no
fabricated decrement, no synthetic returns.

## GATE (run first, stop on failure)

- **G1**: the context genuinely reaches length 8 once 7 real iterations exist.
- **G2**: the K=8 run proposes **different queries** from the K=1 run
  (`max |x_t(K=1) - x_t(K=8)| > 1e-9`). If the proposals are identical, context
  does not reach the decision, there is nothing to compare, and H27 stops.

## Locked predictions (only if the gate passes)

1. **PRIMARY**: K=8 mean final simple regret is **lower** than K=1's
   0.4007 +/- 0.0475, paired across the 10 frozen-protocol seeds, Wilcoxon
   p < 0.05.
2. **FROZEN SUCCESS TEST**: mean+SE < 0.3825 (the bar both previous arms failed).
3. **NULL**: if regret is unchanged, this is the third independent channel
   (context, after state and RTG/BTG) shown not to matter, and it further
   supports the fixed-acquisition-rule account.

## Evaluation

`PROTOCOL.md` untouched. Hartmann 6D, seeds 42--51, cost budget 200 post-init,
identical initial design. Baselines reused from h1 as in H17 (they contain no DT
and cannot be affected). 10 jobs, `num_workers=10 x threads_per_worker=1`.

**ETA: 90--150 min**, stated as a range. H17's ten MF-DRO jobs took 98 min; this
adds an 8x longer inference sequence per proposal, which is a small share of
per-iteration cost (GP fitting and rollout generation dominate). Two prior ETAs
on this class were wrong in both directions, so the range is wide deliberately.
