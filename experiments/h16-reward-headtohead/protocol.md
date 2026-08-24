# H16 — properly-powered reward head-to-head on current code

## Why

H15 showed the gate that justified `rollout_reward="improvement"` does not
reproduce and in fact reverses — but at **n = 10 groups within one batch at one
seed**. That is enough to retract the old claim, not enough to choose a reward.
H16 is the powered version.

## Design

Full crossing: **10 seeds x 2 rewards** (`improvement`, `mes_entropy`), one
200-trajectory batch each, all other config identical and at current defaults.
Within-group Spearman(`rtg[0]`, y) per model-group, pooled across all seeds.

Axes (all three reported for every arm):

- **M3** `f_hf(x_0)` — the original gate's axis (step-0 greediness)
- **M1** `max_tau f_hf(x_tau)` over HF steps — terminal trajectory quality
- **M1b** same over all steps

Paired by seed, so each seed contributes one paired difference per axis.

Compute: 20 jobs, `num_workers=15 x threads_per_worker=1` (compute rule
satisfied). Thread caps set before numpy/torch import in the worker.

## Locked predictions

1. **PRIMARY (M1, the fair axis)**: `mes_entropy` mean within-group Spearman
   exceeds `improvement`'s, paired across seeds, Wilcoxon p < 0.05.
2. **REPRODUCTION (M3)**: the *original* ordering (`improvement` >
   `mes_entropy`) does **not** reappear. Stated as a directional prediction
   because H15 already reversed it; if it DOES reappear at n=10 seeds, then H15
   was a one-seed fluke, my retraction was premature, and I must say so.
3. **NULL / DECISION RULE**: if the two rewards are statistically
   indistinguishable on all three axes, the choice is **not** determined by
   teacher-quality correlation. In that case the decision falls to signal
   health, where `mes_entropy` dominates decisively and is *not* a close call
   (0.0% vs 63.0% dead `rtg[0]`; 100% vs 0.0% LF credit; CV 0.66 vs 1.96).
   Pre-committing to that rule now so it cannot be chosen after seeing results.

## Out of scope

No regret comparison and no change to `PROTOCOL.md`. This selects a reward on
signal quality; whether the selected reward improves final regret is a separate,
later, frozen-evaluation question.

## Guard

Report per-seed values, not just pooled means — H6 taught that pooled estimates
on this project change sign as n grows.
