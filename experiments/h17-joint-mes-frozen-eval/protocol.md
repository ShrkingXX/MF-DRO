# H17 — does the better conditioning signal produce better regret?

## Question

H14/H16 selected `rollout_reward="mes_entropy"` on signal quality. That is a
statement about the *training target*, not about performance. H17 asks the only
question that matters for the paper: **does it move final simple regret under the
frozen evaluation?**

## Design — frozen evaluation, unchanged

`PROTOCOL.md` is not modified. Hartmann 6D, seeds 42-51, `cost_budget=200`
post-init, identical initial design (`initial_hf=36`, `initial_lf=60`),
final simple regret at matched real cost.

**Baselines are reused, not re-run.** MF-MI-Greedy (0.5091 ± 0.1266) and
MF-GP-UCB (1.7934 ± 0.1223) come from `h1-leak-fix-validation` at the identical
seeds, identical initial design and identical cost budget, and neither baseline
contains a DT or a reward — the intervention cannot affect them. This is stated
in advance so it is not mistaken for a post-hoc convenience. Only the MF-DRO arm
runs: **10 jobs**, `num_workers=15 x threads_per_worker=1`.

Arms compared:

- MF-DRO / `improvement` — 0.5047 ± 0.0395 (already measured, h1)
- MF-DRO / `mes_entropy` — this run

## Locked predictions

1. **PRIMARY**: MF-DRO/`mes_entropy` mean final simple regret is **lower** than
   MF-DRO/`improvement`'s 0.5047, paired across the 10 shared seeds, Wilcoxon
   p < 0.05.
2. **FROZEN SUCCESS TEST** (the paper's actual bar): `mes_entropy` mean+SE <
   best-baseline mean−SE = **0.3825**. This is a high bar that `improvement`
   failed at 0.5442. Recorded in advance; a failure here is reported as a
   failure, exactly as H1's was.
3. **NULL**: if prediction 1 fails, then a demonstrably better conditioning
   signal does **not** buy regret — which, combined with H5/H8 (the score head
   barely reads `h`) and H6/H7 (retraining changes 18% of decisions for ~0
   regret), becomes a *coherent negative result*: MF-DRO's transformer is not
   functioning as a conditioned policy, and improving its conditioning target
   does not make it one. That is a publishable finding, not a failed experiment.

## Guards

- Report all 10 seed traces individually alongside mean ± SE, per `PROTOCOL.md`.
- Report wall-time and observed core utilisation.
- No seed, budget or configuration may be selected on the basis of results.
- **ETA discipline**: h1 ran 30 jobs in 6979 s wall. 10 jobs is *not* one third
  of that — wave structure dominates. I previously mis-estimated an extension by
  2.5x by ignoring exactly this. Estimate: **50-90 min**, stated as a range.
