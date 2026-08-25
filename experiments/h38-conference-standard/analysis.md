# H38 — would a conventional conference standard change the verdict? No.

**Status: EXPLORATORY** (user question, no protocol).

`PROTOCOL.md`'s success test (`mean+SE < best-baseline mean−SE`) is unusual. The
natural worry is that it is arbitrarily strict and that a normal reviewer
standard would pass. Tested on the same data:

| test (MF-DRO/joint-MES vs MF-MI-Greedy) | result | verdict |
|---|---|---|
| **frozen**: mean+SE < baseline mean−SE | 0.4481 vs 0.3825 | **FAIL** |
| paired Wilcoxon | p = 0.4316 | FAIL |
| paired t-test | p = 0.3518 | FAIL |
| unpaired Welch t | p = 0.4387 | FAIL |
| 95% CI on paired difference | [−0.3250, +0.1080] | contains 0 — FAIL |
| Cohen's `dz` | 0.311 | small |

**Every conventional standard fails too.** The frozen test is not the obstacle;
the effect is simply not detectable at n = 10. Power analysis: **82 seeds** would
be needed for 80% power against MF-MI-Greedy at this effect size (the ~40-seed
figure quoted elsewhere is for the *reward ablation*, a different and larger
effect — the two must not be conflated).

## Where the work does meet the standard

The **reward ablation** is conventionally significant on anytime performance:
MF-DRO/joint-MES vs MF-DRO/improvement, cost-weighted regret,
**Wilcoxon p = 0.0371, 9/10 seeds**. That is a defensible claim — but it is a
claim about *our own reward variants*, not about beating a baseline.

## Where it falls short of conference norms

- **One benchmark.** BO papers are normally expected to show 5–20 functions.
  Every claim here is Hartmann 6D only.
- **10 seeds.** BO papers typically run 20–50. Ours is fixed by the frozen
  protocol and was not extended, deliberately.
- **No baseline-tuning parity audit.** The baselines were taken as implemented;
  we did not verify they are tuned as carefully as MF-DRO.

## The honest position

Against baselines, on any standard, the method is **not shown to be better**.
The defensible claims are (i) the mechanistic negative result, which is
established by many independent measurements rather than one underpowered
comparison, and (ii) the reward ablation on anytime regret.
