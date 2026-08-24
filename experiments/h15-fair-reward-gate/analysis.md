# H15 — the reproduction check FAILED, and it reversed

Measured on current code, 200 trajectories, 10 model-groups, seed 44:

| axis | `improvement` | `mes_entropy` |
|---|---|---|
| **M3 `f_hf(x_0)`** (the ORIGINAL gate axis) | +0.0522, z=0.61, p=0.544, 4/10 neg | **+0.1726, z=1.98, p=0.0474, 3/10 neg** |
| M1 best HF point (primary) | +0.1739, z=1.55, p=0.121 | +0.1262, z=1.70, p=0.0888 |
| M1b best any point | +0.0637, z=0.95, p=0.343 | +0.1241, z=1.80, p=0.0725 |

Originally recorded: `improvement` **+0.191, z=2.63, p=0.0085, 9/10 positive**;
`mes_entropy` **+0.129, 5/10 negative, p~0.32**.

**Locked prediction 2 (reproduction) FAILS — and not marginally: the ordering
REVERSES.** Per protocol, "BOTH the old and the new conclusions are suspect;
report that and stop." No M1 interpretation is made, and no claim that
`mes_entropy` is the better reward is made from M3 either.

## Why it does not reproduce

`research-log.md:47-52` records that the reward switch was measured in the **same
commit** as a batch of compounding fixes — including the RTG-cap bug that gave
`improvement` "2 distinct values across 200 trajectories" (`rtg[0]` binarised to
{0,1}). The gate therefore compared the two rewards across a code state that has
since changed substantially: the leak fix (`7bcc3b8`), the RTG-cap fix,
candidate scoring, `rollouts_per_model=20`.

The code state the original number was measured on **no longer exists**.

## What this invalidates

> `findings.md`: "**Reward**: `mes_entropy` Spearman +0.129 ... -> regret-based
> `improvement` **+0.191, z=2.63, p=0.0085**, 9/10 positive."

That claim is **not reproducible on current code** and must not be cited as
justification for the current default. `improvement` is not established as the
better reward; on today's measurement it is not significant on any of the three
axes (p = 0.544 / 0.121 / 0.343).

This matters beyond the reward: `rollout_reward="improvement"` is the **current
default** in `dro_runner.py:439`, so the frozen-protocol headline result
(MF-DRO 0.5047 +/- 0.0395) was produced under a reward whose selection
justification does not reproduce. The headline number itself is unaffected — it
is what that configuration produced — but the *reason* that configuration was
chosen is no longer supported.

## Caveat on this measurement too

n = 10 groups within a **single** batch at one seed. The original may have
pooled more. This is enough to show non-reproduction; it is **not** enough to
crown a winner. A properly locked multi-seed head-to-head is H16.
