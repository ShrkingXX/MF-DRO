# H16 — joint MES wins on the fair axis. Both locked predictions PASS.

10 seeds x 2 rewards, 200 trajectories each, paired by seed.

| axis | `improvement` | `mes_entropy` | paired diff | seeds won | Wilcoxon |
|---|---|---|---|---|---|
| M3 `f_hf(x_0)` *(original gate axis)* | +0.1250 ± 0.0321 | +0.1648 ± 0.0319 | +0.0398 | 7/10 | p = 0.193 |
| **M1 best HF point** *(PRIMARY)* | +0.0863 ± 0.0426 | **+0.1826 ± 0.0375** | **+0.0962** | **8/10** | **p = 0.0195** |
| M1b best any point | **−0.0246** ± 0.0338 | +0.0706 ± 0.0375 | +0.0952 | 9/10 | p = 0.0645 |

**PRED 1 (PRIMARY) PASS** — `mes_entropy` beats `improvement` on terminal
trajectory quality, paired, p = 0.0195.
**PRED 2 PASS** — the original ordering does **not** reappear at n=10. H15 was
not a one-seed fluke.

## What this settles

The recorded justification for `rollout_reward="improvement"` —
*+0.191, z=2.63, p=0.0085, 9/10 positive* against `mes_entropy`'s *+0.129,
p~0.32* — is **not reproducible**. On 10 seeds of current code, `mes_entropy`
is ahead on **all three** axes, including the one originally used to reject it.

Note the sign on M1b: `improvement`'s return is **negatively** correlated
(−0.0246) with the best point its own trajectory visited. A return that
anti-correlates with trajectory quality is worse than uninformative as a
conditioning target.

## Honest limits

- p = 0.0195 is a **modest** effect. Three axes were computed; M1 was
  pre-registered as PRIMARY so no multiplicity correction is owed to it, but
  under a strict Bonferroni over three axes (0.0167) it would not survive.
  Reported so the reader can apply their own standard.
- Per-seed differences are not uniform (M1: two seeds negative, −0.048 and
  −0.036). The effect is consistent in direction, not overwhelming in size.
- This measures **teacher-quality correlation**, not regret. Whether the better
  conditioning signal produces better final regret is a separate question under
  the frozen evaluation, and is H17.

## The decision

Two independent lines now favour joint MES, and they were pre-committed in
opposite orders so neither was chosen after the fact:

1. **Signal health** (H14, decisive, not close): 0.0% vs 63.0% dead `rtg[0]`;
   100% vs 0.0% LF credit; CV 0.66 vs 1.96.
2. **Teacher quality on a fair axis** (H16, modest but pre-registered):
   +0.0962 paired, p = 0.0195.

`rollout_reward` should default to `mes_entropy`. That is a **method** change,
permitted by `PROTOCOL.md`; the evaluation is untouched.
