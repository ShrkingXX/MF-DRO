# h178 — **P1 holds. The saturation account is confirmed on trained weights.**

CONFIRMATORY, n=5, 357 iteration-samples.

| iteration | rtg_resp | btg_resp | ratio |
|---|---|---|---|
| 0 | 0.5174 | 0.0054 | 96.0× |
| 23 | 0.5170 | 0.0055 | 93.8× |
| 42 | 0.5169 | 0.0056 | 92.1× |
| 61 | 0.5394 | 0.0057 | 93.8× |
| 94 | 0.5428 | 0.0064 | 84.5× |

| | rtg_resp | btg_resp | ratio |
|---|---|---|---|
| **all iterations** | **0.5216** ± 0.0296 | **0.0056** ± 0.0004 | **92.9×** |
| last quarter (most trained) | 0.5412 | 0.0062 | 86.9× |
| **h177 random-weight prediction** | 0.4869 | **0.0056** | ~87× |

## The part that settles it

h177's smoke reading was dismissed as near-tautological, because a barely-trained
module must agree with a random-weight calculation. **The full run answers that
objection directly: the response barely moves across the whole training
trajectory.** `btg_resp` goes 0.0054 → 0.0064 over 94 iterations — a 19% rise
from a base of essentially zero — and never becomes responsive. The ratio stays
between 84× and 96× throughout.

**Training does not rescue BTG's responsiveness.** The trained `btg_resp` of
0.0056 matches h177's random-weight prediction of 0.0056 exactly, and the
last-quarter ratio (86.9×) matches the predicted ~87× almost exactly.

## Status change

h177's architectural explanation moves from **EXPLORATORY to measured on the
trained network**. R2 fires; R1 does not. The chain is now:

1. BTG is wired but the emitted action is *exactly* invariant to it (h177, 357
   iterations, spread 0.0000).
2. The trained `btg_embed`+`btg_ln` response over BTG's operating range is
   0.0056 — 93× less than RTG's — and stays there through training (h178).
3. Z-scoring restores it to 1.8817, a 336× increase (module-level).

**Still not established:** that making the channel responsive *improves* the
method. That is h179, and its P3 (responsiveness degrades regret) remains live —
six arms on this front paired a collapsed conditioning target with good
performance.
