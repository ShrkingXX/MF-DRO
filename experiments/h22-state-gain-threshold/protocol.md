# H22 — how much state variation WOULD be needed?

## Why

Every result so far reports that the argmax does not move. H21 added that the
invariance survives a $5.7\times$ increase in coefficient variation, i.e. the
system is "far from the threshold, not marginally short". That phrasing is
qualitative. This experiment measures the distance.

## Intervention

Hold the trained network completely fixed. Amplify the *deviation* of each state
from the mean of the state set before it enters the embedding:

    s'(lambda) = s_bar + lambda * (s - s_bar)

`lambda = 1` reproduces the observed system exactly. Sweep
`lambda in {1, 2, 3, 5, 8, 12, 20, 35, 60, 100}` and, at each, measure the
fraction of 12 candidate pools on which the proposed argmax differs between two
genuinely distinct states (different ensemble-model blocks — never H5's
same-state swap).

Report `lambda*`, the smallest gain at which movement first exceeds 30% (the
same bar used throughout).

## What this does and does NOT establish

**Does**: quantify the gain the conditioning channel would need before the state
could change a decision. It characterises the learned map.

**Does NOT**: propose amplification as a fix, or make any claim about regret.
At large `lambda` the states are **out of distribution** — the same confound
that made an earlier RTG sweep uninterpretable until it was re-run in-band. A
threshold found at large `lambda` is a statement about the map's gain, not about
a usable intervention, and will be reported with that caveat attached.

## Locked predictions

1. **PRIMARY**: `lambda*` exists and is `>= 10`. This substantiates "far from
   threshold" with a number.
2. **ALTERNATIVE**: if `lambda* < 5`, the system is closer to the threshold than
   H21's robustness result suggested, and the paper's "far short" phrasing must
   be softened.
3. **NULL**: if no `lambda <= 100` produces movement, the map is invariant to
   state direction over two orders of magnitude, which is stronger still and
   should be reported as such.

## Guard

Also report, at each `lambda`, the pairwise cosine of the coefficient vectors,
so that a movement result can be checked against a real change in `w` rather
than numerical noise.

Single process, 1 thread. `PROTOCOL.md` untouched.
