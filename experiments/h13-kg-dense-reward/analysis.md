# H13 analysis — gate PASSES; decomposition rule was badly chosen

| arm | dead `rtg[0]==0` | LF nonzero | `rtg[0]` mean / CV | `rtg[0]` negative |
|---|---|---|---|---|
| baseline `improvement` | 63.0% | 0.0% | 0.1022 / 1.964 | 0.0% |
| A `topk=1` clipped (=H12) | 4.5% | 27.2% | 0.1396 / 1.374 | 0.0% |
| B `topk=1` SIGNED | **0.0%** | **100.0%** | −0.0204 / −12.934 | 67.0% |
| C `topk=5` SIGNED | **0.0%** | **100.0%** | 0.0676 / 3.677 | 38.5% |

**GATE PASS** (G1 0.0% < 20%, G2 100.0% > 50%).

## A measurement bug in my own gate, found and fixed before interpreting

The first run reported G1 FAIL at 38.5%. That was wrong. My locked criterion
reads `rtg[0] == 0`, but the implementation tested `rtg[0] <= 1e-12`, which is
`(== 0 OR < 0)`. Those coincide only for a non-negative reward — correct for the
`improvement` baseline, wrong the moment `kg_signed` makes RTG able to go
negative, where a negative `rtg[0]` is *informative*, not dead.

The tell was that `dead_frac` and `neg_frac` were **exactly equal** in both
signed arms, meaning no genuinely-zero trajectories existed. I fixed the metric
to match the written criterion and **re-ran** rather than reinterpreting the old
numbers in place.

## My decomposition rule was unusable — it saturated

The protocol said to compare arms by LF-nonzero fraction. Both B and C hit
**100.0%**, so that metric cannot discriminate, and the script's auto-verdict
("clip was the whole story; drop `kg_topk`") is **not supported**. What actually
separates them:

- **B** has mean `rtg[0]` = **−0.0204** — the average rollout makes the belief
  *worse* — with CV −12.9, i.e. a mean indistinguishable from zero with huge
  spread. As a return target that is close to unusable.
- **C** has mean +0.0676, CV 3.677, and 38.5% negative.

So `kg_topk` is **not** inert; it is what keeps the signal centred on the useful
side. The lesson is on me: a discriminating metric must be chosen so it cannot
saturate in the arms being compared.

## Side-effect check (locked prediction 3)

Fires for **B** (67.0% negative > 50% threshold), not for **C** (38.5%). The
floored-dynamic target interacts badly with a mostly-negative signal, so B is
disqualified independently of the decomposition.

## Superseded before use

H13 passed its gate but **no run was ever launched on this reward**. A spec of
the codebase (see H14) showed the existing `mes_entropy` mode already computes
the joint set-level information gain, which is a better-motivated target than a
per-step cumulative reward. `kg_incumbent`/`kg_signed`/`kg_topk` are retained as
working ablation flags, not as the chosen reward.
