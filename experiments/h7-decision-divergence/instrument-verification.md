# H7 instrument verification — done BEFORE trusting the result

## Why this was necessary

The H7 smoke test reported `dist = 0.000000` on every record — the live policy
and its iteration-5 snapshot selecting **bit-identical** proposals. Two very
different things produce that output:

1. **Real**: the extra training genuinely does not change which candidate wins.
   (Expected under H5: the score head ranks by candidate features, so small
   weight changes should not flip the argmax. Under this reading `dist` is
   *exactly* zero, not approximately — same candidate means identical coords.)
2. **Trivial**: the "snapshot" is not independent — `deepcopy` aliased storage,
   or the live model never trained after iteration 5 — so I would be comparing a
   model to itself and reporting a profound-sounding null.

The measurement alone cannot separate these. Explanation 1 is a headline result;
explanation 2 is an artefact. So it had to be checked directly.

## Check and result

    params sharing storage with the live model : 0 / 93     (deepcopy is clean)
    params that CHANGED after iteration 5      : 77 / 93     (training really ran)
    max abs weight delta                       : 5.44e-03

    score-head deltas:
      coef_head.0.weight 4.58e-03   coef_head.2.weight 4.79e-03
      coef_head.0.bias   1.95e-03   coef_head.2.bias   7.74e-04
      bias_head.0.weight 3.69e-05   bias_head.2.weight 1.07e-04

**VERIFIED**: the snapshot is independent AND the live model genuinely diverged
from it.

## What this makes the finding mean

The weights that produce the score — `coef_head` — changed by ~4.6e-03, and the
selected candidate did not change **at all**. So this is not "nothing happened":
it is *measurable training occurred, and it moved the decision zero*.

That is a strictly stronger statement than H5's. H5 showed the score head is
insensitive to **which `h`** it receives. This shows it is also insensitive to
**its own trained weights changing** — the argmax is pinned by the candidate
features regardless of what the network learns.

It also explains H6's null mechanically rather than statistically: freezing the
DT could not change regret much because continued training was not changing the
decisions to begin with.
