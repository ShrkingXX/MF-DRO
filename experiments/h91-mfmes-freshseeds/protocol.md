# H91 — is MF-DRO's deficit against MF-MES itself seed-set specific?

LOCKED BEFORE ANY RUN. 5 runs, ~2 min each.

## Why this is the highest-value unrun experiment

This project's founding diagnosis -- MF-DRO wastes 20.8% of its high-fidelity
budget and scores 0.336 against MF-MES's 0.747 -- was measured entirely on seeds
42-46. The concurrent session's H89 control arm shows the same configuration on
seeds 52-56 scoring **0.685 mean, 4.2% below the initial design, 24 HF queries**
instead of 12. MF-DRO is not notably wasteful there.

But the comparison that matters is against MF-MES, and **MF-MES has never been
run at seeds 52-56**. Without it we know MF-DRO improved in absolute terms and
nothing about whether the GAP closed. The seed sets do not move the two methods
together: h87 measured MF-MES at 9.89 on seeds 47-51 against 6.62 on 42-46, i.e.
WORSE, while MF-DRO is BETTER at 52-56.

Five runs settle it. MF-MES is ~2 min/run on Hartmann.

## Design

| | |
|---|---|
| benchmark | Hartmann_6D |
| arm | MF-MES only |
| seeds | 52, 53, 54, 55, 56 |
| comparator | H89's MF-DRO CONTROL runs at the SAME seeds (already complete) |
| config | identical to h83's MF-MES: KO surrogate, budget 200 post-init, same initial design |

No new MF-DRO runs are needed -- H89 already produced them, and reusing a
different arm's control at matched seeds is exactly what a paired comparison
requires.

## Metric

h83's frozen metric via its own sr_curve/grid: SR at cost 200, relative regret.

## Predictions

**P1 (PRIMARY, registered as GENUINELY UNCERTAIN).** MF-DRO beats MF-MES on
>= 3/5 of seeds 52-56.

I am not predicting this either way, and the reason is on record: I have
predicted five mechanism outcomes in this investigation and been wrong on all
five, four by underestimating an intervention and one (h87's P1) by
overestimating. The relevant facts point in opposite directions -- MF-DRO's
absolute performance at 52-56 is 3x better than at 42-46, but MF-MES's own
performance also swings by seed set and in the opposite direction at 47-51.

**P2.** MF-DRO's paired deficit against MF-MES is SMALLER at seeds 52-56 than
the 1.37 pts measured at 42-46. This is the quantity the founding diagnosis
actually asserts, and it is the one that determines whether the ROI programme
was addressing a general problem or a seed-set artefact.

## What each outcome means

  - If MF-DRO beats or matches MF-MES at 52-56: the founding diagnosis describes
    seeds 42-46, not MF-DRO, and every intervention in h84-h90 was tuned against
    an unrepresentative failure. That must be stated as prominently as the
    diagnosis was.
  - If the deficit persists at similar size: the diagnosis generalises, MF-DRO is
    genuinely behind, and the intervention programme was aimed correctly even
    though the waste metric was inflated by seed choice.

Both outcomes are informative. Neither is a failure of this experiment.
