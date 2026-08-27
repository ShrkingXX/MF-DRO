# H92 — does MF-DRO's BOREHOLE deficit survive a change of seed set?

LOCKED BEFORE ANY RUN. 5 runs, MF-MES on Borehole at seeds 52-56.

## Why

H91 showed MF-DRO's Hartmann deficit against MF-MES is not stable: +1.37 pts at
seeds 42-46, -4.40 at 52-56, and indistinguishable when the two sets are pooled
(n=10, 5/10 seeds, median +0.22). The apparent Hartmann gap was a property of
seeds 42-46.

**Borehole is where MF-DRO's real deficit lives** -- 15.82 against MF-MES's 6.36,
a 9.5-point gap, far larger than Hartmann's ever was. It has NEVER been measured
at any seed set other than 42-46. If it too collapses at fresh seeds, then
MF-DRO's headline weakness is an artefact of one seed draw across the board, and
the h84-h90 intervention programme was aimed at a phantom. If it holds, Borehole
is the genuine deficit and everything learned there stands.

The concurrent session's H89 already produced MF-DRO controls at Borehole seeds
52-56. Only MF-MES is missing. Five runs, ~5 min each.

## Design

| | |
|---|---|
| benchmark | Borehole_8D |
| arm | MF-MES only |
| seeds | 52, 53, 54, 55, 56 |
| comparator | H89's Borehole CONTROL (MF-DRO, no intervention) at the SAME seeds |
| config | identical to h83's MF-MES |

## Metric

h83's frozen metric via its own sr_curve/grid: SR at cost 200, relative regret.

## Predictions

**P1. MF-DRO's Borehole deficit against MF-MES persists at seeds 52-56 --
MF-MES better on >= 4/5 seeds.**

I expect this to HOLD, unlike the Hartmann case. Reasons, so the call can be
judged rather than trusted:
  - Hartmann's deficit was 1.37 pts against a per-method spread of 3-6 pts, i.e.
    well inside the noise. Borehole's is 9.5 pts against a spread of 2.4.
  - Borehole's mechanism is understood and benchmark-intrinsic: its optimum sits
    on the domain boundary in all four dimensions carrying 99.6% of the variance,
    and MF-DRO's teacher takes a flat argmax over uniform candidates that
    essentially never reach several bounds at once. That property does not depend
    on the seed.
  - MF-DRO's Borehole spread at 42-46 was sd 2.36 with every seed between 12.93
    and 19.19 -- no collapse-prone outliers of the kind that made Hartmann's
    seeds 42-46 unrepresentative.

**P2 (NEGATIVE). The pooled n=10 Borehole comparison does NOT become
indistinguishable**, unlike Hartmann's. Registered explicitly because the
Hartmann result makes the opposite tempting to expect.

## What each outcome means

  - DEFICIT PERSISTS: Borehole is MF-DRO's genuine weakness, the boundary
    mechanism is the explanation, and refinement (-5.85 pts there) is aimed
    correctly. The h84-h90 programme retains its target.
  - DEFICIT COLLAPSES: MF-DRO has no stable deficit on any benchmark tested at
    more than one seed set, and the entire intervention programme was tuned
    against seed-draw noise. That must be stated as prominently as the diagnosis.
