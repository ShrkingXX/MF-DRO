# H93 — are Currin's and Ackley's deficits real, or seed artefacts too?

LOCKED BEFORE ANY RUN. Not launched: compute is at 15/15 with H90.

## Why this closes the last gap

h83 concluded MF-DRO beats no baseline on any of four benchmarks. At n=10:

  Hartmann_6D   deficit VANISHES (5/10, median +0.22)   -- H91
  Borehole_8D   deficit is REAL  (8/10, median +8.30)   -- H92
  Currin_2D     never measured at a second seed set
  Ackley_10D    never measured at a second seed set

If Currin's and Ackley's deficits also vanish, **MF-DRO loses exactly ONE
benchmark of four**, not four. That is a materially different claim from the one
h83 reported and the one the published report currently makes, and it changes
what the method's weakness actually is.

Their h83 margins are small, which is precisely why they are worth checking:

  Currin_2D    MI-Greedy 0.00%  vs MF-DRO 0.01%   (margin 0.01)
  Ackley_10D   SF-DRO    3.43   vs MF-DRO 3.83    (margin 0.40)

Both are far tighter than Hartmann's 1.37, and Hartmann's did not survive.

## Design

| | |
|---|---|
| benchmarks | Currin_2D, Ackley_10D |
| seeds | 52, 53, 54, 55, 56 -- the set already used by H89/H91/H92 |
| arms | MF-DRO (no intervention), and the BEST BASELINE per benchmark |
| best baseline | Currin: MF-MI-Greedy. Ackley: SF-DRO. |
| runs | 20 (4 arms x 5 seeds) |

The comparator is the benchmark's OWN best baseline from h83, not MF-MES
throughout -- MF-MES wins neither of these. Using MF-MES here would test a
different and easier question.

## Metric

h83's frozen metric via its own sr_curve/grid: SR at cost 200. Ackley reports
ABSOLUTE simple regret (its optimum is exactly 0, so the relative metric divides
by zero).

## Predictions

**P1 (Currin).** MF-DRO does NOT beat MI-Greedy at seeds 52-56. Registered as
NEGATIVE: MI-Greedy scores 0.00% at 42-46, a ceiling that leaves nothing to win,
and the ROI made Currin WORSE (+0.11, 0/5) which suggests the benchmark is
already saturated for every method.

**P2 (Ackley).** GENUINELY UNCERTAIN. The h83 margin is 0.40 points against a
per-method spread of ~1.0, i.e. well inside the noise -- the same relationship
that made Hartmann's 1.37-point margin evaporate. I am not predicting a
direction.

**P3 (the one that matters).** At least one of Currin or Ackley shows a deficit
that does NOT replicate. Registered because Hartmann's did not and both of these
margins are smaller than Hartmann's.

## What each outcome means

  - BOTH VANISH: MF-DRO has ONE real deficit (Borehole), not four. The h83
    headline needs restating, and the corrected problem statement -- "on
    benchmarks whose optimum lies on the domain boundary" -- becomes the whole
    story rather than one benchmark's explanation.
  - BOTH HOLD: h83's headline stands as reported and Hartmann was the exception.
  - MIXED: report per benchmark; no single headline is available.

## Gate

Launch only when H90 completes and compute shows 0 workers on two samples taken
a few seconds apart. Requires 15 slots for the MF-DRO arms; the baselines are
cheap (MI-Greedy ~0.5 min, SF-DRO ~25-45 min).
