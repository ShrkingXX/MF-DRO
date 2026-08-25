# H64 — does the teacher-pool fix generalise beyond Borehole?

**CONFIRMATORY. Protocol committed before any run.**

## Why

h61's POOL600 arm improved Borehole on **3/3** seeds — 23.7% -> 19.5% relative
regret — by changing one thing: the rollout teacher's candidate pool from 200
uniform points to 600. It is the first intervention in this project to move
Borehole at all.

That is either a general defect in MF-DRO's teacher or a Borehole-specific quirk.
The distinction decides whether it is a contribution toward the north star or a
footnote, and it is answerable with the same one-line change on the other two
benchmarks.

## Design

Hartmann 6D and Currin 2D, seeds 44/46/48, cost budget 200, `n_roi_candidates=600`.
Everything else identical to h57's MF-DRO. **BASE reuses h57's cells** (policy
code verified byte-identical at the h61 regression gate; each result records its
own commit hash).

6 jobs.

## Locked predictions

1. **PRIMARY**: POOL600 beats BASE on Hartmann on >= 2/3 paired seeds.
   Hartmann is the benchmark with headroom — MF-DRO is at 14.7% against
   MF-MES's 8.5%.
2. **CURRIN IS A NO-HARM CHECK, NOT A TEST.** MF-DRO is already at 0.0%
   relative regret there and every non-degenerate method is inside 0.6%. Currin
   cannot show improvement; it can only show that a wider pool does not *hurt*.
   Any Currin "win" is inside the noise of a saturated benchmark and will be
   reported as such, not as support.
3. **BOREHOLE-SPECIFIC**: no movement on Hartmann. Then h61's result is a
   property of that benchmark — plausibly its 8 dimensions, where 200 uniform
   points cover the space far more thinly than in 6D or 2D — and not a general
   teacher defect.
4. **HARMFUL ON HARTMANN**: live. h60 showed Hartmann's fidelity split is
   teacher-driven and unstable (81%/4%/28% across seeds); a sharper teacher
   could push it further toward one extreme, and h58 showed both extremes cost
   regret there.

## The dimensional prediction worth stating

If the mechanism is *coverage*, the effect should scale with dimension: strongest
on Borehole (8D), weaker on Hartmann (6D), absent on Currin (2D). A 200-point
uniform sample covers 2D densely and 8D essentially not at all. That ordering is
the discriminating signature, and it is checkable against h61's Borehole result
without any further runs.

## What this cannot settle

n = 3 per cell. Whether a still-larger pool keeps helping is untested and stays
untested — h47-variant-d's acquisition-value curve was still rising at 4000
points, but a 2000-point arm costs ~10x BASE per seed (~15 h), which is not
affordable here.
