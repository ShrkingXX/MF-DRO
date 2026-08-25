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

---

## PRE-RESULT ADDENDUM: the dimensional prediction is refuted, and h64's outcome is now predicted

Measured directly — acquisition value found by a pool of size N, with y* held
fixed from an independent 4096-point Sobol reference so the comparison is fair
(y* depends on |X| in both location and scale):

| benchmark | dim | N=200 | N=600 | N=2000 | 600/200 | 2000/200 |
|---|---|---|---|---|---|---|
| Currin 2D | 2 | 0.3486 | 0.4566 | 0.4566 | **1.31x** | 1.31x |
| **Hartmann 6D** | 6 | 0.2275 | 0.2275 | 0.2730 | **1.00x** | 1.20x |
| Borehole 8D | 8 | 0.0335 | 0.0481 | 0.0566 | **1.44x** | 1.69x |

**The coverage/dimension hypothesis is refuted.** I predicted the gain would
order 8D > 6D > 2D. It orders **8D > 2D > 6D**. Currin, which 200 points should
cover densely, gains 31%; Hartmann gains **nothing at all** at N=600.

### Falsifiable prediction for h64, registered before its results

Widening 200 -> 600 buys Hartmann **zero** additional acquisition value
(1.00x). If acquisition quality is the channel through which POOL600 helped
Borehole, then:

- **Hartmann POOL600 should be NULL.** No acquisition gain -> no regret gain.
- **Currin POOL600 should be null on regret** despite its 1.31x acquisition
  gain, because the benchmark is saturated (every method inside 0.6%).

So h64 is predicted to come back null on both arms — which would make h61's
Borehole result **benchmark-specific for an acquisition-value reason**, not a
general teacher defect and not a dimensional one.

**If Hartmann POOL600 DOES improve regret**, the acquisition-value channel is
wrong: something else about a wider pool helps, and the mechanism is unexplained.
That outcome is more interesting than the predicted one and must not be
explained away.

This addendum was written with h64 at 0/6 and h61's REFINE arm at 0/3.
