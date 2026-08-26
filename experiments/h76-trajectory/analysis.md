# H76 result — NULL fired. The advantage is late, and SF-DRO is *behind* early.

**CONFIRMATORY** against `protocol.md`. **Reproduction control PASS: 30/30 cells
bit-for-bit identical to h72.** This control is real — fresh runs of the same
cells, so it cannot pass by reading another experiment's files (h73's did).

## Verdict — both predictions wrong

| prediction | bar | result | verdict |
|---|---|---|---|
| PRIMARY | Hartmann crossing by iteration 12 of 25 | **iteration 18** | **NOT MET** |
| SECONDARY | gap non-decreasing over 2nd half | non-monotone | **NOT MET** |
| NULL | gap opens only in last third (>= 17) | crossing at 18 | **FIRED** |
| CONTROL | no early crossing on Borehole/Currin | never on either | **MET** |

## Mean regret curves (relative regret %, n=10 SF-MES, n=7-8 SF-DRO)

| Hartmann iter | 1 | 7 | 13 | 19 | 25 |
|---|---|---|---|---|---|
| SF-DRO | 76.05 | 57.99 | **26.38** | **16.95** | **7.70** |
| SF-MES | **66.08** | **48.51** | 32.85 | 23.47 | 21.17 |

| Borehole iter | 1 | 26 | 51 | 76 | 100 |
|---|---|---|---|---|---|
| SF-DRO | 48.66 | 20.16 | 15.02 | 14.48 | 14.38 |
| SF-MES | **40.46** | **16.82** | **14.88** | **13.73** | **12.76** |

| Currin iter | 1 | 17 | 34 | 50 | 66 |
|---|---|---|---|---|---|
| SF-DRO | 11.73 | 1.74 | 1.39 | 0.22 | 0.11 |
| SF-MES | **9.60** | **0.56** | **0.13** | **0.00** | **0.00** |

## The descriptive facts, which are not what I predicted

**1. SF-DRO is BEHIND SF-MES early on all three benchmarks** — including Hartmann,
where it eventually wins by 12.71 points. At iteration 1 it is worse everywhere
(76.05 vs 66.08; 48.66 vs 40.46; 11.73 vs 9.60), and it stays behind through
iteration 7 on Hartmann. The advantage is **not** early search. My PRIMARY assumed
it was, and was wrong.

**2. What differs on Hartmann is that SF-MES PLATEAUS and SF-DRO does not.**
SF-MES goes 32.85 -> 23.47 -> 21.17 over the last half, flattening. SF-DRO goes
26.38 -> 16.95 -> **7.70**, still descending steeply at the final iteration. On
Borehole and Currin both methods flatten together and SF-DRO never catches up.

**3. SF-DRO has not converged on Hartmann at budget exhaustion.** Its curve is
still falling at iteration 25. Whether more budget would widen the gap is
untested, and this experiment cannot answer it.

## What this does and does not license

It localises the effect: **late, not early**, and it is about SF-MES stalling
rather than SF-DRO starting well. That constrains any future mechanism — one that
explains an early exploration advantage is ruled out by fact 1.

It identifies no mechanism. SF-DRO differs from SF-MES in both the policy (learned
DT vs greedy MES) and the surrogate (10-model GP ensemble vs single GP), and this
experiment separates neither. Six mechanisms have been proposed and refuted in this
project; this one deliberately proposes none.
