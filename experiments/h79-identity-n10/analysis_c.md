# H79c — the coinciding endpoints are literally the same point. Verified.

**CONFIRMATORY** against `protocol_c.md`. **Reproduction control PASS**: all 8
re-runs reproduce the analysed runs' `final_regret` bit-for-bit, so these
coordinates belong to the same trajectories h79/h79b examined.

## Verdict — both predictions met

| prediction | bar | result | verdict |
|---|---|---|---|
| PRIMARY | matching seeds: best `x` identical within 1e-9 | **max \|dx\| = 0.000e+00** | **MET** |
| SECONDARY | divergent seeds: best `x` differ | 2.19e+04 and 4.11e+04 | **MET** |
| NULL | matching seeds have different `x` with equal `f` | — | did not fire |

| seed | | MI-Greedy best `y` | SF-EI@1000 best `y` | max \|dx\| |
|---|---|---|---|---|
| 44 | match | 287.442918962 | 287.442918962 | **0.000e+00** |
| 46 | match | 288.650965262 | 288.650965262 | **0.000e+00** |
| 45 | differ | 286.198665477 | 282.406266857 | 2.19e+04 |
| 49 | differ | 281.884095188 | 272.583252528 | 4.11e+04 |

## What this settles

h79b's addendum said the matching seeds "plausibly" find the same point,
inferred from equal `f`. **They do — exactly, to every coordinate.** The
inference is now a measurement, which is what lesson 29 asks for.

The complete, verified account of h70's original observation:

1. The two methods **diverge from the first optimization query** (h79b) — the
   starting regrets themselves differ.
2. On **8 of 10** Borehole seeds they nonetheless finish at the **same best
   point** (h79; verified as coordinate-identical on 2 of those 8 here).
3. They reach it at **different iterations** (80 vs 96, 26 vs 31) — independent
   arrival, not one tracking the other.
4. On the other **2 of 10**, one method plateaus early at a genuinely different,
   worse point — the coordinates differ by 2.19e+04 and 4.11e+04 in raw domain
   units.

> **Borehole has an attractor that a sufficiently wide EI search reliably finds
> from this initial design, by different routes, most of the time.** That — not
> algorithmic equivalence — is why giving single-fidelity EI a 1000-point pool
> reaches MI-Greedy's regret on 8 of 10 seeds, and why the 4.72-point gap closes.

## Scope

Two of the eight matching seeds were checked at coordinate level, not all eight.
Four seeds, one benchmark. It checks where each search ended, not the path.
