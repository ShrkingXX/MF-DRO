# H79b — the "identity" is a coincidence of endpoints, not the same search

**CONFIRMATORY** against `protocol_b.md`. **Reproduction control PASS**: all four
re-run finals reproduce h72 bit-for-bit (4/4).

## Verdict — both predictions wrong, and the SECONDARY's failure is the finding

| prediction | bar | result | verdict |
|---|---|---|---|
| PRIMARY | divergent seeds agree >= 10 iterations | split at iteration **1** on both | **NOT MET** |
| SECONDARY | control seeds agree at **every** iteration | split at iteration **2** and **1** | **NOT MET** |
| NULL | split before iteration 10 | fired | see below |

**The NULL fired but its stated conclusion does not follow.** It read "then the
methods differ early and the reduction to single-fidelity EI is genuinely
incomplete". That framing assumed the *matched* seeds would show matched
trajectories. They do not — seeds 44 and 46 split at iterations 2 and 1 and still
finish bit-for-bit identical. The protocol's branches did not cover the outcome
that actually occurred.

## What is actually true

| seed | MI-Greedy start -> final | SF-EI@1000 start -> final | improvements |
|---|---|---|---|
| 44 | 28.141 -> **7.149** | 28.141 -> **7.149** | 3 vs 5 |
| 46 | **34.109** -> 6.759 | **36.597** -> 6.759 | 5 vs 6 |
| 45 | 34.014 -> 7.551 | 41.455 -> 8.776 | 1 vs 4 |
| 49 | 27.380 -> 8.945 | 44.487 -> 11.949 | 6 vs 8 |

The two methods **diverge from the first optimization query** on 3 of 4 seeds —
the starting regrets themselves differ — and take different numbers of incumbent
improvements to arrive. On seed 46 they begin 2.5 points apart and finish
identical to ten significant figures.

> **The 8/10 endpoint match is convergence to the same best point by different
> searches — not the same algorithm run twice.**

## The claim, restated a second time

- **h70 (n=3):** "MI-Greedy's advantage is *entirely* candidate pool size —
  SF-EI@1000 reproduces it exactly, seed for seed."
- **h79 (n=10):** the endpoint match holds on 8/10, not 10/10. "Entirely" dropped.
- **h79b:** the trajectories were never the same. What pool size buys is the
  **performance level**, not equivalence of search.

**Final form:** *on Borehole, giving single-fidelity EI a 1000-point candidate
pool is enough to reach MI-Greedy's final regret on 8 of 10 seeds, despite the
two following different search paths from the first query onward.*

That is still the cleanest mechanistic result here — the 4.72-point gap between
SF-MES and MI-Greedy really is closed by pool size alone — but it is a statement
about outcomes, and the earlier phrasing invited a stronger reading about
algorithms that the data never supported.

## Protocol-design lesson

The SECONDARY was the load-bearing check and I treated it as a formality. Had I
run only the PRIMARY, I would have concluded "early split -> genuinely different
methods" and missed that matched seeds split early too. **A control that only
checks the endpoint cannot detect two different paths to the same endpoint** —
which is precisely the thing h70's claim rested on.

## What this cannot settle

It locates the split and rules out trajectory identity; it does not explain why
8 of 10 endpoints coincide. Borehole only.

---

## EXPLORATORY addendum — why 8 of 10 endpoints coincide

h79b established the trajectories differ but left the endpoint coincidence
unexplained. This is descriptive, computed from curves already on disk, and
proposes no mechanism.

**When does each method reach its final value?**

| seed | | MI-Greedy final @ iter | plateau | SF-EI@1000 final @ iter | plateau |
|---|---|---|---|---|---|
| 44 | match | 7.149% @ **80** | 20 | 7.149% @ **96** | 4 |
| 46 | match | 6.759% @ **26** | 74 | 6.759% @ **31** | 69 |
| 45 | differ | 7.551% @ **2** | 98 | 8.776% @ 15 | 85 |
| 49 | differ | 8.945% @ 85 | 15 | 11.949% @ **35** | 65 |

**On the matching seeds the two methods reach the same value at different
iterations** — 80 vs 96, and 26 vs 31. They are not tracking each other. Each
arrives independently at the same best value, which is what one expects if both
are finding the same point, plausibly a dominant attractor reachable from the
shared 10-point initial design.

**On the divergent seeds, one method plateaus early at a worse value.** Seed 45 is
the extreme: MI-Greedy reaches its final value at **iteration 2** and does not
improve across the remaining 98 — a total incumbent stall — and still finishes
ahead of SF-EI@1000's 8.776%. Seed 49 is the mirror image, with SF-EI@1000
plateauing at iteration 35 and staying 3.00 points behind.

**Consequence for the claim.** The endpoint match is not evidence that pool size
makes the methods equivalent; it is evidence that **on most Borehole seeds, any
sufficiently-wide EI search from this initial design finds the same best point**.
That is a property of the benchmark as much as of the methods, and it is the
honest reason the 4.72-point gap closes.

**Not established:** that the coinciding endpoints are literally the same `x`.
These runs record regret curves, not query coordinates, so equal value is
observed and equal location is inferred. Confirming it would need the traces
re-run with coordinates recorded.
