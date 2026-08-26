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
