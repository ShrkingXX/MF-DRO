# H115 — fill the missing MF-MES comparator at Borehole seeds 47-51

LOCKED BEFORE ANY RUN. **A comparator fill, not a hypothesis.** No prediction is
registered because nothing is being tested: these are baseline runs at seeds that
happen never to have been run.

## Why

A peer session observed that if h113's combined arm comes out additive, it would
land essentially on top of MF-MES on Borehole — the one benchmark where MF-DRO
genuinely loses. That comparison is worth having.

**It cannot currently be made at h113's seeds.** Borehole MF-MES exists at:

      42-46   h83   usable
      52-56   h89, h92   usable but not h113's seeds
      47-51   ONLY seed 48, and only in h57

h57 is one of the six experiments my code-drift audit flagged as spanning
behavioural changes in , so its seed-48 run is not comparable to current
code and is deliberately not used.

So h113 spans seeds 42-46 and 47-51, and **half of it has no comparator.** Five
runs fix that.

## Design

| | |
|---|---|
| benchmark | Borehole_8D |
| method | MF-MES, via h83's own worker (shimmed, output redirected) |
| seeds | 47, 48, 49, 50, 51 |
| runs | 5 |

Same code path that produced every other MF-MES figure in this project, so the
new runs are directly comparable to h83's and h92's.

## What is NOT being claimed

This experiment answers no question by itself. It exists so that h113's result —
whatever it is — can be read against MF-MES at matched seeds instead of against
MF-MES at *different* seeds, which is the error I flagged in a peer's work
earlier tonight and then had to avoid repeating in my own per-dimension table.

**h89 measured up to 3.67 points of seed-set difficulty difference on this exact
benchmark.** That is larger than several of the effects this project has spent
the night measuring, which is precisely why the comparator has to be seed-matched.
