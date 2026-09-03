# h194 Stage 0 — **G-PASS. h27 is overturned on current code.**

**CONFIRMATORY** against the gate registered before the runs. Two short runs, same seed,
ROI-Q10, differing **only** in `inference_context_k` (1 vs 8).

## Control check first

Iteration 0 has **no history**, so both arms run at ctx=1 and must agree bit-for-bit. If
they did not, any later difference would be RNG divergence rather than the window.

| iter 0 | max \|Δx\| | **0.0000** — **CONTROL CHECK PASS** |
|---|---|---|

## Result

Per-iteration difference in the **unit box**:

| iter | max \|Δx\| | L2 |
|---|---|---|
| 0 | 0.0000 | 0.0000 |
| 1 | 0.1323 | 0.2340 |
| 2 | 0.1517 | 0.2044 |
| 3 | 0.1161 | 0.1603 |
| 4 | 0.1056 | 0.1862 |
| 5 | 0.2497 | 0.3055 |
| 6 | 0.2707 | 0.3897 |
| 7 | 0.1585 | 0.2672 |

**7/8 iterations differ.** Mean L2 from iteration 1 on: **0.2496**, max **0.3897**.

On this project's own scales that is large:

| reference | value | the window's effect |
|---|---|---|
| swapping the teacher's whole decision **rule** (h180) | 0.044 | **5.7×** |
| changing only the random **seed** (h180) | 0.82 | 0.30× |

**The window moves the DT's query 5.7× more than replacing its teacher's entire decision
rule does.**

## h27 is stale, confirmed. And h185 predicted this.

h27 measured these bit-identical (max |Δx| = 0.000e+00). **That does not replicate.** The
33 intervening commits — three behaviour-changing, including the Aug 27 ROI-resolution fix
whose own message says it "confounded every ROI A/B" — are enough to void it.

The tension recorded when this gate was written resolves **in h185's favour**: a
per-timestep constant predictor, read out at position T−1 instead of 0, *must* emit a
different constant, and it does. h27's null was the anomaly.

## Stage 1 is live — and the mechanism now predicts it FAILS

Stage 0 shows the window **changes** the decision. That is not the same as improving it,
and the mechanism makes a sharp, falsifiable prediction in the opposite direction from
the arm's motivation:

- With a window, the readout moves to position **T−1**, so the DT emits its **late-τ**
  constant instead of its **τ=0** constant.
- h171/h173 established that the **τ=0 step is the one that matters**: HEAD-MES
  (acquisition at τ=0 only) works — 16.96 Borehole / 25.16 Hartmann — while TAIL-MES
  (acquisition at τ=1…7, random first) **fails** — 43.94 / 46.45.
- Emitting a late-τ constant is therefore closer to TAIL than to HEAD.

> **The mechanism predicts P3 (the window HURTS), not P1.** Recorded before Stage 1 runs.
> The registered gate is unchanged; what is added is that the outcome the arm was
> *motivated by* (P1) is the one the current account says will not happen.

**P1 would still retract more than P3 confirms**: it would force an explicit exception
into "input-side fixes cannot help", which currently unifies three Phase-1 nulls.

## Cost

Two short runs, ~6 minutes. Stage 1 is 10 worker-hours and is **not** launched — the
autoresearch loop is paused and this is the expensive half.
