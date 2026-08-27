# H89 analysis — 25 runs, 0 failures. One claim withdrawn, one survives at a third of its size.

Two independent confirmation arms at seeds 52-56, never used before, with both
treatments hardcoded and both controls run first.

## HF floor (Hartmann) — WITHDRAWN

| seed | control | floor | diff | control LF streak |
|---|---|---|---|---|
| 52 | 2.54% | 2.54% | +0.00 | 4 |
| 53 | 0.82% | 0.82% | +0.00 | 1 |
| 54 | 0.76% | **9.07%** | **+8.30** | 11 |
| 55 | 5.78% | 5.45% | -0.33 | 6 |
| 56 | 3.39% | 3.39% | +0.00 | 1 |

sd 2.08 -> **3.17** (P1 PRIMARY FAILED). Paired mean **+1.60 pts**, better 1/5
(P2 FAILED). P3 MET (collapse recurs). **P4's correlation is -0.84 -- the wrong
sign**: the floor harms the seeds that collapse hardest. h85's claim of a 66%
variance cut is withdrawn, and the mechanism is inverted rather than merely
unconfirmed.

## Teacher refinement (Borehole) — SURVIVES, at ~36% of its h85 size

| seed | control | refined | diff |
|---|---|---|---|
| 52 | 10.52% | 14.01% | +3.49 |
| 53 | 15.33% | 6.76% | -8.57 |
| 54 | 16.13% | 14.92% | -1.21 |
| 55 | 16.13% | 15.35% | -0.79 |
| 56 | 16.99% | 13.53% | -3.46 |

paired mean **-2.11 pts, better 4/5**

- **P5 PRIMARY MET.** Negative on 4/5. The effect is real.
- **P6 FAILED.** Required at least half of h85's -5.85, i.e. <= -2.9. It is
  -2.11, about 36% of the original. **Per the registered falsifier, h85's
  figure is hereby corrected: teacher refinement on Borehole is worth roughly
  -2 points, not -5.85.**
- **P7 MET.** 1.25x control wall-clock (0.772 -> 0.962 min/query), well inside
  the 2.5x bar and much cheaper than h85's in-flight 1.97x measurement.
- **P8 MET** (registered NEGATIVE). Refinement does NOT close the gap to
  MF-MES: 12.91% against 10.07% at these seeds.

## P8 was nearly reported wrongly, twice

First, Amendment 3 recorded before the arm finished that these seeds are +3.67
pts HARDER for MF-MES (10.07% vs 6.40% at h83's seeds) while the MF-DRO control
is only -0.80 pts different. Had refinement reproduced h85's -5.85, MF-DRO would
have landed near 9.2% and appeared to beat MF-MES -- a seed artifact, not a
method result. It did not reproduce, so the question is moot, but the
interpretation was fixed in advance rather than after.

Second, the P8 check initially printed CANNOT EVALUATE because it looked for the
comparator in h83's results directory while the matched-seed runs sat in h89's.
A path bug, not a data gap.

## Standing position after this experiment

- h83's headline STANDS: MF-DRO beats no baseline on any of four benchmarks.
- Two claims announced today were withdrawn (ROI Hartmann flip; HF floor
  variance). Both met their bars at n=5 on one seed set and died at fresh seeds.
- **One survives**: teacher refinement improves Borehole by ~2 points on 4/5
  fresh seeds at 1.25x cost. It does not make MF-DRO competitive there.
- The single mechanism claim that held all session is the one that predicted
  this: refinement moves the rollout teacher 4.6x more than the ROI, because the
  teacher's flat argmax over random candidates is the binding constraint.
