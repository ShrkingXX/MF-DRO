# h168 — **R1 fires.** The inference conditioning is EXONERATED.

CONFIRMATORY, n=5, 357 probed iterations. Hartmann, RANDOM-POOL arm.

## The probe is exactly read-only

Beyond the probe-OFF bit-identity gate, the probed run was compared against
h149's unprobed run of the same policy and seeds:

| seed | h149 regret | h168 regret | x_t_trace identical |
|---|---|---|---|
| 42–46 | 2.610002, 2.581931, 0.753135, 2.598508, 2.277771 | identical | **True, all five** |

Saving and restoring the RNG around the sweep worked: the arm is bit-identical
to its unprobed twin. Any effect below is a property of the network, not of the
measurement.

## The result

| conditioning RTG | mean d(box centre) | sd | n |
|---|---|---|---|
| 0.00 | 0.0810 | 0.0445 | 357 |
| 0.02 | 0.0810 | 0.0444 | 357 |
| 0.05 | 0.0811 | 0.0444 | 357 |
| 0.10 | 0.0812 | 0.0444 | 357 |
| 0.20 | 0.0816 | 0.0444 | 357 |
| **0.30** *(the realised target)* | **0.0822** | 0.0444 | 357 |
| 0.50 | 0.0837 | 0.0450 | 357 |
| 0.75 | 0.0861 | 0.0462 | 357 |
| 1.00 | 0.0884 | 0.0476 | 357 |

- **P1 HOLDS.** At the real target the emitted action is 0.0822 from the box
  centre — collapsed (Hartmann is 6D, where the far corner is √6/2 ≈ 1.22).
- **P2 FAILS.** At in-support RTG=0.02 it is 0.0810. Ratio **0.986**; P2 needed
  ≥ 2.0.
- Across the **entire** sweep from 0 to 1 the action moves **0.0074** — 8.9% of
  its own mean — and moves *away* from the centre as RTG rises, the opposite
  direction to the hypothesis.

**The emitted action is essentially independent of the conditioning.**

## What this closes

R1 as pre-registered: *"the conditioning is NOT the cause. The collapse would be
unconditional, and the suspect becomes the trained network itself."*

This **closes the conditioning line evaluably** — which is what h148 set out to
do and could not, because the statistics it needed were never serialised. The
line is now shut on a direct measurement rather than a proxy, and it will not be
reopened without new evidence of a different kind.

The smoke test's early signal, recorded before the arm ran, predicted exactly
this (spread 0.028 at 5 iterations; 0.0074 at full length).

## The contrast, from existing data

Per-query distance from the box centre on Hartmann:

| arm | mean d(centre) | centroid d |
|---|---|---|
| control (works) | **0.6503** | 0.6287 |
| h165 UCB-LOC (works) | **0.6194** | 0.5896 |
| ORACLE (fails) | 0.1140 | 0.0363 |
| RANDOM-POOL (fails) | **0.0830** | 0.0237 |

So h167's Borehole finding replicates on Hartmann, and the probe's 0.0822 sits
exactly on RANDOM-POOL's own 0.0830.

## P3 was not run, and why it is now nearly moot

P3 (probe a working arm and show no gap) was not launched — compute was at
14/15. With P2 failing at ratio 0.986 and a whole-sweep spread of 8.9%, there is
**no gap in the failing arm for the control to contrast against**. P3 would now
test only whether the control is *also* insensitive, which is a different and
much weaker question. Recorded as unrun rather than quietly dropped.

## What the suspect becomes

The network fits its teacher well during training (h167c: 2.5–4× better than any
constant) and emits the box centre at inference regardless of RTG. Since RTG is
now excluded, the remaining differences between training and inference are the
**state** (simulated rollout states vs real trajectory states), the BTG
conditioning, and the timestep/context. State-distribution shift is the natural
next suspect, and h169 is registered to decompose it.
