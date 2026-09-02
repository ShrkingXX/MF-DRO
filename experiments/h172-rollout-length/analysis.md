# h172 — **R1 fires.** rollout_length=1 beats the control at 6.2× the speed.

CONFIRMATORY, n=5. L=2 and L=4 still running.

| arm | rel% | improves | rtg_target | HF frac | centroid | wall |
|---|---|---|---|---|---|---|
| control (L=8) | 15.82 | 5/5 | 0.9761 | 0.883 | 0.7604 | **82.4 min** |
| **h172 L=1** | **13.69** | **5/5** | 0.4590 | 0.939 | 0.7681 | **13.2 min** |

Per-seed L=1: 7.43, 11.11, 16.73, 16.85, 16.32 — better than the control on
**4 of 5 seeds** (15.28, 14.77, 12.93, 16.90, 19.19).

**P1 holds and then some.** It predicted L=1 would land within ~3 rel% of the
control; it lands **2.13 rel% better**, improving on every seed.

## The contention confound runs the favourable way here

The wall-clock comparison is cross-run, as it was for h171's withdrawn "2.1×".
But the direction is now protective rather than threatening: **h172 ran at
15/15 workers**, the highest contention of this session, while h83's control ran
at an unknown and probably lower load. A 6.2× gap measured under *worse*
conditions cannot be manufactured by contention — it can only be understated.

That is an argument about direction, not a substitute for a matched measurement,
and it is stated as such.

## SC1 passes

HF fraction 0.939 against the control's 0.883 — no collapse, slightly higher.

## What this establishes

h171 showed the seven later rollout steps do not affect the real query. h172
shows they can simply be **removed**: one-step rollouts match and slightly beat
eight-step ones at a fraction of the cost. This is the first change to what the
**code should do** to come out of this front.

The conditioning target drops to 0.4590 — mechanically, since
rtg[0] = log(b_0) − log(b_T) over one step instead of eight. **Sixth arm to pair
a collapsed target with good performance**, and the strongest one yet, since it
also outperforms.

## What it does not establish

n=5, Borehole only, and L=2/L=4 have not reported. A dose that is non-monotone
(if L=2 or L=4 were worse than both L=1 and L=8) would complicate the reading and
is not yet excluded. And h172's registered asymmetry stands: this result supports
h171's mechanism but a null here would not have refuted it.
