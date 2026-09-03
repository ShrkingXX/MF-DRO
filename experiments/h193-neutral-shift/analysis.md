# h193 — **ABANDONED before launch**, per its own protocol. Reported as infeasible.

No runs. Four designs, all rejected by their own SC. The protocol committed in advance:

> "If SC-2 fails on Hartmann too, the test is abandoned and reported as geometrically
> infeasible **rather than iterated further**."

SC-1 failed on Hartmann. Abandoning.

## What the arm was for

h192 moved the teacher's τ=0 mean toward the box centre and got two results at once: the
DT followed **one-for-one** (transfer ratio 1.094), and regret collapsed 13.69 → 43.18.
The direction was not neutral, so the **regret** half is a joint test of "the DT follows"
and "the centre is bad". h193 was to displace by the same magnitude in a
distance-preserving direction and separate them.

## Why it cannot be done — two independent obstructions, both measured

### 1. On Borehole, the geometry forbids it

| design | SC-2: distance preserved? | cause |
|---|---|---|
| per-point rotation | 0.7837 → **0.5319** | rotation planes decohere across points; each point's radius is kept but the *mean* shrinks |
| fixed plane (0,6) | 0.7837 → **0.6823** | in-plane radius 0.5817 > 0.5; rotation exits the box and clipping pulls it back |
| fixed plane (3,6) | 0.7837 → **0.6684** | the mean's circle fits, but individual points near the walls still clip |

**Measured cause:** of the control's 486 real HF queries, **80.9% have a coordinate within
0.05 of a box wall**, and dim 0 sits at mean |coord| **0.465** against a half-width of
0.5. Borehole's good region *is* the boundary, so the working policy's constant is pressed
against the wall and **the only direction with room to move it is inward**.

### 2. On Hartmann, the intervention is not controllable

Hartmann is interior (0 wall-pinned dims vs Borehole's 1; 28.4% near a wall vs 80.9%), so
the geometry allows it. The smoke ran both arms:

| arm | dist from centre | displacement from control mean |
|---|---|---|
| control | 0.4013 | — |
| centre-shift | 0.2349 ✓ halved | **0.1757** |
| neutral rotation | 0.5005 ✗ *rose* | **0.2934** |

**The displacements differ by 67%**, so the two arms are not comparable and the gate
cannot be read. And the rotation *raised* the distance from the centre rather than
preserving it.

**The reason is not fixable by choosing a better plane.** The shift changes the run: the
DT emits different queries, the GP sees different data, and the teacher — which re-decides
from that model — produces a **different action distribution**. Realised displacement and
realised distance are therefore **outcomes of the intervention, not inputs to it**. One
cannot hold magnitude fixed while varying only direction, because the perturbation
changes the system that generates the actions.

## What this leaves standing, and what it does not

- **h192's primary result is untouched.** The transfer ratio (1.094, the DT reproduces an
  imposed shift of its teacher's τ=0 mean one-for-one) does not depend on the shift
  direction. The mechanism remains interventionally confirmed.
- **h192's secondary remains confounded, and now permanently so by this route.** The
  regret collapse (13.69 → 43.18, improves-on-own-init 5/5 → 1/5) is a joint test of "the
  DT follows" and "the centre is bad". h182's centre-collapse account keeps its
  correlational support (ρ −0.967 / −0.841, DT-specific) but does **not** gain a clean
  causal test from this direction.
- **A different design would be needed** — one that intervenes on something other than the
  action's position, or that measures rather than imposes the displacement. Nothing of the
  sort is proposed here; four attempts is enough, and the protocol said so in advance.

## Cost and what was kept

No compute was spent on runs. The `tau0_shift_mode='tangent'` code path stays in
`mf_dro.py`, disabled by default and identity-gated (122.29066752728207, exact), for any
future attempt.
