# h193 — separate "the DT follows its teacher's mean" from "the centre is bad"

**CONFIRMATORY.** Committed before the code is written and before any result exists.

## Why

h192 moved the teacher's τ=0 mean toward the box centre and got two results at once:
the DT's query followed **one-for-one** (transfer ratio 1.094), and regret collapsed
**13.69 → 43.18** with improves-on-own-initial-design falling **5/5 → 1/5**.

h192's own limits section names the problem: **the shift direction was not neutral.**
It aimed at the centre, which h182 had already identified as a bad region. So the
*regret* half is a **joint** test of two claims and does not independently establish the
second. Any displacement might be equally harmful.

## The arm

Displace the teacher's τ=0 action by the **same magnitude** but in a
**distance-preserving** direction: rotate it about the box centre, in the plane spanned
by `(x − c)` and a fixed orthogonalised direction, by

```
theta = 2 arcsin(chord / 2r) = 34.0 degrees      (chord 0.4585, r 0.7837)
```

which reproduces h192's measured displacement of **0.4585** while leaving the distance
from the centre **unchanged**. Borehole, seeds 42–46, rollout_length=1, frozen metric.
Controls: **ROLLOUT1** (unshifted, 13.69) and **h192 TAU0SHIFT** (centre-shifted, 43.18).

## SC before the regret

Two conditions, both read first:
1. **displacement ≈ 0.4585** — the teacher's τ=0 mean must move by about as much as
   h192's did, else the arms are not comparable.
2. **distance from centre ≈ 0.78, i.e. essentially unchanged** — if the rotation shrinks
   it materially, the arm is a weaker copy of h192 and cannot separate anything.

If either fails, regret is not read.

## Gates

**Primary — the key statistic, regret change vs the unshifted control (13.69):**

- **P1 — centre-specificity CONFIRMED**: change **< +10.0**. Displacing the constant is
  cheap when it stays equally far from the centre; h192's +29.49 was about *where* it
  landed, not *that* it moved.
- **P2 — partial**: **+10.0 to +20.0**.
- **P3 — centre-specificity REFUTED**: **> +20.0**. Any displacement of the constant is
  roughly as harmful, and h182's "the centre is bad" does **not** explain h192.

**Secondary — a re-test of h192's primary in a new direction:** the transfer ratio should
again be ≈1. If the DT follows a centre-ward shift but *not* a tangential one, the
mechanism is direction-dependent in a way nothing predicts.

## What this could RETRACT

- **P3 fires → h182's centre-collapse account loses its causal reading.** "Failing arms
  collapse to the centre" would remain a true description, but the centre would not be
  what costs the regret — displacement from wherever the teacher would otherwise have
  pointed would be. h182's analysis, findings.md's Phase 2 header and the report's
  centre-collapse section would all need rewording.
- **A low transfer ratio here** would contradict h192 and make the mechanism
  direction-dependent — unpredicted by anything, and it would reopen h192.
- P1 makes "the centre is bad" an independently established causal claim rather than an
  inference from a confounded shift.

## Prerequisite

`tools/identity_gate.py` must PASS exactly on the patched core before launch.

## Compute

5 workers × 1 thread. Machine idle.

---

# REVISION, before launch — the arm moves to HARTMANN. Borehole is geometrically infeasible.

**Three successive designs failed their own SC, and the reason is a property of the
problem, not of the code.**

| design | SC-2 (distance preserved?) | why it failed |
|---|---|---|
| per-point rotation, own plane | 0.7837 → **0.5319** | planes decohere across points; the *mean* shrinks even though each point's radius is kept |
| fixed plane (0,6), θ=46.4° | 0.7837 → **0.6823** | in-plane radius 0.5817 > 0.5, so rotation pushes a coordinate past the wall and clipping shrinks it |
| fixed plane (3,6), θ=56.7° | 0.7837 → **0.6684** | the *mean's* circle fits, but individual points near the walls still clip |

**The obstruction, measured.** Of the unshifted control's 486 real HF queries on Borehole,
**80.9% have at least one coordinate within 0.05 of a box wall**, and dim 0 sits at mean
|coord| **0.465** against a half-width of 0.5 — pinned. A distance-preserving rotation
must trace a sphere; with the points on the walls, every rotation exits the box, clipping
pulls them back, and the distance is not preserved.

**This is informative, not merely an obstacle.** Borehole's good region *is* the boundary
(consistent with its known boundary optima), so the working policy's constant is pressed
against the wall — and the only direction with room to move it is **inward**. h192's
confound may therefore be **unavoidable on Borehole**: any feasible displacement of the
constant there is necessarily toward the centre.

## The arm moves to Hartmann, where the optimum is interior

| | points within 0.05 of a wall | wall-pinned dims | mean dist from centre |
|---|---|---|---|
| Borehole | **80.9%** | **1 of 8** (0.465) | 0.806 |
| **Hartmann** | **28.4%** | **0 of 6** (max 0.357) | 0.629 |

Hartmann's actions are interior, so a distance-preserving rotation has room.

**Two arms, both at rollout_length=1, both on Hartmann, seeds 42–46:**
- **CENTRE-shift** (h192's manipulation, ported) — needed because the Borehole
  centre-shift cost (+29.49) cannot be assumed to transfer, so the comparison must be
  within-benchmark.
- **NEUTRAL rotation** — same displacement, distance preserved.

Control: h174's ROLLOUT1 (Hartmann, L=1, 10.91). h174 was voided as a *rollout-length*
comparison by its own fidelity-mix SC, but it is a valid control **here** because all
three arms share the identical L=1 config, so that mix is common to all of them.

**Revised gate.** The registered thresholds (+10 / +20) were calibrated on Borehole's
+29.49 and do not transfer. The statistic becomes the **ratio**:

> **neutral-shift regret change ÷ centre-shift regret change**, both vs the same
> unshifted control, measured in the same runs.
>
> - **P1 — centre-specificity CONFIRMED**: ratio **< 0.35**
> - **P2 — partial**: **0.35 – 0.70**
> - **P3 — REFUTED**: **> 0.70** — displacement is about as costly wherever it points

SC unchanged: displacement must match between the two arms, and the neutral arm's
distance from centre must be preserved. **If SC-2 fails on Hartmann too, the test is
abandoned and reported as geometrically infeasible** rather than iterated further.
