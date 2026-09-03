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
