# h190 — **DECLINED before launch.** Two independent reasons, both found by pre-launch checks.

The protocol was committed; the arm was **not run**. Recorded here in full because a
declined arm is a result.

## Reason 1 — the intervention is ONE-SIDED (found by the smoke test)

`max_hf_fraction` is a **ceiling on high fidelity**: it can force *more* LF, never more
HF. Smoke on Hartmann seed 42:

```
SC-a teacher path still taken : PASS (33 iters)
SC-b lf_fraction              : 0.970   (h189's unstable teacher on this seed: 0.989)
ceiling overrides fired       : 1
```

The seed was already LF-heavy, so the ceiling had almost nothing to do. Applied across
h189's five seeds it would lift the two HF-heavy ones (0.000, 0.368) toward 0.80 and
**leave the three LF-heavy ones (0.924, 0.934, 0.989) untouched**. That is a *partial*
stabilisation, and the SC threshold I registered (0.68–0.90 on every seed) was
mis-specified because I had not noticed the one-sidedness. The smoke caught it.

## Reason 2 — h189's own per-seed data undercuts the account

The account says the teacher is hurt by *bad* fidelity allocations. Its own performance,
sorted by allocation:

| teacher `lf_fraction` | teacher rel% | MF-DRO rel% | gap |
|---|---|---|---|
| 0.000 | 16.41 | 10.16 | +6.25 |
| 0.368 | 16.34 | 0.67 | +15.67 |
| 0.924 | 14.02 | 7.42 | +6.60 |
| 0.934 | 15.54 | 5.28 | +10.26 |
| 0.989 | **47.09** | 16.41 | +30.68 |

**Excluding the single extreme seed, the teacher scores 14.02–16.41 across allocations
spanning 0.000–0.934** — essentially the whole fidelity range, and its performance
barely moves. **MF-DRO beats it at every allocation it tried.**

The teacher is not losing because its allocation is unstable or badly chosen. **It is
losing at every allocation.** The bistability is real and was worth recording, but it is
not what costs the teacher its 13.89 points.

## Consequence

**h190 is not run.** Running it would have tested a weakened account with a half-strength
intervention, and the likeliest outcome (P3, no effect) would have been uninformative
because it could equally be blamed on the one-sidedness.

**The Hartmann sign flip has NO established mechanism.** h189's candidate account was
explicitly marked "not established"; it is now **undercut**, not merely unproven. That
leaves two open items on this project — the benchmark asymmetry and the sign flip — and
I should stop proposing mechanisms for them until something other than a five-seed rank
correlation suggests one. Four mechanisms have been proposed across these two questions;
three were refused by direct measurement and this is the fourth.

This is the **third arm declined on a measured premise** (after h174's follow-up and the
teacher-rotation arm). The pattern is worth keeping: the checks cost minutes, the arms
cost hours.
