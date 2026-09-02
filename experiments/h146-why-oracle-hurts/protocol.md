# h146 — WHY does a better teacher hurt? Separating QUALITY from DIVERSITY.

STATUS: LOCKED before computing.
TYPE: CONFIRMATORY.
STATISTIC: rel% of |optimum| @cost_curve 200, h83 frozen `sr_curve`+`grid`,
paired, Borehole seeds 42-46. **Borehole only** — h145's Hartmann arm is
confounded (corr(HF share, degradation) = -0.830) and cannot be attributed.

## The confound in h145's own design

h145's oracle teacher is simultaneously:

  1. **high QUALITY**  — every trajectory walks to the true optimum, and
  2. **zero endpoint DIVERSITY** — every trajectory ends at the SAME point, x*.

Its result (+28.13 rel% worse, effect 4.49, 0/5 on Borehole) cannot distinguish
these. **"Better trajectories hurt" and "degenerate targets hurt" predict the
same outcome**, and only the first is what the experiment claims to have shown.

## The standing hypothesis, and why it needs this test

The DT fits state -> action. If every trajectory ends at x*, the target has a
large component that does NOT depend on the state, and x* is not in the state and
not inferable from it. The head would then fit a near-constant map that is useless
at inference.

The existing evidence is consistent with this but does not establish it: the
oracle runs show **L_loc 3x LOWER** (0.018 vs 0.040 — the head fits its targets
*better*) and **lower query dispersion** (0.189 vs 0.277 — a more collapsed
policy). Both are what "learned a degenerate map" predicts. Neither rules out
"quality itself is harmful".

## The arm that separates them

**DIVERSE-GOOD.** Identical to h145's oracle in every respect except the
destination: each trajectory interpolates from a random `x_start` toward a
**different** high-quality endpoint, drawn as the best of a fresh random pool
under the TRUE objective (top-of-pool, so quality is high but the endpoint varies
per trajectory). Same `forced_x` hook, same info-gain fidelity selection at the
forced point, same config, same seeds.

    ORACLE        quality HIGH   endpoint diversity ZERO   (h145, have it)
    DIVERSE-GOOD  quality HIGH   endpoint diversity HIGH   (this arm)
    CONTROL       quality MES    endpoint diversity HIGH   (h83, have it)

## Predictions (locked)

**P1 (PRIMARY).** DIVERSE-GOOD does **not** collapse the way ORACLE does: its
degradation versus control is **less than half** of ORACLE's +28.126.
FALSIFIED if it degrades by >= 14.06 rel%.

**P2.** If P1 holds, **diversity — not quality — is what the oracle destroyed**,
and "better trajectories hurt MF-DRO" is the wrong reading of h145. The right one
would be that *degenerate* trajectories hurt, and h145 happened to make its
trajectories degenerate in the course of making them good.

**P3 (no direction).** L_loc and query dispersion for DIVERSE-GOOD, reported
whatever they show. Under the hypothesis they should sit near the control's
rather than near ORACLE's; I register no threshold because I have no calibration
for them.

## What this could RETRACT

**h145's headline as I stated it to the user: "better trajectory quality does not
improve MF-DRO — it degrades it markedly."** If DIVERSE-GOOD is fine, that
sentence is wrong as written. The defensible version becomes narrower and more
interesting: *teacher trajectories whose targets do not vary with the state
destroy the policy, however good those targets are.*

That is the outcome I should watch hardest, because it means I reported a
confounded result to the user as a clean one — for the second time in this
experiment, after the Hartmann fidelity confound.

## Gate

Reuses h145's `forced_x` hook, already gated: SC4 passed bit-identical on the
corrected tree (137 queries, 0 differing). No new src change, so no new gate.

---

## AMENDMENT 1 — the arms cannot be quality-matched, measured before running

DIVERSE-GOOD's endpoints, measured over 300 draws on Borehole (POOL=256):

    endpoint quality   mean y = 220.76  =  71.3% of the true optimum (309.58)
    endpoint diversity per-dim sd on the unit cube = 0.2048
                       mean pairwise distance      = 0.8572
    ORACLE for comparison: quality 100%, diversity 0.0000 BY CONSTRUCTION

**The two arms are not quality-matched and cannot be.** At the limit of perfect
quality the endpoint *is* x*, which is exactly zero diversity — quality and
diversity are intrinsically coupled at the top. So this is not a clean 2x2 and I
am not going to present it as one.

**What the contrast can still decide.** ORACLE (100% quality, 0 diversity)
degrades by +28.126. If DIVERSE-GOOD (71% quality, high diversity) degrades far
less, then **the collapse is specific to the degenerate limit** rather than a
monotone cost of teacher quality. If it degrades similarly, teacher quality
really is harmful across the range, and h145's headline survives.

**The sharper design this implies, registered now and not yet run:** a dose over
POOL — {16, 256, 4096} — traces quality UP and diversity DOWN along one axis. If
degradation is monotone in POOL, the tradeoff is the whole story. That is 15 runs
and is the natural follow-up if the single contrast is informative; recording it
here so it is not presented later as though it had been the plan all along.
