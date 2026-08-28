# h132 — If the late STALL is over-restriction, switching the ROI OFF should recover it

STATUS: LOCKED before any run. NOT YET LAUNCHED.
TYPE: CONFIRMATORY, and it tests **my own post-hoc hypothesis** — see the
      weakness statement below, which is the most important section here.
COMPUTE: queued BEHIND the peer's h123. Does not launch until h123 is fully
      dispatched. h127 seed 51 also outranks it.
READ POINT: post-init cost, h83 frozen `sr_curve` + `grid`, reported at 25, 50,
      100, 200 — not 200 alone, since a schedule mixes regimes.

## The finding this follows from

h131 (exploratory) measured that the ROI's Borehole advantage erodes +3.588% of
optimum between cost 100 and 200, effect 1.40, 5/5 seeds, and that the mechanism
is a **stall**: over that span the control's regret falls 5.193% of optimum while
the ROI's falls 1.605%. The ROI stops improving; the control does not.

## The arm

`ROI-STEP` = q=0.10 until post-init cost 100, then `use_roi=False` for the
remainder. A step, not a ramp. The peer's h123 tests the paper's smooth `beta_t`;
this isolates the mechanism with maximum contrast, because a step confounds
nothing with the schedule's shape.

## Predictions (locked)

**P1.** ROI-STEP beats constant q=0.10 at cost 200, >= 4/5 seeds, effect >= 1.0.

**P2.** ROI-STEP is **indistinguishable** from constant q=0.10 at cost 100 —
effect < 0.5 — because the two arms are byte-identical before the switch. This is
a **manipulation check, not a finding**: if it fails, the implementation is wrong
and P1 is void regardless of which way P1 landed.

**P3.** ROI-STEP does not fully close to the control's late progress: it recovers
part, not all, of the 3.588%. Registered so that "recovers everything" would be
as surprising as "recovers nothing", and neither can be narrated as success.

**Falsified if** ROI-STEP is worse than constant q=0.10 at cost 200 with effect
>= 1.0. That would mean the late restriction is doing useful work and the stall
has another cause.

## THE WEAKNESS, stated first

**This design is tuned to my own hypothesis, which was post-hoc.** I ran the
erosion analysis after seeing the curve that suggested it, then built the arm most
likely to confirm it. That is a weaker evidential position than the peer's h123,
which registered no direction and was not designed around my idea.

**So h123 adjudicates my prediction and h132 does not.** If both land the same
way, h123 carries the weight and h132 explains the mechanism. If they disagree,
h123 governs. Recording this before either runs, because afterwards it would be
indistinguishable from an excuse.

## A read-point hazard this creates, recorded now

h131 showed Borehole's benefit PEAKS at −7.813% (cost 100) and DECLINES to
−4.224% (cost 200). The peak is the more flattering number and it sits exactly at
this experiment's switch point.

**The frozen metric is cost 200. −4.224% remains the headline.** The peak is a
mechanistic fact, not an alternative headline, and quoting it as the ROI's benefit
would be precisely the read-point flexibility this project has spent the day
guarding against. Registered here so that neither session drifts into it.
