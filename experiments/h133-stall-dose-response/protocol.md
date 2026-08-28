# h133 — Does the late STALL scale with tightness?

STATUS: LOCKED before computing. ZERO NEW COMPUTE.
TYPE: CONFIRMATORY.
READ POINT / STATISTIC: regret fall between post-init cost 100 and 200, as % of
|optimum|, per seed, from h83's frozen `sr_curve` + `grid`. Borehole, seeds
42-46, paired within seed. Named per convention: **late-gain, rel% @cost_curve
100->200**.

## Why

h131 (exploratory) found the ROI's advantage erodes 3.588% of optimum over the
back half, because the ROI **stalls**: control regret falls 5.193% over that span,
ROI's falls 1.605%. h132 P4 found the training-distribution signature (late
`L_loc` lower, effect 1.42, 5/5).

Both are consistent with **over-restriction** — a region well-sized while the
surrogate is uncertain, too tight once it is not. If that is the cause, **the
stall must scale with how tight the region is.** If it does not, the stall has
some other cause and both h123's and h132's premise weaken before their runs are
spent.

Four Borehole arms with known, constant, *measured* realized acceptance rates
already exist at seeds 42-46: ROI-Q10 (0.0999), ROI-FIX2 (0.2141), ROI-ANN
(0.4934, a constant arm because its schedule never advanced), ROI-OFF (1.0).

## Predictions (locked)

**P1 (PRIMARY).** Late-gain increases monotonically with realized q across the
four arms: Q10 < FIX2 < ANN < OFF.

**P2 (THE GUARD — this is why P1 alone is not enough).** A ranking bar over four
items always produces a ranking. **P1 counts only if the extreme pair (Q10 vs OFF)
is separable** at effect >= 1.0 and >= 4/5 seeds, AND I report the separability of
every adjacent pair alongside the ordering. Adjacent pairs that are ties are to be
reported as ties, not as ordering evidence.

This is the standing rule from the T2 failure ("an ordering bar must be paired
with a separability check on the adjacent pairs that carry its meaning") and from
my own error today of reading a rank order off a 0.007 gap. Registered as a gate
rather than a note because notes have not worked.

**Falsified if** late-gain is flat or non-monotone in q *while* the extreme pair
separates — that would mean the stall is real but not caused by tightness.

## Confound, stated before looking

Arms that are better at cost 100 have less regret left to remove by cost 200, so
late-gain is mechanically depressed for whichever arm is ahead at the midpoint —
and the ROI arms ARE ahead (h131: −7.81% at cost 100). **This confound pushes in
exactly the direction P1 predicts**, which makes a naive pass uninformative.

So I additionally report **late-gain normalised by regret remaining at cost 100**
(fraction of the still-available regret that each arm removes). P1 is judged on
the normalised statistic; the raw one is reported beside it. If the two disagree,
the normalised one governs and I will say so.
