# h129 — Is the ROI's Borehole benefit MEDIATED by the fidelity mix?

**Status:** CONFIRMATORY. Registered BEFORE h127 (q=0.30) results exist —
h127 is running now, 6/10 dispatched, 0 result files written.

## Why

h120 confirmed the ROI reallocates fidelity, but on a SINGLE contrast. A single
contrast cannot distinguish "the ROI shifts fidelity and that is why it helps"
from "the ROI shifts fidelity and also, separately, helps". A dose-response can:
if the fidelity shift mediates the benefit, then **the shift and the benefit must
track each other across q**, not merely co-occur at one q.

## What is already measured (Borehole, post-init, seeds 42-46, paired)

    arm                HF fraction
    control (no ROI)   0.8829
    ROI q=0.10         0.7390     shift -0.1439, sd 0.0874, effect 1.65, 5/5 lower

The benefit at q=0.10 is -4.22% rel, effect 1.74, 5/5 (peer's recomputation from
raw `hf_regret_curve`). Effect sizes 1.65 and 1.74 are close, which is suggestive
but not itself evidence — closeness of two effect sizes is not mediation.

Direction of the mechanism: a tighter ROI spends LESS on HF, buying MORE total
evaluations (n_post 106.6 -> 115.8 -> 121.4 as q goes 1 -> 0.10 -> 0.05).

## The model

HF fraction is linear in log q, anchored at q=0.10 and at the q->1 limit (an ROI
accepting everything is no ROI, so it must return to the control value). Benefit
is proportional to the fidelity shift.

## PREDICTIONS (locked)

**P1 — held-out dose, h127 q=0.30.** HF fraction = **0.808 +- 0.020**.
Falsified if it lands at the control value 0.883 (no fidelity effect) or at the
q=0.10 value 0.739 (a step, not a dose-response). Both alternatives are >3 sems
away, so P1 has real content.

**P2 — mediation, h127 q=0.30.** Benefit = **2.21% rel**, bounded by the two
measured doses (4.22% at q=0.10, 1.31% at q=0.493). Falsified if the benefit
falls outside [1.31, 4.22] while the HF fraction lands inside P1's interval —
that would show the shift happens without the benefit tracking it.

**P3 — the already-existing test, h128 q=0.493.** HF fraction = **0.839 +- 0.012**.
This one is testable the moment the peer supplies h128's fidelity numbers; I
cannot compute it myself (their `results/ckpt` is empty). Registering it here so
it is locked before I see it.

## Honest statement of what P3's two routes do and do not show

Two routes predict the loose arm independently: a log-q fit that never sees
h128's benefit (0.8387), and a benefit-mediation ratio that never sees the log
form (0.8382). They are NOT circular — they share the control and q=0.10 anchors
but differ in their third input.

**Their agreement to 0.0005 is far luckier than the data warrants.** The paired
sd of the shift is 0.0874, so the sem is 0.039; anything inside about +-0.04 is
indistinguishable. The correct claim is "the two routes agree well within
uncertainty", NOT "they agree to four decimals". Recording this here so the
figure is never quoted at its face precision later.

## Gate

P1 is the primary. Reported whatever it shows, including if h127's remaining
seeds change the mean. n=5-6, no p-values.
