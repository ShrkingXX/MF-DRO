# h137 — Does the best configuration beat the strongest baseline?

STATUS: LOCKED before computing. ZERO NEW COMPUTE.
TYPE: CONFIRMATORY.
STATISTIC / READ POINT / QUANTITY: simple regret via h83's frozen `sr_curve` +
`grid`, as **rel% of |optimum| @cost_curve 200**, paired within seed. Borehole,
seeds 42-51 (n=10).

## Why this is the question that has not been asked

h135 established the best configuration: **ROI + L1 = -5.958% of optimum vs the
no-intervention control, effect 2.25, 10/10.** Every comparison in this project
has been against MF-DRO's own control arm. **Nobody has compared the best
configuration against the strongest external baseline at n=10.**

The record says MF-DRO is behind MF-MES on six of ten seeds, but that is the
*unimproved* method. Whether the improvement changes the competitive picture is
the question a write-up turns on, and it has never been computed.

Data exists and needs no runs: `MF-MES` on Borehole at seeds 42-46 (h83) and
47-51 (h115) — matching ROI-L1's own 42-51 exactly.

## Predictions (locked)

Let `d = (ROI+L1) - MF-MES` per seed, in rel% (negative = ROI+L1 better).

**P1 — three-way verdict, partitioned by construction:**
- **OVERTAKES** if `d.mean() < 0`, effect >= 1.0, and >= 8/10 seeds better.
- **STILL BEHIND** if `d.mean() > 0`, effect >= 1.0, and >= 8/10 seeds worse.
- **TIED** otherwise.

Every outcome maps to exactly one verdict. Effect threshold checked with
`tools/check_gate.py`, calibrated by h135's own contrasts on the same statistic
and read point (2.25 and 2.00).

**My expectation, stated so it can be wrong: STILL BEHIND.** The record's
"behind on six of ten" concerns unimproved MF-DRO, and -5.958 is a large gain,
so TIED is plausible. I am not predicting OVERTAKES.

**P2 (no direction).** How much of the MF-DRO-to-MF-MES gap does the
configuration close? Reported as a fraction with its spread, **never as a bare
percentage** — "57% of the gap" is already recorded in findings.md as the wrong
summary because the ratio's denominator is itself an estimate with spread.

## What this result could RETRACT

Named before running, per the rule adopted today:

1. **The report's section "It beats no baseline here"** — if P1 returns
   OVERTAKES or TIED, that heading is false as published and must be rewritten,
   not softened.
2. **"The contribution is a mechanism study, not a competitive result."** I have
   framed the work this way repeatedly. OVERTAKES would make that framing wrong
   and understated rather than cautious.
3. **research-state.yaml's headline**, which quotes the gain against MF-DRO's own
   control only. If the external comparison is unfavourable, the headline is
   quoting the flattering comparator and needs the other one beside it.

**The asymmetric risk is worth naming.** STILL BEHIND retracts nothing and
confirms my expectation — which is exactly the outcome I should scrutinise
hardest, since every one of today's four uncomputed-spread failures was a number
that supported a conclusion already held.
