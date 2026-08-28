# h135 — The full 2x2 at n=10: do the ROI and the L1 loss compose?

STATUS: LOCKED before computing. ZERO NEW COMPUTE.
TYPE: CONFIRMATORY.
STATISTIC / READ POINT / QUANTITY: simple regret from h83's frozen `sr_curve` +
`grid`, expressed as **rel% of |optimum| @cost_curve 200**, paired within seed.
Borehole, seeds 42-51 (n=10).

**Gates verified to partition with `tools/check_gate.py` before registration** —
both returned "pass and falsifier partition the outcome space". Three gates today
had holes in them; this is the first checked mechanically rather than by eye.

## Why

The composition of the ROI with the L1 localisation loss was recorded earlier as
-5.96 at 10/10 and "0.11 from additive", which would make it the **strongest
configuration in the project** — stronger than the ROI alone (pooled -3.86,
effect 1.58, 9/10). But it was never recomputed at the frozen read point, and
"0.11 from additive" was quoted without a separability check on the interaction.

All four cells exist at n=10 and no new runs are needed:

    control    h83 `MF-DRO` 42-46      + h90 `NO-ROI`  47-51
    ROI only   h84 `ROI-Q10` 42-46     + h90 `ROI-Q10` 47-51
    L1 only    h108 `L1-LOSS` 42-46    + h102 `L1-LOSS` 47-51
    both       h113 `ROI-L1` 42-51

The 42-46 control substitution was verified bit-identical at 5/5 (h120 Amendment
3, discharged); 47-51 uses its native h90 control.

## Predictions (locked)

**P1.** ROI+L1 beats control, effect >= 1.0. FALSIFIED if effect < 1.0.
(Partitioned; calibrated by the pooled ROI-alone effect of 1.58.)

**P2 — additivity.** Define
`interaction = (both - control) - [(ROI - control) + (L1 - control)]`, per seed.
**ADDITIVE if |mean|/sd < 1.0; NON-ADDITIVE if >= 1.0.** Partitioned by
construction, so every outcome has a registered verdict. Calibrated by doubling
the project's 0.5 sd effect bar, since an interaction is a difference of
differences and carries more noise than either.

**P3 — is "both" actually better than the best single?** Reported with **no
direction registered**: `both - ROI-only` and `both - L1-only`, paired, with
effect sizes. I have no grounds for a sign, and the earlier -5.96 was not read at
this read point.

## Stated before looking

- The interaction is a **difference of differences**, so its spread is larger
  than any single contrast's. Today's recurring failure has been comparing a
  difference against the wrong denominator; P2's denominator is the sd of the
  interaction itself, not of any component.
- "0.11 from additive" is a **point estimate with no spread attached**. If the
  interaction's sd turns out large, then 0.11 was never evidence of additivity —
  it was a number that happened to be small. That would be a correction to the
  record, not a new finding.
