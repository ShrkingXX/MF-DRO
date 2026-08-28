# H113 — do the two surviving interventions compose, or share a bottleneck?

LOCKED BEFORE ANY RUN. Arm checked for prior existence before this file was
written (peer session's rule, adopted after they nearly duplicated h111): no
combined ROI+L1 arm exists anywhere in the tree.

## Why this is the right question now

Exactly two interventions have replicated at a second seed set, and they are
strikingly parallel:

      calibrated ROI (q=0.10)   -4.08 pts, 10/10   Borehole only
      L1 location loss          -2.21 pts,  9/10   Borehole only

Both improve Borehole regret. Both are **Borehole-only** — h111 has now shown the
ROI fails at two tightness settings on Hartmann and Ackley. And both are
**mechanistically unexplained in the same specific way**: each improves regret
*without moving the quantity its proposed mechanism operates on*. The ROI
increases proposal dispersion while the founding diagnosis said dispersion was
the problem; L1 leaves boundary-reaching indistinguishable from zero while it was
chosen to raise it.

**Two effects with the same footprint, on the same benchmark, both unexplained.**
Whether they compose is the cheapest available test of whether they are one
phenomenon or two.

## Design — a 2x2, three cells already run

| | |
|---|---|
| benchmark | Borehole_8D |
| seeds | 42-46 **and** 47-51 — n=10 from the start, not n=5 plus a re-test |
| new arm | **ROI+L1**: `use_roi=True, roi_target_accept=0.10, loc_loss='l1'` |
| reused | no-ROI/MSE (h83, h90) · ROI alone (h84, h90) · L1 alone (h102, h108) |
| runs | 10 |

q=0.10 rather than q=0.05 deliberately: q=0.05's advantage was **withdrawn** when
h110 reversed it at a third seed set, so q=0.10 is the setting with the strongest
evidence (−4.08 at 10/10) rather than the one that briefly looked better.

Running both seed sets at once is the direct lesson of this session: five of
tonight's results needed a second seed set, and two of those did not survive it.

## Gate

Both manipulations must be **observed**, not read back from config:
  - `accept_frac` in [0.095, 0.105] — the ROI is active
  - final `L_loc` > 0.10 — the L1 objective is active (MSE runs give 0.033–0.038)
Either failing voids the arm regardless of regret.

## Predictions

**P1.** ROI+L1 beats no-ROI/MSE. Registered **POSITIVE** and near-certain; both
components do individually at 10/10 and 9/10.

**P2 — the actual question.** ROI+L1 beats **ROI alone**, clearing |paired mean| >
0.59 AND ≥8/10. Registered **GENUINELY UNCERTAIN.** Ten mechanism predictions
have been refuted in this investigation and I am not adding an eleventh.

**P3 — the diagnostic, stated as arithmetic rather than as a prediction.**
Additive would be ≈ −6.3 (−4.08 + −2.21); a shared bottleneck would be ≈ −4.1,
i.e. no better than the ROI alone. The measured value falls somewhere on that
line and its position is the result. **No threshold is registered for P3** — it
is descriptive, and calling a midpoint "partially additive" after the fact would
be exactly the unfalsifiable reading this project keeps catching.

## What each outcome means

  - **Composes (≈ −6.3):** two independent mechanisms, and the combination is the
    strongest configuration measured in this project.
  - **Does not (≈ −4.1):** one shared bottleneck reached two ways. That is the
    first real constraint on the mechanism question, which h111 just showed
    cannot be constrained by adding benchmarks.
  - **Worse than either alone:** they interfere, which would be the most
    surprising outcome and the most informative about what the head is doing.
