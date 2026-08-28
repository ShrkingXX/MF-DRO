# H127 — where does the ROI stop working? Bisecting the untested gap

LOCKED BEFORE ANY RUN. Arm checked first: no q≈0.30 arm exists in the tree.
**Needs no `src/` change** — it is a config value, so it can run while the shared
tree stays quiet-locked for a peer's h117.

## Why

h125 refuted my claim that tightness is a null axis: at a 5x contrast the effect
is 5.69 at 5/5, the largest in the project. Every acceptance level measured on
Borehole, seeds 42-46, against the same no-ROI control:

      q = 0.050   -5.79
      q = 0.100   -4.22
      q = 0.214   -4.81
      q = 0.493   -1.31

**The benefit is roughly flat from 0.05 to 0.21 and has largely collapsed by
0.49.** The degradation begins somewhere in between, and **nothing has been run in
that interval.** That gap is the entire remaining uncertainty about how to set
this knob.

This matters for the task's primary question directly. "Find an ROI strategy"
is not answered by a single working value; it is answered by knowing the usable
range. A practitioner needs to know whether 0.10 sits comfortably inside a wide
plateau or near an edge.

## Design

| | |
|---|---|
| benchmark | Borehole_8D |
| arm | **ROI-Q30** — `roi_target_accept=0.30`, otherwise identical to h97's Q05 arm |
| seeds | 42-46 and 47-51 — n=10 registered, launched as slots free |
| comparators | q=0.10 (h84, h90), q=0.214 (h84 FIX2), q=0.493 (h84 ANN), no-ROI |
| runs | 10 |

q=0.30 bisects the untested interval on a log scale (0.21 → 0.49 has midpoint
≈0.32 geometrically, 0.30 chosen as the round value inside it).

## Gate

Same G3, non-vacuous: **every run's logged `accept_frac` must lie in
[0.29, 0.31]**. h97/h107 hit their 0.05 target to within 0.0002 across ten runs,
so a miss here means the calibration failed, not that the bar is tight.

## Predictions

**P1.** ROI-Q30 still beats no-ROI (negative paired mean, ≥8/10). Registered
**POSITIVE**: even q=0.493 beats no-ROI by 1.31, so a setting less than two-thirds
as loose should too.

**P2 — the actual question.** Registered as **GENUINELY UNCERTAIN, no direction.**
The two readings are a plateau that extends to ≈0.3 and then falls off sharply,
or a smooth decline already underway by 0.3. I have been wrong about this axis
once today by generalising beyond the measured interval, and will not do it again
by predicting inside an interval nobody has measured.

**P3.** Does not make MF-DRO competitive with MF-MES. Registered **POSITIVE**.

## What each outcome means

  - **Q30 ≈ Q10 (within the 0.59 bar):** the plateau extends to at least 0.3, and
    the collapse is sharp somewhere in (0.3, 0.49). The knob is forgiving over a
    6x range, which is a genuinely useful practical statement.
  - **Q30 clearly worse than Q10 but better than ANN:** the decline is gradual and
    begins below 0.3. Then "q ≤ 0.10" is a real recommendation rather than a
    conservative one.
  - **Q30 ≈ ANN:** the collapse happens early, between 0.21 and 0.30, and
    ROI-FIX2's 0.214 sits near a cliff edge — which would make every result using
    it precarious.
