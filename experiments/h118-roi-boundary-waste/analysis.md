# H118 — analysis

Data: h90-borehole-confirm only (Borehole_8D, seeds 47-51, three MF-DRO arms).
Zero new runs. No exclusions: all 15 runs cleared the n>=15 floor.

## 1. VERDICT ON THE LOCKED PREDICTION: NOT SUPPORTED. GATE FAILED.

Predicted: waste(ROI-Q10) < waste(NO-ROI) in >=4/5 seeds AND |mean|/sd >= 1.0.

Wasted HF fraction (non-init HF queries with z0 < 0.9), per seed:

| arm | 47 | 48 | 49 | 50 | 51 | mean |
|---|---|---|---|---|---|---|
| NO-ROI | 15.5% | 10.1% | 14.3% | 5.3% | 3.0% | 9.6% |
| ROI-Q10 | 10.2% | 8.3% | 17.4% | 2.7% | 0.0% | 7.7% |
| REFINE-100 | 5.3% | 2.2% | 4.5% | 1.1% | 1.1% | 2.8% |

Paired ROI-Q10 minus NO-ROI: mean **-1.90 pts**, sd 3.08, **|mean|/sd = 0.62**,
lower in 4/5 seeds. Criterion 1 met, criterion 2 **FAILED**. GATE FAILED.

Sensitivity at the declared 0.95 cut: mean -4.63 pts, sd 6.10, effect **0.76**,
lower in 4/5. Same verdict. Both cuts reported as promised.

**The resolution-amplifier account of the ROI is not supported.** The direction
is consistent (4/5 at both cuts) but the effect is not separable at n=5.

## 2. EXPLORATORY (not pre-registered): why, mechanistically

REFINE-100 is `use_roi=False, teacher_refine_samples=100, teacher_refine_noise=0.05`.
It draws 100 Gaussians around the teacher's broad winner and — the operative
detail — **clamps them to the box** (`mf_dro.py:1543`,
`torch.max(torch.min(_loc, bounds[1]), bounds[0])`). Clamping puts positive
probability mass EXACTLY ON the boundary.

Plain MF-DRO's teacher draws candidates from `_draw_raw()` = `torch.rand`,
which returns [0,1) and, uniform in 8 dimensions, essentially never approaches
the boundary corner. The ROI then FILTERS those draws.

Fraction of non-init HF queries landing on the boundary (z0 >= 0.999):

| arm | 47 | 48 | 49 | 50 | 51 | mean |
|---|---|---|---|---|---|---|
| NO-ROI | 10.3% | 19.2% | 19.5% | 5.3% | 19.2% | 14.7% |
| ROI-Q10 | 18.2% | 24.0% | 11.6% | 20.5% | 18.6% | 18.6% |
| REFINE-100 | 43.6% | 61.5% | 59.1% | 48.3% | 58.5% | **54.2%** |

REFINE-100's per-seed range (43.6-61.5%) does not overlap NO-ROI's (5.3-19.5%).
Paired waste reduction vs NO-ROI: REFINE-100 **-6.79 pts, effect 1.90, 5/5
seeds** — it would clear the gate ROI-Q10 failed, though it was never gated.

**A filter cannot create probability mass the proposal distribution never had.**
That is the cleanest statement of why the ROI does not fix this waste and local
clamped refinement does. It also unifies the h117 amendment-2 observation:
MF-MES reaches the boundary via box-constrained L-BFGS-B, which converges onto
active constraints. Three arms, three boundary-mass mechanisms, same ordering.

## 3. THE NEGATIVE THAT MATTERS: waste does not predict regret

Mean final best HF y: NO-ROI **260.84**, ROI-Q10 **271.64**, REFINE-100 **271.03**.

ROI-Q10 and REFINE-100 reach effectively the SAME final value while differing
2.8x in wasted budget (7.7% vs 2.8%). Across all 15 runs the association
between waste and final value is weak (Pearson r = -0.255; 5 seeds, 3
non-independent arms, description only, no inference).

**So the boundary waste is real, is reproducibly reduced by clamped refinement,
and does not buy final performance here.** Halving it again bought nothing. Any
account in which MF-DRO loses BECAUSE of this waste is not supported by these
data, and the h116 line of investigation should not be extended on the
assumption that it is.

## 4. What this leaves for the primary question

The ROI's Borehole regret benefit (3.5-4.2 pts, sd 0.37, 9-10 of 10 seeds — the
project's most reproducible quantity) is real and is NOT explained by boundary
resolution: h118 shows the ROI barely moves boundary mass (14.7% -> 18.6%) and
fails the waste gate, while an arm that moves boundary mass decisively buys no
extra regret. The channel through which the ROI helps remains unidentified.

## 5. Limitations

- n=5 seeds, one benchmark. No p-values.
- Section 2 is EXPLORATORY: REFINE-100 was reported alongside by protocol but
  was not part of the hypothesis and is not an ROI arm (`use_roi=False`).
- h118 inherits h116's status: the waste measure itself is exploratory pending
  h117.
- Section 3's correlation pools three non-independent arms over five seeds. It
  is a description. It is not evidence of absence of a relationship, only
  absence of a strong one in this sample.
