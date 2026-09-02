# h176 -- does the project's BEST result survive a one-step rollout?

STATUS: protocol locked, nothing run. TYPE: CONFIRMATORY.

## Why this connects the front to the project

The teacher front's answer is about the ROLLOUT. The project's actual
contribution is the **ROI** -- a filter on the candidate pool the teacher
argmaxes over. Those two things meet here.

Borehole, seeds 42-46, frozen metric:

| arm | rel% | wall | per-seed |
|---|---|---|---|
| control (no ROI, L=8) | 15.82 | 82.4 min | 15.3, 14.8, 12.9, 16.9, 19.2 |
| **ROI-Q10 (L=8)** | **11.59** | 117.4 min | 11.5, 12.3, 11.4, 11.2, 11.6 |
| L=1 (no ROI) | 13.69 | 13.2 min | 7.4, 11.1, 16.7, 16.8, 16.3 |

The ROI is applied **once per rollout** (mf_dro.py:1334, outside the tau loop),
so it shapes the pool every step draws from -- including tau=0. If only tau=0
reaches inference, the ROI's benefit should survive shortening the rollout.

## Why this comparison is resolvable at n=5 when earlier ones were not

ROI-Q10's per-seed spread is **11.2-12.3, sd ~0.4** -- far tighter than the
control's 12.9-19.2. h172's "L=1 beats the control" could not be claimed because
the spreads overlapped; here a 2 rel% shift would sit well outside ROI-Q10's own
spread. Noted in advance so the claim made at the end matches what the design can
carry.

## The arm

`use_roi=True, roi_beta_mode='quantile', roi_target_accept=0.10` (h84's ROI-Q10,
verbatim) **plus** `rollout_length=1`.

## Predictions

P1 ROI+L1 lands near ROI-Q10's 11.59 -- within ~1.5 rel%, i.e. inside a band a
   few times ROI-Q10's own sd.
P2 Wall-clock falls to roughly L=1's range (~15-25 min), against ROI-Q10's 117.4.

## What this can RETRACT

R1 ROI+L1 loses the ROI benefit (lands near 13.69 or worse) -> **the ROI's
   benefit does NOT come only through tau=0.** That would bound the front's
   answer: the ROI acts through something the one-step rollout removes, and the
   "only the first step matters" reading -- already Borehole-scoped -- would be
   further limited to the no-ROI configuration. This is the outcome that costs
   the most and it is named first.
R2 P1 and P2 both hold -> the project's best result runs at roughly a sixth of
   its cost, and the two findings compose.
R3 ROI+L1 BEATS ROI-Q10 materially -> unexpected; would need its own explanation
   rather than being banked, since nothing predicts it.

## Named confound

Realised HF fraction against ROI-Q10's, checked before the regret is read.

## Design

Borehole_8D seeds 42-46, n=5. 5 workers.
