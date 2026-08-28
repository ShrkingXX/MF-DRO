# H108 — does the L1 regret gain survive a second seed set?

LOCKED BEFORE ANY RUN. ID claimed via `tools/claim_id.sh`.

## Why

h102 found that training the location head under an L1 loss improves Borehole
regret by **−2.08 points at 4/5**, clearing the same |mean| > 0.59 AND ≥4/5 bar
that h97's tightness result cleared and that two ROI settings 2.1x apart failed
to clear. Its mechanism prediction (P1) was **refuted** — L1 reaches boundaries
*less*, not more — so the gain is real and **unexplained**.

I wrote in h102's own write-up that it "needs the same re-test h107 is giving
h97". This is that re-test. Running it is the follow-through on that sentence;
not running it while building on the result would be the failure this session has
catalogued repeatedly.

## Design

| | |
|---|---|
| benchmark | Borehole_8D |
| seeds | 42, 43, 44, 45, 46 — the original set |
| arm | L1-LOSS (`loc_loss='l1'`), identical to h102's |
| control | h83's MF-DRO at the same seeds (use_roi=False) |
| runs | 5 |

The control is legitimate reuse: the `use_roi=False` branch of `mf_dro.py` is
**byte-identical (md5 ff70f008c0ac)** across every commit involved, established
in h105 and corroborated by h84's reproduction control at 0.000e+00.

## Gate

Same G3, non-vacuous: **final L_loc must exceed 0.10** where MSE runs report
0.033–0.038. h102's five came in at 0.1188–0.1302.

## Predictions

**P1 (regret).** The gain replicates, clearing |paired mean| > 0.59 AND ≥4/5.
Registered **GENUINELY UNCERTAIN.** h102 is one seed set, and this session has
watched −5.85 become −2.11, a 4/5 become 2/5, and two claims withdrawn entirely
on exactly this re-test.

**P2 (mechanism, and a test of my own reasoning).** L1 again reaches boundaries
**LESS** than the control, ≥4/5. Registered **POSITIVE.**

This one is worth isolating. h102's P1 failed because I predicted from an
*assumed* shape for the teacher's target distribution and never measured it.
P2 here predicts the same quantity from the *measurement* h102 produced. If
prediction-from-measurement succeeds where prediction-from-assumption failed,
that is a concrete instance of Lesson 23 working rather than merely being stated
— and if it also fails, the lesson is insufficient and I should say so.

**P3.** L1 does not make MF-DRO competitive with MF-MES on Borehole. Registered
**POSITIVE**; true of every intervention tried.

## What each outcome means

  - **P1 holds:** an unexplained but replicated regret effect from the output
    loss alone, on two independent seed sets, with its proposed mechanism already
    refuted. That is a real finding and the mechanism becomes the open question.
  - **P1 fails:** h102 joins the withdrawn list, and the honest position is that
    the only replicated intervention remains the calibrated ROI.
