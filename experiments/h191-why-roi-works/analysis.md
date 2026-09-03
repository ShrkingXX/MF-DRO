# h191 — the mechanism explains the project's POSITIVE results too

**EXPLORATORY.** No new runs. Re-analysis of saved Borehole traces.

Every previous application of the mechanism explained a *failure*. This asks whether
it also explains the project's **surviving interventions** — the ROI (its main result:
−3.86 pts, 9/10, confirmed at fresh seeds), teacher refinement, and the L1 loss.

## Prediction

If the DT emits its teacher's τ=0 action mean, then an intervention can only help by
**moving that mean**. It cannot help by making the DT "learn better", because the DT is
already at the best-constant solution (h185, loss/var 0.750–1.054 across 10 arms).

## The cleanest test: ROI at rollout length 1

At L=1 the recorded all-τ teacher mean **is** the τ=0 mean, so nothing is averaged away.
ROLLOUT1 and ROI-Q10-L1 differ only in whether ROI constrains the teacher's pool:

| | ROLLOUT1 (no ROI) | ROI-Q10-L1 (ROI) | change |
|---|---|---|---|
| frozen rel% | 13.69 | **10.81** | −2.88 |
| teacher τ=0 mean, dist from centre | 0.7788 | **0.8869** | **+0.108** |
| teacher action variance | 0.0359 | **0.0243** | −32% |
| DT query centroid, dist from centre | 0.8484 | **0.9231** | **+0.075** |
| \|teacher mean − DT centroid\| | 0.1353 | 0.1605 | — |

**ROI moves the teacher's τ=0 mean 0.108 further from the centre, and the DT follows by
0.075 — about 70% of the shift.** It also tightens the teacher's action distribution by
32%. The DT tracks its teacher's mean at the same fidelity in both arms.

## It generalises to every intervention that works

| arm | centre dist | rel% | vs control |
|---|---|---|---|
| MF-DRO control (plain) | 0.852 | 15.82 | — |
| L1-LOSS (loss change) | 0.890 | 13.47 | −2.34 |
| REFINE-100 (teacher refinement) | 0.928 | 9.96 | −5.85 |
| ROI-L1 (best ROI arm) | **0.981** | **9.81** | **−6.00** |

**All three of the project's surviving interventions move the DT's constant further from
the centre than the control's, and the ordering is monotone: further out, better. None
that helps moves it inward.**

So the ROI, teacher refinement and the L1 loss are not three separate mechanisms. **They
are one mechanism: relocating the point the DT memorises.** None of them makes the DT
smarter — it is a constant predictor before and after.

## Honest limits

- **The ROI-vs-non-ROI group split is NOT clean.** ROI arms span 0.875–1.007 (n=9,
  mean rel% 11.40); non-ROI working arms span 0.752–0.928 (n=17, mean 16.73). The
  groups **overlap by 0.053**, with L1-LOSS and REFINE-100 reaching into the ROI range.
  That overlap is reported because it is real — and it happens to *support* the account,
  since those two arms are precisely the project's other successful interventions.
- **Correlational.** Nothing here intervenes on the teacher's mean directly and observes
  the DT. The L=1 pair is the nearest thing to a controlled comparison, and it is one
  pair.
- n=5 seeds per arm, Borehole only. Only one ROI arm (h176) carries
  `teacher_action_stats`, so the teacher-side half of the argument rests on that arm.

## What could RETRACT it

- An intervention that helps while moving the constant *inward*. None of the three does.
- A direct intervention on the teacher's τ=0 mean that fails to move the DT's query.
  That experiment does not exist and would be the proper causal test.
