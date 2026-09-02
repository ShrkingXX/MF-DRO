# h157 -- the 2x2 readout, PRE-COMMITTED before h153/h155 land

STATUS: locked while h153 is at 157/240 and h155 at 165/240, both 0/5 finished.
TYPE: CONFIRMATORY. Analysis code and interpretation table are committed here
BEFORE any result exists, so neither can be chosen after seeing the numbers.

## The 2x2

|  | CLOSED-loop (adaptive) | OPEN-loop (frozen) |
|---|---|---|
| **MES rule** | control **15.82**, 5/5, rtg_target 0.9761 | **h153** |
| **non-MES rule** | **h155** UCB-LOC | ORACLE / DIVERSE-GOOD / RANDOM-POOL **43.94**, 0/5, ~0.30 |

## Registered forecasts

h153 (= harness C2, four measurements): rtg_target **85-96% of the control's**,
i.e. ~0.83-0.94, NOT ~0.31. Performance **near the control, not 43.94**.

h155 (= harness C6): forecast pending the C6 run launched at h155 cost=165/240
with 0/5 finished. Whatever it returns will be committed before h155 reports; if
h155 lands first, the forecast is NOT blind and will be reported as such.

## Interpretation table -- all four outcomes named now

O1  h153 near control AND h155 near control
    -> The tail account holds and the operative variable is INFORMATION GAIN,
       not the loop type and not the specific MES rule. Both cells that keep
       information-seeking behaviour work. Front closes.
O2  h153 near control AND h155 fails (~43.94)
    -> Information gain matters AND the MES rule is doing something UCB is not.
       RETRACTS the general claim "any information-seeking teacher suffices";
       the account narrows to MES-derived teachers.
O3  h153 FAILS (~43.94)
    -> **The tail account is REFUTED at its central forecast.** h156's whole
       chain -- max-not-mean, C2 retaining 85-96%, the h149 reinstatement --
       must be withdrawn from findings.md and from the published report. This
       is the outcome that costs the most and it is named first-class.
O4  Either lands in between (say 25-35 rel%)
    -> No clean attribution at n=5. Reported as inconclusive for that cell.
       Named now so an intermediate number is not read toward whichever
       hypothesis it sits nearer.

## Named confound for h155, checked BEFORE its number is read

Realised HF fraction. h60's `thompson` teacher collapsed the fidelity head to
2/196 HF, which confounds "different rule" with "different HF/LF mix". h155
holds the fidelity channel fixed by construction, but this must be VERIFIED,
not assumed. Control reference ~98% HF. If h155's mix collapses, the arm is
CONFOUNDED and no verdict is issued for that cell regardless of its regret.

## h153 sanity checks, read before its number

SC1 pass-2 path reproduces pass-1 (max |err| < 1e-9)
SC2 open-loop penalty > 0 inside the real pipeline
SC3 fidelity flip fraction between passes -- a large value means the frozen arm
    differs in HF/LF mix as well as in adaptivity, reintroducing a confound
All three are serialised in the run's `_h153` block by the worker.

## Metric

Frozen: rel% of |optimum| @cost_curve 200 via h83's sr_curve + grid. n=5,
Borehole seeds 42-46. **No p-values.** Improvement counted as any post-init HF
query beating the best initial HF value.
