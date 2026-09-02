# h157 -- BLIND FORECASTS, committed before either arm reported

Recorded when h153 was unfinished and h155 was at cost=186.0/240 with **0/5**
result files written. Both forecasts are therefore blind.

## Harness readout (6 states, N=100, 10-model ensemble, 2 replicates)

| condition | rep0 | rep1 | mean | % of control |
|---|---|---|---|---|
| control (MES, closed) | 1.0269 | 0.9078 | 0.9673 | 100.0% |
| **C2 = h153 MES-FROZEN (open)** | 0.8457 | 0.7439 | **0.7948** | **82.2%** |
| **C6 = h155 UCB-LOC (closed)** | 1.0084 | 0.9655 | **0.9869** | **102.0%** |
| RANDOM-POOL | 0.3330 | 0.3308 | 0.3319 | 34.3% |
| ORACLE | 0.2579 | 0.2423 | 0.2501 | 25.9% |
| DIVERSE-GOOD | 0.2646 | 0.2695 | 0.2670 | 27.6% |

Failing-arm band: 25.9-34.3% of control. C6 sits at **3.49x** that band.

## The forecasts

**h153 (MES-FROZEN):** rtg_target retains **82-96%** of the control's across all
five harness measurements (82.2, 87.3, 90.9, 95.5, 95.9) -> ~0.80-0.94, NOT
~0.31. Performance predicted **near the control**, not 43.94.
(The band widens from the 85-96% quoted last tick because this run is coarser --
6 states, N=100 -- traded deliberately for blindness.)

**h155 (UCB-LOC):** rtg_target ~**102%** of the control -> ~0.98. Performance
predicted **near the control**, not 43.94. UCB with beta=2 targets
high-variance points, which is information-seeking, so it earns a comparable
tail. It is NOT predicted to sit between the control and the failing arms.

## Which outcome this commits me to

Both forecasts point at **O1** in protocol.md: the operative variable is
INFORMATION GAIN, not loop type and not the MES rule specifically.

If h153 fails -> **O3**, and the entire h156 tail account is refuted and comes
out of findings.md and the published report.
If h155 fails while h153 succeeds -> **O2**, and the account narrows to
MES-derived teachers; "any information-seeking teacher suffices" is retracted.
If either lands at 25-35 rel% -> **O4**, inconclusive for that cell.

Also to be checked before h155's number is read: its realised HF fraction
against the control's ~0.98. A collapse voids the cell regardless of regret.
