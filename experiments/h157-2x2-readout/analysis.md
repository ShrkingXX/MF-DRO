# h157 readout — h155 CELL FILLED (n=3/5). Blind forecast CONFIRMED.

CONFIRMATORY. Protocol, readout code and forecast all committed before any
result existed (h155 was at cost 184/240 with 0/5 files written).

| arm | rel% | improves | rtg_target | HF frac | n |
|---|---|---|---|---|---|
| control MES (closed) | 15.82 | 5/5 | 0.9761 | 0.88 | 5 |
| **h155 UCB-LOC (closed)** | **15.16** | **3/3** | **1.0418** | 0.93 | **3** |
| ORACLE (open) | 43.94 | 0/5 | 0.3113 | 0.63 | 5 |
| RANDOM-POOL (open) | 43.94 | 0/5 | 0.2965 | 0.26 | 5 |

## The blind forecast held

Forecast: rtg_target **~102%** of the control's. Observed: **106.7%**
(1.0418 / 0.9761). Error **+4.6 points**, inside the harness's own 8–13% noise
floor. Performance forecast "near the control, not 43.94, and NOT intermediate":
observed **15.16 vs 15.82**, improving **3/3**.

## Confound check, run before the number was read

HF fraction **0.93 vs control 0.88** — no collapse. h60's `thompson` arm
collapsed to 2/196 HF and was uninterpretable; h155 holds the fidelity channel
fixed by construction and the data confirms it. **The cell is valid.**

## What this retracts

**"The MES selection rule specifically is what the DT needs" is RETRACTED.**
That reading survived h150 (which retracted policy-distillation-of-MES) and has
been the fallback explanation since. A UCB teacher — a different acquisition,
β=2 on the HF posterior — matches the control on regret, on improvement rate,
and on the conditioning target.

Combined with the three failing arms at 43.94, the 2×2's non-MES row now reads:
closed-loop non-MES **works**, open-loop non-MES **fails**. The operative
variable is whether the teacher is **information-seeking**, not which
information-seeking rule it uses. UCB with β=2 targets high-variance points,
which earns the tail; interpolating toward a known-good point does not.

## Caveats

n=3 of 5; seeds 44 and 45 still running. No p-values. The h153 cell (MES,
open-loop) is still running at 205/240 and is the outcome that could still
refute the whole account (O3 in protocol.md).
