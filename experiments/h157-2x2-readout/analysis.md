# h157 readout — h155 CELL FILLED (n=5/5, COMPLETE). Blind forecast CONFIRMED.

CONFIRMATORY. Protocol, readout code and forecast all committed before any
result existed (h155 was at cost 184/240 with 0/5 files written).

| arm | rel% | improves | rtg_target | HF frac | n |
|---|---|---|---|---|---|
| control MES (closed) | 15.82 | 5/5 | 0.9761 | 0.88 | 5 |
| **h155 UCB-LOC (closed)** | **15.13** | **5/5** | **1.0402** | 0.87 | **5** |
| ORACLE (open) | 43.94 | 0/5 | 0.3113 | 0.63 | 5 |
| RANDOM-POOL (open) | 43.94 | 0/5 | 0.2965 | 0.26 | 5 |

## The blind forecast held

Forecast: rtg_target **~102%** of the control's. Observed: **106.6%**
(1.0402 / 0.9761). Error **+4.6 points**, inside the harness's own 8–13% noise
floor. Performance forecast "near the control, not 43.94, and NOT intermediate":
observed **15.13 vs 15.82**, improving **5/5** — matching the control exactly on
improvement rate.

Per-seed, frozen metric:
```
control  [15.28, 14.77, 12.93, 16.90, 19.19]  mean 15.82
UCB-LOC  [15.03, 11.79, 13.88, 16.29, 18.67]  mean 15.13
```
UCB-LOC is better on 4 of 5 seeds and tracks the control seed-by-seed — the two
rise and fall together, which is what two teachers solving the same problem the
same way look like, and is not what the failing arms' flat 43.94 looks like.

## Confound check, run before the number was read

HF fraction **0.87 vs control 0.88** — no collapse, and closer than at n=3. h60's `thompson` arm
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

n=5, COMPLETE. No p-values computed at n=5. The h153 cell (MES,
open-loop) is still running at 205/240 and is the outcome that could still
refute the whole account (O3 in protocol.md).
