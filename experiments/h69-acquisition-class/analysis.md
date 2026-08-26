# H69 result — NULL. Acquisition class is NOT the lever.

**CONFIRMATORY** against `protocol.md`. Regression gate passed **bit-for-bit**
(MESCHECK reproduced h59's SF-MES Borehole seed44 to 0.000e+00), so SF-EI differs
from SF-MES in the acquisition and nothing else.

## Verdict

| prediction | bar | result | verdict |
|---|---|---|---|
| PRIMARY | Borehole: SF-EI wins >=2/3 **and** >=2 pts on mean | wins **1/3**, mean +0.29 pts | **NOT MET** |
| CONTROL | Hartmann: SF-EI worse-or-equal | SF-EI **better** by 1.50 pts, wins 2/3 | **VIOLATED** |
| NULL | no separation on Borehole | +0.29 pts | **FIRED** |

| benchmark | seed | SF-MES | SF-EI | winner |
|---|---|---|---|---|
| Borehole | 44 | 15.14% | 12.73% | SF-EI |
| Borehole | 46 | 13.08% | 13.08% | tie/SF-MES |
| Borehole | 48 | 11.63% | 13.16% | SF-MES |
| Borehole | **mean** | **13.28%** | **12.99%** | +0.29 pts |
| Hartmann | 44 | 22.67% | 22.67% | tie/SF-MES |
| Hartmann | 46 | 30.74% | 29.50% | SF-EI |
| Hartmann | 48 | 10.76% | 7.50% | SF-EI |
| Hartmann | **mean** | **21.39%** | **19.89%** | +1.50 pts |

## What this establishes

Swapping the information-seeking acquisition (MES) for an improvement-seeking one
(EI), **with the surrogate held exactly fixed**, moves Borehole by 0.29 points.
MI-Greedy's advantage over SF-MES on Borehole is **5.0 points** (13.28% -> 8.3%).
Acquisition class accounts for essentially none of it.

The CONTROL is what makes this readable. It was written to catch precisely this:
*"If EI helps on both, the story is not acquisition class at all — it is simply
that EI is a better acquisition here."* EI helped on both (+0.29 Borehole,
+1.50 Hartmann). So the Borehole/Hartmann ordering flip in the standings table —
EI best on Borehole, worst on Hartmann — is **not** an acquisition-class effect.
The most likely explanation for that flip is the confound the protocol already
flagged: MI-Greedy runs at **12% HF on Hartmann** versus **100% on Borehole**, so
its Hartmann column is a fidelity result, not an acquisition result.

> **h68's "model failure" now narrows to the SURROGATE.** Acquisition is
> eliminated as the explanation for Borehole with one difference isolated and a
> bit-for-bit gate behind it.

## The specific remaining difference

MI-Greedy builds its HF GP with `mf_baselines._build_gp`; SF-MES/SF-EI use
`_build_ko_style_gp`. Both are SingleTaskGP + RBF/ARD + ScaleKernel +
Normalize/Standardize, but the KO-style builder additionally imposes an
**Interval lengthscale constraint with geometric-mean initialisation**. That is
now the sole identified difference between a method at 8.3% and a method at
12.99% on Borehole, and it is directly testable by swapping the builder — one
difference, same acquisition, same loop.

## Honest accounting of the predictions

Both locked predictions failed: PRIMARY not met, CONTROL violated. The
*hypothesis* was wrong. The *protocol* worked — the CONTROL was designed to
distinguish "acquisition class matters" from "EI is just better", and it did.
Running numbers 3 and 4 (NULL / REVERSED) were pre-specified and NULL is what
fired.

Prediction record for this project is now poor enough that it is itself a
finding: mechanism intuitions here are reliably wrong, and pre-registered
discriminators are reliably informative. That is the pattern to keep exploiting.

## What this cannot settle

n=3. SF-EI is greedy single-fidelity, so this says nothing about whether an
improvement-aware reward would help *inside* DRO's rollout. And EI's small
consistent gain (+0.29, +1.50) is not separable from noise at n=3 — it is
reported as a direction, not a result.
