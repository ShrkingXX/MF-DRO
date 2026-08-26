# H70 result — MI-Greedy's Borehole advantage is 100% CANDIDATE POOL SIZE

**CONFIRMATORY** against `protocol.md`. Reproduction control passed bit-for-bit
(h70's SF-EI reproduced h69's to 0.000e+00).

## Verdict — all three predictions met

| prediction | bar | result | verdict |
|---|---|---|---|
| PRIMARY | POOL1000 <= 10.6% on Borehole | **8.27%** | **MET** |
| SECONDARY | ALTGP moves Borehole < 1 pt | **0.00 pts** | **MET** |
| CONTROL | POOL1000 must not hurt Hartmann | 19.89% -> 13.41% | **MET** |

| arm | Borehole | per seed | Hartmann | per seed |
|---|---|---|---|---|
| SF-MES (h59) | 13.28% | 15.1 / 13.1 / 11.6 | 21.39% | 22.7 / 30.7 / 10.8 |
| SF-EI (h69) | 12.99% | 12.7 / 13.1 / 13.2 | 19.89% | 22.7 / 29.5 / 7.5 |
| SF-EI + ALTGP | 12.99% | 12.7 / 13.1 / 13.2 | 13.48% | 15.7 / 17.2 / 7.5 |
| **SF-EI + POOL1000** | **8.27%** | **7.1 / 6.8 / 10.9** | 13.41% | 22.7 / 10.3 / 7.3 |
| MI-Greedy (h57) | **8.27%** | **7.1 / 6.8 / 10.9** | 23.93% | 18.6 / 32.6 / 20.6 |

**POOL1000 does not merely close the 4.72-point gap — it reproduces MI-Greedy
exactly, seed for seed.** Residual +0.00. With a 1000-point pool, EI on a GP over
HF data *is* MI-Greedy on Borehole, which is what MI-Greedy reduces to when its
LF phase is inert at 100% HF.

The GP construction contributes **exactly zero** on Borehole. It is not a null
result from an inactive arm: the two builders were verified to produce materially
different models (max lengthscale difference **4.27**; e.g. 0.43 vs 4.70 on one
dimension). Different surrogates, identical EI argmax sequence, identical regret.

## The consequence: h57's comparison is not acquisition-effort-matched

Pool sizes read from source:

| method | acquisition pool |
|---|---|
| **MF-DRO** | **200** (`n_roi_candidates`) |
| SF-MES / SF-EI | 200 (`GreedyMFBase.n_candidates`) |
| MF-MI-Greedy | **1000** (`_CANDIDATE_POOL`) |
| MF-GP-UCB | **1000** |
| MF-MES (Takeno) | **2048 Sobol + top-K L-BFGS-B refinement** |

**CORRECTED after this file was first written.** MF-DRO's 200 is the *rollout
teacher's* pool, not an inference-time acquisition pool: `_propose_next_query`
builds no candidate set, and the regression head emits `action_head(h).clamp(0,1)`
directly, so **MF-DRO does no inference-time acquisition search at all**. The
"5-10x effort gap" framing was wrong — these are different mechanisms, not two
sizes of one. The accurate statement is a categorical asymmetry: baselines run a
1000-2048-point argmax per query; MF-DRO runs none.

The Borehole measurement itself is unaffected: for the *greedy baselines*, pool
size alone is worth 4.72 points and reproduces MI-Greedy exactly.

This is consistent with, and retrospectively explains, h61 (widening MF-DRO's
Borehole pool bought 1.44x acquisition value) and h64's POOL600 direction. It does
**not** rescue MF-DRO: it sits at 23.7% while a 1000-pool greedy EI reaches 8.27%.
But it does mean the *size* of MF-DRO's deficit has never been measured under a
matched acquisition budget.

## EXPLORATORY — not predicted

The KO-style GP construction **costs 6.41 points on Hartmann**: ALTGP (plain
`mf_baselines._build_gp`) reaches 13.48% against SF-EI's 19.89%, winning 2 seeds
and tying the third. So `_build_ko_style_gp`'s LogNormal lengthscale prior and
geometric-mean initialisation — used by every DRO and MES arm in this project —
appears actively harmful on Hartmann while being exactly neutral on Borehole.
n=3, unpredicted, and it needs its own protocol before it is claimed.

## What this cannot settle

n=3. It explains a gap between greedy single-fidelity baselines and does not fix
MF-DRO. Whether a pool-matched h57 changes the standings is untested — that is
now the highest-value open experiment, and it is a *method/harness* question, not
a change to the frozen evaluation.
