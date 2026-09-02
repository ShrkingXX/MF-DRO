# h155 -- UCB-LOC: the missing cell of the 2x2

STATUS: protocol locked, nothing run.
TYPE: CONFIRMATORY.

## The 2x2 this completes

The whole programme has only ever filled two diagonal cells:

|  | CLOSED-loop (adaptive) | OPEN-loop (frozen / non-adaptive) |
|---|---|---|
| **MES rule** | control **15.82**, improves 5/5 | **h153** MES-FROZEN (running) |
| **non-MES rule** | **h155 UCB-LOC** (this) | ORACLE / DIVERSE-GOOD / RANDOM-POOL **43.94**, 0/5 |

h153 alone cannot distinguish "adaptivity is the channel" from "the MES rule
specifically is the channel" -- it only removes adaptivity from MES. h155
removes MES while KEEPING adaptivity. Both cells are needed.

## Why the existing thompson arm does NOT fill this cell

h60 ran rollout_policy="thompson" -- adaptive, non-MES -- and it failed hard
(43.8% vs BASE 23.7%). But it **collapsed the fidelity head to ~99% LF**
(2, 2 and 5 HF queries out of ~198), so it changed the HF/LF mix at the same
time as the location rule. It is confounded and cannot be read as the non-MES
closed-loop cell. Its numbers are also on h57/h60's metric, not the frozen h83
metric used here.

## Design: hold the FIDELITY channel fixed

New branch rollout_policy="ucb_loc" (mf_dro.py). LOCATION by UCB on the HF
posterior, beta=2.0, over the same roi_candidates pool -- adaptive, re-decided
every step from the current fantasy-conditioned model, and not MES. FIDELITY by
the SAME cost-normalised info-gain criterion the forced_x path already uses
(mf_dro.py:1618ff, the block h145 built and h145 v2 corrected). Only the
location rule varies.

BIT-IDENTITY GATE: **PASSED**. Ran the real pipeline on the pre-patch and
post-patch module (loaded into sys.modules under the canonical name) at
rollout_policy="mes": x, y, fidelity, rtg_target traces and final_regret
(122.2906675273) all identical. The default path is untouched.

## Predictions

ADAPTIVITY hypothesis: UCB-LOC lands far better than 43.94, near the control.
MES-SPECIFIC hypothesis: UCB-LOC fails with the frozen arms near 43.94.

## Named confound, checked before the result is read

Realised HF fraction. If UCB-LOC's HF mix collapses the way thompson's did, the
arm is CONFOUNDED and will be reported as such rather than as evidence -- the
comparison would then vary two things again. Control reference: ~98% HF.
Recorded per seed.

## Design

Borehole_8D seeds 42-46, n=5. Frozen metric: rel% of |optimum| @cost_curve 200
via h83 sr_curve+grid. No p-values at n=5. 5 workers; h153's 5 still running,
total 10 <= 15.

## What this can RETRACT

R1 UCB-LOC succeeds -> RETRACTS "the MES selection rule specifically is what the
   DT needs" (currently the surviving reading after h150 retracted policy
   distillation). Replaced by "any adaptive teacher suffices". Together with
   h153 this would close the front.
R2 UCB-LOC fails -> RETRACTS adaptivity as a SUFFICIENT explanation. Whatever
   h153 returns would then be at most half the story, and "adaptive AND
   information-gain-based" becomes the claim -- narrower than what findings.md
   currently sets up as h153's motivation.
R3 UCB-LOC's HF fraction collapses -> arm CONFOUNDED, no verdict, cell stays
   empty. Named now so a confounded failure is not read as R2.
