# h176 — **R2 fires.** The project's best result survives a one-step rollout.

CONFIRMATORY, n=5.

| arm | rel% | sd | improves | HF frac | wall | per-seed |
|---|---|---|---|---|---|---|
| control (no ROI, L=8) | 15.82 | 2.36 | 5/5 | 0.883 | 82.4 min | 15.3, 14.8, 12.9, 16.9, 19.2 |
| **ROI-Q10 (L=8)** | **11.59** | **0.41** | 5/5 | 0.739 | 117.4 min | 11.5, 12.3, 11.4, 11.2, 11.6 |
| L=1 (no ROI) | 13.69 | 4.24 | 5/5 | 0.939 | 13.2 min | 7.4, 11.1, 16.7, 16.8, 16.3 |
| **ROI-Q10 + L=1** | **10.81** | 2.79 | 5/5 | 0.787 | **24.6 min** | 8.8, 10.8, 13.2, 7.3, 13.9 |

**P1 holds** (10.81 against ROI-Q10's 11.59, within the 1.5 band). **P2 holds**:
24.6 min against 117.4 — **4.77×**. SC passes (HF 0.787 vs 0.739). **The ROI's
benefit survives the one-step rollout**, so the ROI acts through τ=0 — R1 does
not fire.

## My resolvability argument was flawed, and I made it confidently

Last tick I argued this comparison would be resolvable at n=5 *because* ROI-Q10's
per-seed sd is 0.41, and recorded that as applying the h172 lesson.

**Resolvability depends on both arms' spreads, and I only looked at the
baseline's.** ROI-Q10 + L=1 has sd **2.79**, and the paired difference is +0.78
with sd **2.72**. So:

- **Resolvable:** ROI+L1 (10.81) beats the control (15.82) and L=1 alone (13.69).
- **Resolvable:** the 4.77× speedup.
- **NOT resolvable:** whether ROI+L1 is better than ROI-Q10. The data is equally
  consistent with it being ~2 rel% worse.

So the claim is **"the ROI's benefit survives at roughly a fifth of the cost"**,
not "and improves it". Shortening the rollout also **costs stability**: sd rises
from 0.41 to 2.79, and ROI-Q10 at L=8 remains by far the most consistent arm.

That stability cost is a real trade-off and it is not visible in the means.

## What this establishes

The teacher front's answer and the project's contribution **compose**. The ROI is
applied once per rollout, outside the τ loop, so it shapes the pool τ=0 draws
from — and shortening the rollout to that single step keeps its benefit.

**Scope:** Borehole, n=5. Not tested on Hartmann, where h174's SC1 fired on the
plain L=1 arm and h173 showed the later steps do real work.
