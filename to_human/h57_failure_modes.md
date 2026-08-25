# MF-DRO failure modes on Hartmann 6D and Borehole 8D
### Evidence from H57 (36 runs), H58 (12 runs), H59 (18 runs). Every number below is measured.

**Setup common to all:** cost budget 200 post-init, seeds 44/46/48, all methods
share the identical initial design, one pinned code commit recorded in every
result file. Regret = f(x*) − best HF value found. "rel" = regret / f(x*).

---

## Headline: MF-DRO never beats the best baseline

| method | Currin 2D | Hartmann 6D | Borehole 8D |
|---|---|---|---|
| **MF-DRO** | 0.0% | 14.7% | 23.7% |
| MF-MES (Takeno) | 0.6% | **8.5%** | 11.3% |
| MF-MI-Greedy | 0.2% | 23.9% | **8.3%** |

Paired wins for MF-DRO out of 3 seeds: **0-3 to MF-MES on Hartmann**,
**0-3 to both baselines on Borehole**. Currin does not discriminate — every
non-degenerate method finishes inside 0.6% of the optimum.

Wall clock: MF-DRO 82–114 min per run; MF-MI-Greedy 0.4–0.7 min. **120–230×.**

---

## THE TWO FAILURE MODES ARE DIFFERENT

### Hartmann 6D — RETRACTED. The original claim here was wrong.

**What I claimed:** the LF function's optimum (4.0019) exceeds HF's (3.3224), so
the fidelity head "chases" it and starves the incumbent.

**Why it is wrong — two independent reasons, both checked in code and data:**

**1. Nothing in the method scores LF by its own value.** The LF branch
(`_compute_mes_lf_vectorized`) is scored against `y_star_arr`, documented in
source as "shared **HF** Thompson samples". It computes information gain about
the *high-fidelity* optimum through `rho`, `mu_H`, `var_delta`. LF's own optimum
never enters any decision. And at real inference the fidelity is drawn from the
DT's `fidelity_head`, not from the MES argmax at all.

**2. The LF function is an EXCELLENT surrogate here, not a misleading one.**

| measurement | value |
|---|---|
| domain-wide corr(f_LF, f_HF) | **+0.925** |
| mean f_HF in the top-5% LF region | 1.4398 (domain mean 0.2588) |
| true HF value at the 176 LF-queried points, median | **3.0287 = 91% of f(x\*)** |

The LF budget went to a region genuinely containing near-optimal HF points. The
policy was not misled; it correctly located the good region cheaply.

**3. And the seed I called pathological is the BEST one.**

| seed | HF% | queries | regret | rel |
|---|---|---|---|---|
| **46** | **2%** | 179 | **0.2875** | **8.7%** |
| 48 | 28% | 67 | 0.4228 | 12.7% |
| 44 | 81% | 31 | 0.7531 | 22.7% |

**Regret is monotone in HF fraction, in the direction opposite to my claim.** The
LF-heaviest seed wins; the HF-heaviest loses. At c_H=8 vs c_L=1 with rho=0.925,
spending on the cheap fidelity is correct, and seed 44's 25 expensive HF queries
bought a worse answer than seed 46's 176 cheap ones.

**How the error happened:** I observed that 166/176 of seed 46's LF queries
exceeded f(x*)_HF and read it as evidence of chasing. The statistic is real and
extreme — uniform sampling puts only 0.1% of LF values above that line, so the
concentration is 94.3% vs 0.1% — but it measures that the policy found the good
region, not that it was lured somewhere bad.

**What remains true about Hartmann:** the fidelity split varies 81%/4%/28%
across seeds differing only by the initial-design draw, and h58's 25% floor
improved seed 46 by 22%. Both stand. But "the fidelity head chases a misleading
LF optimum" is withdrawn, and with it the framing that LF-heaviness is the
Hartmann failure mode.

### Borehole 8D — no fidelity problem at all; the search plateaus short

**1. It is already effectively single-fidelity.** 98–100% HF, spread 4 points
across seeds; only **1–3 LF queries per run**. There is nothing to misallocate.

**2. It plateaus ~25% short of the optimum:**

| seed | best HF found | f(x*) | shortfall |
|---|---|---|---|
| 44 | 233.94 | 309.58 | 24.4% |
| 46 | 232.99 | 309.58 | 24.7% |
| 48 | 241.61 | 309.58 | 22.0% |

**3. The search is not bad.** Stalled queries sit at the **99.7–99.9th
percentile** of the domain's value distribution (20k Sobol reference), against
incumbents at 99.8–99.9th. It is searching the very top of the landscape and
still finishing 25% short.

> **Borehole in one line:** fidelity allocation is fine and the queries are in
> the top 0.3% of the domain by value, yet the incumbent converges ~25% below
> the optimum and stops — a search-resolution failure, not a fidelity failure.

---

## What was RULED OUT by measurement (do not put these on the slide as causes)

| ruled out | evidence |
|---|---|
| **Incumbent freeze** (same point re-proposed) | `distinct == n_queries` in **all 9** MF-DRO cells, up to 116 queries |
| **Aimless / low-value search** | stalled HF queries at the 88.7–99.9th percentile of domain value |
| **Searching far from the optimum** | retracted: on Hartmann the best value within 0.3 of x* (2.9196) equals the best beyond 1.0 away (2.8338), so distance to x* carries no information |

---

## Caveats to state out loud

- **n = 3 seeds per cell. No p-values.** Directions and paired win counts only.
- The best baseline **changes identity by benchmark** (MF-MES on Hartmann,
  MI-Greedy on Currin and Borehole), so "at least as good as the baselines" is a
  per-benchmark bar, not one number.
- **MF-GP-UCB is not a competitor** — it is structurally all-LF on all three
  benchmarks (its cost ratio exceeds the fidelities' prior variance ratio
  everywhere), so its incumbent only moves on initial-design points. Present it
  as a floor or omit it.
