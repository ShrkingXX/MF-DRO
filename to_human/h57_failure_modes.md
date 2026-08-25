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

### Hartmann 6D — the low-fidelity function is a *more attractive* objective

**1. LF's optimum is higher than HF's.** Recovered by multi-start L-BFGS-B:

| | HF f(x*) | LF max |
|---|---|---|
| Hartmann 6D | 3.3224 | **4.0019  (+20%)** |
| Borehole 8D | 309.58 | 246.35 (lower) |
| Currin 2D | 13.80 | 13.58 (lower) |

Hartmann is the only one of the three where this is true.

**2. The policy chases it.** Seed 46 spent **176 LF queries against 3 HF**, and
**166 of those 176 LF values exceeded the true HF optimum** (LF median 3.783 vs
f(x*) = 3.322). The incumbent only updates on HF, so it barely moved.

**3. The fidelity split is unstable across seeds** that differ *only* by the
initial-design draw:

| method | s44 | s46 | s48 | spread |
|---|---|---|---|---|
| **MF-DRO** | 81% | **4%** | 28% | **76 pts** |
| MF-MES | 26% | 38% | 67% | 40 pts |
| MF-MI-Greedy | 12% | 13% | 11% | **2 pts** |

**4. Forcing HF fixes part of it (H58).** A 25% floor at inference — copied
verbatim from the rollout simulator, untuned — applied to seed 46:

| | queries | HF% | regret |
|---|---|---|---|
| free | 179 | 2% | 0.2875 |
| **floor** | **74** | 26% | **0.2230  (−22%)** |

**176 LF queries were worth less than 19 HF queries.**

> **Hartmann in one line:** the fidelity head is rewarded for querying a cheap
> surrogate whose optimum is 20% *above* the target function, so it starves the
> incumbent, which only HF observations can move.

---

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
