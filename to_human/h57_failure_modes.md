# MF-DRO failure modes on Hartmann 6D and Borehole 8D
### Evidence from H57 (36 runs), H58 (12 runs), H59 (18 runs). Every number below is measured.

**Setup common to all:** cost budget 200 post-init, seeds 44/46/48, one pinned
code commit recorded in every result file. Regret = f(x*) - best HF value found.
"rel" = regret / f(x*).

**Initial design is NOT symmetric - say this if asked.** All *multi-fidelity*
arms (H57) share an identical initial design. The *single-fidelity* arms (H59)
get the HF init only, no LF points. The initial design is also **not charged**
against the 200-unit budget, so MF arms receive free low-fidelity information
worth **+15 / +45 / +20** cost units on Currin / Hartmann / Borehole - on
Hartmann that is 22.5% of the optimization budget. This runs *against* MF, so
wherever SF beats MF below, that is a conservative statement rather than an
artifact.

---

## Headline: MF-DRO never beats the best baseline

![standings](north_star_standings.png)

| method | Currin 2D | Hartmann 6D | Borehole 8D |
|---|---|---|---|
| **MF-DRO** | 0.0% | 14.7% | 23.7% |
| **SF-DRO** | 0.4% | 11.5% | 15.1% |
| MF-MES (Takeno) | 0.6% | **8.5%** | 11.3% |
| SF-MES | **0.0%** | 21.4% | 13.3% |
| MF-MI-Greedy | 0.2% | 23.9% | **8.3%** |
| MF-GP-UCB | 10.0% | 45.3% | 44.1% |

**No DRO variant is best on any benchmark except Currin**, and Currin does not
discriminate. Note SF-DRO beats MF-DRO on both hard benchmarks *despite* getting
no free LF initial design.

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

### Borehole 8D — not a fidelity problem, and not a late plateau either

**1. It is already effectively single-fidelity.** 98-100% HF, spread 4 points
across seeds; only **1-3 LF queries per run**. There is nothing to misallocate,
so no fidelity-based explanation can apply here.

**2. It finishes 22-25% short of the optimum:**

| seed | best HF found | f(x\*) | shortfall |
|---|---|---|---|
| 44 | 233.94 | 309.58 | 24.4% |
| 46 | 232.99 | 309.58 | 24.7% |
| 48 | 241.61 | 309.58 | 22.0% |

**3. The gap opens EARLY and never closes.** Indexed by HF query number (not
cost), MF-DRO and MI-Greedy share the identical 10-point HF init and both start
at 72.07:

| HF query # | MF-DRO | MI-Greedy |
|---|---|---|
| 0 (shared init) | 72.07 | 72.07 |
| 10 | 173.07 | 219.71 |
| **20** | 190.38 | **264.41** |
| 109 | 236.18 | 283.97 |

**MI-Greedy reaches 264.41 within 20 HF queries; MF-DRO does not reach it in
109.** So this is *not* a method that converges and stalls — it is one whose
first ~20 high-fidelity evaluations are worth about a third less than a greedy
baseline's, from an identical starting point.

**4. Calibrated against random search**, which puts the size of the deficit in
context:

| strategy | best HF found |
|---|---|
| random search, 100 evals | 200.94 |
| **MF-DRO, 99 HF queries** | **236.18** |
| random search, 1000 evals | 240.32 |
| random search, 20000 evals | 266.08 |
| **MI-Greedy, 100 HF queries** | **283.97** |

MF-DRO's 99 queries buy roughly what 1000 random draws buy — it *is* optimising,
about 10x better than random. MI-Greedy's 100 beat what **20,000** random draws
buy. Same budget, same start.

**5. The search is not in the wrong place.** Stalled queries sit at the
**99.7-99.9th percentile** of the domain's value distribution (20k Sobol
reference), against incumbents at 99.8-99.9th.

> **Borehole in one line:** fidelity is inert, the queries are in the top 0.3% of
> the domain by value, and the deficit is concentrated in the first ~20
> high-fidelity evaluations rather than in a late plateau.

**Mechanism: unresolved. Say so.** Three geometric explanations were proposed
and each was refuted by measurement — (a) an uninformative LF corrupting the
surrogate (corr(f_LF,f_HF) = **1.000** on Borehole), (b) failure to refine
locally (MF-DRO is the *only* method that contracts, 0.68x, and it loses), and
(c) boundary aversion preventing it reaching a corner optimum (its queries are
the **closest** to x*, 0.900 vs 1.030 and 1.175, and it is the worst). A
follow-up ablation (H60) then excluded the reward schema and the LF initial
design, and showed the rollout teacher is load-bearing — swapping it moved regret
23.7% -> 43.8%. Two candidates remain untested: teacher optimisation quality and
the surrogate class.


---

## What was RULED OUT by measurement (do not put these on the slide as causes)

| ruled out | evidence |
|---|---|
| **Incumbent freeze** (same point re-proposed) | `distinct == n_queries` in **all 9** MF-DRO cells, up to 116 queries |
| **Aimless / low-value search** | stalled HF queries at the 88.7–99.9th percentile of domain value |
| **Searching far from the optimum** | retracted: on Hartmann the best value within 0.3 of x* (2.9196) equals the best beyond 1.0 away (2.8338), so distance to x* carries no information |

---

---

## Threads CLOSED by measurement — do not re-open without a new reason

Each of these was a live hypothesis about why MF-DRO underperforms. Each is now
answered, and the answer is negative.

| thread | verdict | key evidence |
|---|---|---|
| **Fidelity allocation** | not the lever | corr(HF%, regret) = −0.685 overall but **+0.071** excluding a 2%-HF outlier; regret spans only 19.3–24.7% across 18–98% HF |
| **rho / KO misspecification** | rho is a *regulariser*, not a slope estimate | fitted rho is 0.84 / 0.78 / 0.84 vs true slopes 1.26 / 0.98 / 1.01 — **the (0,1) ceiling never binds anywhere**. Pinning rho to the *true* slope was **−16.6%** on Hartmann, where it is representable |
| **Incumbent stall** | stall length carries no signal | in HF-opportunity units MF-DRO and MI-Greedy tie at **34%** terminal stall with opposite regret; MI-Greedy has the *lowest* terminal stall on Hartmann (2%) and the *worst* regret there |
| **"DRO buys consistency, not mean"** | refuted | DRO has lower sd in 4/6 pairs but lower **worst-case** regret in only **2/6**. Borehole SF-DRO: lower sd (1.02 vs 1.77) around a *worse* mean **and** a worse worst case |

The rho thread matters most: it retires an entire family of follow-ups ("fit rho
more faithfully" — better link, more optimizer steps, OLS init), because h63's
own control shows the *faithful* rho is not the *good* rho. A planned experiment
(H67, unbounded rho via softplus) was built, its regression gate passed, and it
was **cancelled before launch** when a three-minute pre-flight refuted its
premise — 6 jobs avoided.

---

## RESOLVED — and the answer is a WITHDRAWAL

**Did widening the acquisition candidate pool (200 -> 600) help on Hartmann?**
At n=3 it measured 7.6% vs MF-MES's 8.5% and was announced as this project's
first result beating a baseline. Replicated at **n=10**, with the analysis script
committed while the arm stood at 0/7:

| | POOL600 | MF-MES |
|---|---|---|
| mean | 0.2207 (6.6%) | 0.2737 (8.2%) |
| **paired wins** | **5/10** | 5/10 |
| Wilcoxon p | **1.0000** | |

The pre-registered failure branch (<=5/10 wins) **fired**.

**The mean advantage is a single seed.** It comes almost entirely from seed 49,
where MF-MES posts its worst run of the ten (0.8390 vs POOL600's 0.0667).
Excluding that seed the advantage **reverses** — MF-MES ahead by 0.0270. The
median gap is 0.0103. The protocol demanded both a better mean *and* >=6/10 wins
precisely because at n=10 one catastrophic baseline run moves a mean further than
the effect being tested.

> **Say this plainly on the slide: the claim is withdrawn. No DRO variant in this
> project beats the best baseline on any benchmark that discriminates, and none
> ever did.**

This is the second time the same shape has appeared — h45 read 5/6, then 7/8,
then finished worst-on-mean at 10/10. The difference is that this time the
withdrawal condition was written down before the data existed, so the result was
read rather than argued about.

## Caveats to state out loud

- **n = 3 seeds per cell. No p-values.** Directions and paired win counts only.
- The best baseline **changes identity by benchmark** (MF-MES on Hartmann,
  MI-Greedy on Currin and Borehole), so "at least as good as the baselines" is a
  per-benchmark bar, not one number.
- **MF-GP-UCB is not a competitor** — it is structurally all-LF on all three
  benchmarks (its cost ratio exceeds the fidelities' prior variance ratio
  everywhere), so its incumbent only moves on initial-design points. Present it
  as a floor or omit it.
