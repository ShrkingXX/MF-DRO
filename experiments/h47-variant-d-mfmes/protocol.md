# H47 — Variant D: MF-MES with the acquisition actually optimized

**Status at write time: CONFIRMATORY. Protocol committed before any run.**

## Why

Every "MF-MES" number in this project comes from argmaxing the Takeno
acquisition over **200 uniform random points** (`mf_dro.py:2644`,
`n_infer_candidates=200` at `mf_dro.py:1806`). Takeno et al. 2020
(`papers/MF-MES.pdf`, §3.4) specify something different for continuous spaces:

> "For the acquisition function maximization (argmax in line 4), if the
> candidate space X is a discrete set, we simply calculate the acquisition
> values for all x ∈ X. For a continuous space, popular approaches such as
> DIRECT (Jones et al., 1993) and gradient-based optimizers are applicable.
> Note that our acquisition function is differentiable..."

Hartmann 6D is continuous. So our baseline is not the published method.

### Measured size of the handicap (exploratory, pre-protocol)

Holding y* fixed from an independent 4000-point reference set so scores are
comparable across pool sizes (Hartmann 6D, seed 42, post-initial-design GP):

| N | best HF/c_H | best LF/c_L | best | vs N=200 |
|---|---|---|---|---|
| 200 | 0.00886 | 0.02275 | 0.02275 | 1.00× |
| 500 | 0.01078 | 0.03310 | 0.03310 | 1.46× |
| 1000 | 0.02395 | 0.06344 | 0.06344 | 2.79× |
| 2000 | 0.02395 | 0.06344 | 0.06344 | 2.79× |
| 4000 | 0.03487 | 0.09778 | 0.09778 | 4.30× |

Not saturated at 4000. A bigger pool alone is not the fix; the optimizer is.

**Second handicap, found while debugging the above.** `compute_joint_mf_mes`
Thompson-samples y* *from the same pool it then argmaxes over* (line 32 of that
function). So the sparse pool also corrupts the estimate of the optimum's
value. This is self-concealing: scored by its own y*, the 200-point version
reports 0.16004 while variant D reports 0.16080 — indistinguishable. Scored
under a **common** y*, the same two proposals are 0.05407 vs 0.19383 (3.58×).
The pool version cannot detect its own under-optimization, because
underestimating y* inflates apparent information gain.

## What changes

Exactly one thing: the proposal step. `src/policy/mf_mes_optimized.py`:

1. y* Thompson-sampled from an independent `n_ref=2000` reference set
   (Takeno estimates the optimum over the search space, not over the pool).
2. Two-stage batched maximization of the joint cost-normalized acquisition
   over (x, ℓ): dense stage `n_dense=2000`, then shrinking-ball local
   refinement — `n_starts=8`, `n_refine=4` rounds × `n_pert=32`, radius
   0.10 → 0.0125. ~3000 acquisition evaluations, measured 0.47 s per call.

Everything else — initial design, cost accounting, regret curve, benchmark,
seeds — is h31's code path verbatim, so D is comparable to the table below.

## Held fixed

Hartmann 6D · seeds 42–51 · `cost_budget=200` post-init ·
`initial_hf=36, initial_lf=60` · `ITER_CAP=250` (runaway guard; analysis flags
any run it binds) · `c_H`/`c_L` unchanged.

## Comparison (existing, 10 seeds, same budget)

| method | final HF regret |
|---|---|
| MF-MI-Greedy | 0.5091 ± 0.1266 |
| MF-MES teacher, pool-200, no DT | 0.4781 ± 0.0414 |
| MF-DRO / joint MES | 0.4007 ± 0.0475 |
| **variant D (this)** | **?** |

## Prediction

**Primary: variant D beats the pool-200 teacher (< 0.4781).** A 3.58× better
acquisition optimum should translate into better queries.

**I am flagging the opposite outcome as genuinely live, not as a hedge.** We
measured (h28) that the teacher's choices sit at the **2.9th percentile of
posterior σ_H** — MF-MES here is already strongly exploitative. Optimizing the
acquisition *harder* optimizes that exploitation harder. If D lands at or above
0.4781, the reading is that the acquisition, not its optimizer, is the ceiling —
which is the same conclusion the DT work reached from the other side.

Both outcomes are informative. Recorded before running so neither can be
narrated as the expected one afterwards.

## Analysis plan (fixed now)

Report mean ± s.e. of final HF regret over the 10 seeds; paired per-seed diff
vs the pool-200 teacher (same seeds, same initial design → pairing is valid);
Wilcoxon signed-rank on the 10 paired diffs. **No metric selection**: final HF
regret at the budget is the single pre-declared endpoint. Anything else I look
at gets labelled EXPLORATORY.

---
## KILLED 2026-08-25 — redundant with H48, not a result

Killed at ~3.5 h with 0/10 seeds. Two reasons:

1. **Redundant.** h47D overrides `propose_mf` entirely, so the DT decides
   nothing: it is MF-MES-with-a-real-optimizer *replacing* the DT, inside the
   DRO harness. H48 isolates exactly that, standalone, in ~1.5 min/seed
   instead of ~4 h/seed, and additionally validates the acquisition (V1-V6).
2. **Wrong baseline.** Its comparison target was MF-DRO/scoring-head (0.4007).
   h45 shows the **regression head** is the stronger MF-DRO (0.3711 vs 0.4523
   on 6 shared seeds, better on 5/6), so the scoring head is no longer the
   reference configuration.

Note for the record: h47D's config had `use_candidate_scoring=False` (the
`_build_mf_dro_config` default), i.e. the regression-head config — but the head
was irrelevant, since `_propose` bypassed both heads.

No conclusions are drawn from this directory.
