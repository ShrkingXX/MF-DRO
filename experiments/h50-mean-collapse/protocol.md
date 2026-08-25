# H50 — does the regression head's output sit BETWEEN the teacher's modes?

**CONFIRMATORY. Protocol committed before any run.**

## The claim being tested

h45 (10/10 seeds, regression head) found a failure mode that is NOT the
incumbent freeze: seeds 49 (regret 1.3640) and 50 (1.1818) had **zero
improvements** while proposing a **distinct point every iteration**
(distinct/iters = 144/144 and 74/74). Fresh proposals forever, none ever
beating the incumbent.

The proposed mechanism (Change 1a's argument, restated by the peer session):
MSE regression onto a **multimodal** teacher argmax distribution minimises to
the conditional **mean**, which lands *between* the modes — a point no mode
endorses, hence distinct-but-useless queries.

This is currently a hypothesis. The only data touching it is h45's `x_spread`
column, where the 2 failing seeds explore **0.72x** as widely as the 8
succeeding ones (0.1385 vs 0.1912) — directionally consistent with collapsing
onto a region, but n=2, and `x_spread` is computed over only the first 40
queries. H50 measures the thing itself.

## Design

Four seeds, chosen as the two extremes of h45's bimodal outcome:

| group | seeds | h45 regret | improvements |
|---|---|---|---|
| FAIL | 49, 50 | 1.3640, 1.1818 | 0, 0 |
| PASS | 42, 44 | 0.2051, 0.1157 | 2, 9 |

Configuration is **h45's verbatim** (regression head,
`use_candidate_scoring=False`, `rollout_reward="mes_entropy"`,
`cost_budget=200` post-init, `initial_hf=36 / initial_lf=60`,
`dkl_threshold=9999`, `num_epochs=10`, `rollout_length=8`, Hartmann 6D).
Seeds are the same, so each run reproduces its h45 counterpart.

**No `mf_dro.py` edit.** Instrumentation wraps `mf.dt.propose_mf` from the
worker, the same pattern h31/h47 used. The wrapper calls the original, then
measures the teacher at that exact GP state. It cannot change the trajectory:
it returns the original's output unmodified.

## What is measured, per real BO iteration

**Teacher argmax distribution.** On the SAME 200-point uniform pool the real
proposal uses, call `compute_joint_mf_mes` once per (ensemble member m, draw r)
for m in 0..9, r in 0..4 -> **50 argmax samples**. Each call redraws its own
Thompson y*, so the 50 samples span both ensemble disagreement and y*
uncertainty — the two sources of spread the DT's training data actually has.
Pool size 200 matches the teacher the DT distilled, so modes are genuine pool
points rather than an artefact of a denser grid.

**Mode structure.** Cluster the 50 samples in normalised [0,1]^6 with average-
linkage hierarchical clustering at distance 0.15. Record `n_modes`, mode
weights, centroids, and `mode_sep` (weighted mean pairwise centroid distance).

**Where the DT sits.**
- `d_nearest_mode` = min_i ||x_dt - c_i||
- `d_teacher_mean` = ||x_dt - sum_i w_i c_i||   (the conditional mean)
- `between_ratio` = `d_nearest_mode` / `mode_sep`

**Also recorded per iteration** (already produced by `run()`, no new code):
inference regret (`inference_regret_curve`), simple regret
(`hf_regret_curve`), the real queried location (`x_t_trace`), its value
(`y_t_trace`), and the fidelity (`fidelity_trace`). `use_gp_refinement=False`,
so `x_t_trace` IS the DT's raw proposal — no refinement stands between them.

## Prediction, and what would falsify it

**Mean-collapse is SUPPORTED only if all three hold, FAIL vs PASS:**
1. `n_modes` is higher on FAIL (the teacher is genuinely more multimodal), AND
2. `d_teacher_mean` is **lower** on FAIL (the DT sits at the conditional mean), AND
3. `between_ratio` is **higher** on FAIL (and > ~0.5, i.e. the DT is far from
   every individual mode relative to how far apart the modes are).

**Falsified if** the DT sits *on* a mode in both groups (`between_ratio` small
everywhere), or if FAIL and PASS are indistinguishable on all three, or if
`n_modes` ~ 1 throughout — in which case there are no modes to fall between and
the mechanism cannot be operating, whatever else is wrong.

A distinct possibility worth naming now: the DT may sit near the teacher mean in
**both** groups, with the groups differing only in how multimodal the teacher
is. That would make mean-collapse a permanent property of the regression head
rather than the cause of these two failures, and would predict the failure is
about the *landscape*, not the head.

## Statistics — stated in advance

**n = 2 vs 2 seeds. No significance test will be run or reported.** Iterations
within a seed are not independent, so per-iteration counts must not be treated
as sample size. I will report per-seed medians and the full per-iteration
distributions, and describe the comparison as descriptive. Any claim beyond
"these 4 runs look like X" would need more seeds.
