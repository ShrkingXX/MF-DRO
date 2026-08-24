# Protocol — H1 leak-fix validation under the frozen evaluation

**Locked before running.** Results commit must be separate.

## Why this experiment

A target-leakage bug (commit 7bcc3b8) was found in the Decision Transformer
readout AFTER every prior number in `data/results_inventory.csv` and
`findings.md` was generated. Both the MF path (`forward_mf`) and the SF path
(`get_action_hidden_states`) read out from the ACTION token, which under
`triu(diagonal=1)` self-attention embeds the true action — so every head was
reading its own target, and `propose_mf` zeroes that slot at inference.

Every DRO-family number carried into this project therefore predates the fix
and cannot be compared against a post-fix method. This experiment produces the
first post-fix numbers under the frozen protocol.

## Hypothesis

**H1-fix**: with the leakage fixed (plus the τ=0 state collapse, BTG cost-floor,
RTG-cap and reward fixes in 7bcc3b8), MF-DRO's final simple regret on Hartmann
6D improves materially versus its pre-fix value of ~1.31.

**Prediction (confirmatory, locked):** MF-DRO post-fix mean final simple regret
< 1.0. Separately, the frozen success test — MF-DRO mean+SE strictly below
best-baseline mean−SE — is evaluated and reported whether or not it passes.

Encouraging but NOT the prediction (3 seeds, 40 iters, not the protocol):
post-fix mean regret 0.761 in the fidelity measurement.

## Frozen evaluation (from PROTOCOL.md — unchanged)

| Item | Value |
|---|---|
| Benchmark | Hartmann 6D |
| Methods | MF-DRO, MF-MI-Greedy, MF-GP-UCB |
| Seeds | 42–51 (10, identical across methods) |
| Budget | matched real cost, `cost_budget=200` post-init, identical for all |
| Metric | final simple regret at matched cost |
| Success test | MF-DRO mean+SE < best-baseline mean−SE |
| Initial design | identical: `initial_hf=36`, `initial_lf=60`, same LHS per seed |

`cost_budget=200` is chosen for compute feasibility BEFORE seeing any result.
`bo_iterations` is capped at 250 purely as a runaway guard; the analysis MUST
report realized final cost per method and flag any run where the cap bound
instead of the budget (that run is not cost-matched).

## Method configuration (MF-DRO)

Post-fix defaults as committed in 7bcc3b8: `rollout_reward="improvement"`,
`use_candidate_scoring=True`, `use_candidate_features=True`,
`use_state_standardization=True`, `use_linear_score_head=True`,
`fidelity_sampling=True`, `rollouts_per_model=20`, `rollout_length=8`,
`num_epochs=10`, `dkl_threshold=9999`, `bes_delta=0.0`.
`use_teacher_pool=False` (default) — deliberately held OFF so this measures the
leak fix alone, not the teacher pool.

## What each outcome means

- MF-DRO mean < 1.0 → the leak was a real driver of the freeze; H1 supported
  with a concrete mechanism.
- MF-DRO mean ~1.3, unchanged → the leak was real but NOT the freeze driver;
  H1's mechanism is wrong and the DT-internal search continues.
- Success test passes → a within-frame fix closes the gap. Report seed traces.
- Success test fails but regret improves → partial; report the residual gap
  honestly, per PROTOCOL.md ("no within-frame fix closes the gap" is valid).

## Compute

30 jobs (3 methods × 10 seeds). `num_workers=15 × threads_per_worker=1 = 15 ≤ 15`.
Env vars set before the torch import inside each spawned worker.
