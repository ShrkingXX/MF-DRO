"""
Run loop for Experiment 1 (MES reward ablation) and Experiment 2 (RTG schema
comparison). Checkpoint-aware: safe to interrupt and re-run, already-completed
(benchmark, variant, seed) runs are skipped.

Usage:
    python run_experiment.py --experiment mes_reward
    python run_experiment.py --experiment rtg_schema
    python run_experiment.py --experiment mes_reward --seeds 42 43 44   # quick subset
"""
import argparse
import time

import numpy as np

from checkpoint import (setup_dirs, is_completed, load_all_results,
                         log_global, print_progress)
from dro_runner import run_single_seed

SEEDS = list(range(42, 47)) # 42-46, 5 seeds (reduced from 10 for runtime -- see conversation)

# --- Experiment 1: MES reward ablation ---------------------------------
EXP1_NAME = "mes_reward"
EXP1_BENCHMARKS = [
    "Ackley_2D", "Ackley_5D", "Rosenbrock_2D", "Hartmann_6D", "Currin_2D",
]
EXP1_VARIANTS = ["DRO_Improvement", "DRO_MES", "NaiveBO"]

_EXP1_SHARED = dict(
    gp_num_models=5, rollouts_per_iter=75, rollout_length=4,
    bo_iterations=50, initial_points=5,
    dt_hidden=128, dt_layers=4, dt_heads=4,
    gp_kernel="rbf", gp_ard=True,
)
EXP1_VARIANT_CONFIGS = {
    "DRO_Improvement": dict(use_mes_reward=False, rtg_schema="floored", alpha_floor=0.5, **_EXP1_SHARED),
    "DRO_MES": dict(use_mes_reward=True, rtg_schema="floored", alpha_floor=0.5, **_EXP1_SHARED),
    "NaiveBO": dict(is_naive_bo=True, bo_iterations=50, initial_points=5),
}

# --- Experiment 2: RTG schema comparison --------------------------------
EXP2_NAME = "rtg_schema"
EXP2_BENCHMARKS = ["Ackley_2D", "Ackley_5D", "Eggholder", "Hartmann_6D"]
EXP2_VARIANTS = [
    "DRO_Fixed", "DRO_Dynamic", "DRO_Floored",
    "DRO_Quantile_0.25", "DRO_Quantile_0.5", "DRO_Quantile_0.75", "DRO_Quantile_0.9",
]

_EXP2_SHARED = dict(
    use_mes_reward=False, # Reward held fixed (Improvement) to isolate the RTG effect
    gp_num_models=5, rollouts_per_iter=75, rollout_length=4,
    bo_iterations=50, initial_points=5,
    dt_hidden=128, dt_layers=4, dt_heads=4,
    gp_kernel="rbf", gp_ard=True,
    lambda_rtg=1.0, alpha_floor=0.5, rtg_warmup=3,
)
EXP2_VARIANT_CONFIGS = {
    "DRO_Fixed": dict(rtg_schema="fixed", **_EXP2_SHARED),
    "DRO_Dynamic": dict(rtg_schema="dynamic", **_EXP2_SHARED),
    "DRO_Floored": dict(rtg_schema="floored", **_EXP2_SHARED),
    "DRO_Quantile_0.25": dict(rtg_schema="quantile", alpha_inference=0.25, **_EXP2_SHARED),
    "DRO_Quantile_0.5": dict(rtg_schema="quantile", alpha_inference=0.5, **_EXP2_SHARED),
    "DRO_Quantile_0.75": dict(rtg_schema="quantile", alpha_inference=0.75, **_EXP2_SHARED),
    "DRO_Quantile_0.9": dict(rtg_schema="quantile", alpha_inference=0.9, **_EXP2_SHARED),
}

# --- Experiment 3: MES switching-condition ablation --------------------
# Prerequisite for the multi-fidelity DRO (MF-DRO) pipeline: tests whether
# using MES (mutual information about y*) as the ROLLOUT query-selection
# criterion -- instead of EI -- produces better DT training data, crossed
# independently with reward type (2x2), plus NaiveBO-EI/-MES references.
# EXP1_VARIANT_CONFIGS/EXP2_VARIANT_CONFIGS are untouched by this block.
EXP3_NAME = "mes_switching"
EXP3_BENCHMARKS = ["Ackley_2D", "Ackley_5D", "Rosenbrock_2D", "Hartmann_6D", "Currin_2D"]
EXP3_VARIANTS = [
    "DRO-EI-Impr", "DRO-EI-MES", "DRO-MES-Impr", "DRO-MES-MES",
    "NaiveBO-EI", "NaiveBO-MES",
]

_EXP3_SHARED = dict(
    gp_num_models=5, rollouts_per_iter=75, rollout_length=4,
    bo_iterations=50, initial_points=5,
    dt_hidden=128, dt_layers=4, dt_heads=4,
    gp_kernel="rbf", gp_ard=True,
    rtg_schema="floored", alpha_floor=0.5,
)
EXP3_VARIANT_CONFIGS = {
    "DRO-EI-Impr": dict(use_mes_reward=False, rollout_acq_function="ei", **_EXP3_SHARED),
    "DRO-EI-MES": dict(use_mes_reward=True, rollout_acq_function="ei", **_EXP3_SHARED),
    "DRO-MES-Impr": dict(use_mes_reward=False, rollout_acq_function="mes", **_EXP3_SHARED),
    "DRO-MES-MES": dict(use_mes_reward=True, rollout_acq_function="mes", **_EXP3_SHARED),
    "NaiveBO-EI": dict(is_naive_bo=True, naivebo_acq_function="ei", bo_iterations=50, initial_points=5),
    "NaiveBO-MES": dict(is_naive_bo=True, naivebo_acq_function="mes", bo_iterations=50, initial_points=5),
}

# --- Phase 4: joint RTG empirical validation ----------------------------
# Compares rtg_schema="floored" (per-step RTG, current default) against
# rtg_schema="joint" (log-ratio of Thompson-sampled Gumbel scales; see
# DirectRegretOptimization._simulate_trajectory_joint), on Currin_2D only.
# Scope: validation/phase2_rtg_correlation.py (v4, post roi_candidates
# bugfix) and validation/phase3_scale_check.py (v4) showed joint RTG is only
# consistently viable on Currin_2D across all BO stages; Hartmann_6D's
# early-stage signal is noise-dominated (SNR=0.01, rho_all=0.06 at n_initial=5)
# -- confirmed to persist into a real 30-rollout check
# (validation/phase4_step2_neg_frac_check.py: 40% neg_frac early-stage vs
# 3.3% mid-stage) -- and was excluded from this experiment by explicit user
# decision after reviewing that evidence.
EXP4_NAME = "joint_rtg_validation"
EXP4_BENCHMARKS = ["Currin_2D"]
# DRO-MES-Joint (original, batch_max_rtg = max() of 75 rollouts) went flat
# for 28-30/30 iterations on 2/3 seeds -- diagnosed as the floored-target
# ratchet locking onto early Thompson/Gumbel-MLE noise spikes and never
# recovering (validation/phase4_step3_diagnosis.py). DRO-MES-Joint-P90 tests
# the fix (batch_max_rtg = 90th percentile instead of raw max, for
# rtg_schema=="joint" only -- see _compute_rtg_target in src/policy/dro.py).
# Kept as a separate variant name (not overwriting DRO-MES-Joint) so both
# results remain in the checkpoint for before/after comparison.
EXP4_VARIANTS = ["DRO-MES-PerStep", "DRO-MES-Joint", "DRO-MES-Joint-P90"]
EXP4_SEEDS = [42, 43, 44]

_EXP4_SHARED = dict(
    use_mes_reward=True,
    gp_num_models=5, rollouts_per_iter=75, rollout_length=4,
    bo_iterations=30, initial_points=5,
    dt_hidden=128, dt_layers=4, dt_heads=4,
    gp_kernel="rbf", gp_ard=True, alpha_floor=0.5,
)
EXP4_VARIANT_CONFIGS = {
    "DRO-MES-PerStep": dict(rtg_schema="floored", **_EXP4_SHARED),
    "DRO-MES-Joint": dict(rtg_schema="joint", **_EXP4_SHARED),
    "DRO-MES-Joint-P90": dict(rtg_schema="joint", **_EXP4_SHARED),
}

# --- Phase 5: entropy_joint RTG validation -------------------------------
# Compares rtg_schema="floored" (per-step RTG, DRO-MES-PerStep -- same as
# EXP4's, but now at rollout_length=8 to match the entropy_joint arm exactly)
# against rtg_schema="entropy_joint" (pure differential entropy RTG[tau] =
# H(y*|D_tau) = log(b_tau)+euler_gamma+1, no b_T subtraction -- see
# DirectRegretOptimization._simulate_trajectory_joint's entropy_joint branch).
# Uses _make_fantasy_model (not raw condition_on_observations) for fantasy
# conditioning -- verified bit-identical to a from-scratch ground-truth GP,
# including chained multi-step conditioning (see that method's docstring).
# entropy_joint's RTG target uses _compute_rtg_target's "dynamic"-style
# dispatch (plain batch_max_rtg, no floor, no percentile trimming), by
# explicit user instruction -- NOT "joint"'s percentile-floored treatment.
# Benchmarks: Hartmann_6D (sanity-checked at n_initial=5/25, rollout_length=8:
# RTG is negative in 100% of steps at both stages -- confirmed correct,
# differential entropy of a sufficiently narrow Gumbel(a,b) can be negative;
# Hartmann_6D's b_tau stays below exp(-euler_gamma-1)~0.207 throughout) and
# Currin_2D (untested here -- Phase 1 b_tau of 0.17-0.49 straddles that same
# threshold, a genuinely different regime worth checking empirically).
EXP5_NAME = "entropy_joint_validation"
EXP5_BENCHMARKS = ["Hartmann_6D", "Currin_2D"]
EXP5_VARIANTS = ["DRO-MES-PerStep", "DRO-MES-EntropyJoint"]

_EXP5_SHARED = dict(
    use_mes_reward=True,
    gp_num_models=5, rollouts_per_iter=75, rollout_length=8,
    bo_iterations=50, initial_points=5,
    dt_hidden=128, dt_layers=4, dt_heads=4,
    gp_kernel="rbf", gp_ard=True, alpha_floor=0.5,
)
EXP5_VARIANT_CONFIGS = {
    "DRO-MES-PerStep": dict(rtg_schema="floored", **_EXP5_SHARED),
    "DRO-MES-EntropyJoint": dict(rtg_schema="entropy_joint", **_EXP5_SHARED),
}

# --- Stage 1: ROI-state augmentation audit -------------------------------
# Minimal test (20 iters, 3 seeds) of whether adding GP posterior mean/std
# over roi_candidates to the state vector (use_roi_state=True, see
# DirectRegretOptimization._extract_state) changes the DT's proposals in the
# expected direction -- away from the domain corner, tracking the GP's own
# (correct) signal about where the true optimum is -- before committing to
# the full Stage 2 run. Fixed rollout_length=4 (original DRO default, NOT
# EXP5's rollout_length=8) and rtg_schema="floored" (NOT entropy_joint) --
# isolating the state-augmentation variable alone, unconfounded by the other
# changes explored in EXP5.
# DRO-Slim8Dim / DRO-SlimQuantiles added later, same exp_name/benchmark/seeds/
# shared config, so they're directly comparable to DRO-Original without
# re-running it: DRO-Slim8Dim ablates slots 0-9 (raw GP hyperparameters) out
# of the state entirely, as a control -- if this doesn't hurt (or helps),
# those slots were dead weight; if it hurts, they carry real signal.
# DRO-SlimQuantiles combines that same 8-dim slim base with
# use_roi_std_quantiles=True: [min,p25,median,p75,max] of sigma_roi (posterior
# std over roi_candidates) per GP ensemble member, a distributional view of
# uncertainty instead of DRO-ROIState's single mean/std summary. Both reuse
# the same _extract_state posterior call, no extra GP evaluations.
#
# DRO-Slim8Dim's result (final regret 2.77+-0.17 vs DRO-Original's 2.29+-0.08,
# ~21% worse, non-overlapping SE bands) showed the hyperparameter slots carry
# real signal rather than being dead weight -- invalidating the ablation
# hypothesis DRO-SlimQuantiles was built to test as a follow-up, so
# DRO-SlimQuantiles was never run (left defined/sanity-checked for reference
# only, not in EXP6_VARIANTS).
#
# DRO-ROISigmaIQR added after that pivot: keeps ALL 18 original slots (no
# ablation) and appends [mean(sigma_roi), IQR(sigma_roi)] per GP ensemble
# member (2*num_models=10 extra slots, 28-dim total) via use_roi_sigma_iqr=True
# -- pairs mean_sigma_roi with the interquartile range of the per-candidate
# posterior std instead of DRO-ROIState's mean_mu_roi, distinguishing uniform
# uncertainty from saturated-region uncertainty.
#
# DRO-ROIState's and DRO-ROISigmaIQR's first Hartmann_6D runs used pre-fix
# normalizations (mu_denom re-derived from current best_value each call;
# IQR normalized by the frozen initial sigma instead of the current one) --
# both fixed in _extract_state, and those checkpoints were discarded (not
# valid under the current code).
#
# Ackley_5D added because Hartmann_6D's Stage 1 audits showed 11/12 runs
# (every variant, nearly every seed) never find a single improving point in
# 20 iterations -- final regret there is dominated by which seed's initial
# LHS sample got lucky, not by state design, making it uninformative for this
# comparison. Ackley_5D reliably shows DT improvement over LHS in Experiment 1,
# so it's used for DRO-Original vs DRO-ROISigmaIQR instead. Hartmann_6D is kept
# in EXP6_BENCHMARKS since DRO-Original/DRO-Slim8Dim's completed Hartmann_6D
# checkpoints remain valid, comparable historical data for that question.
#
# NOTE on seeds for the Ackley_5D DRO-Original/DRO-ROISigmaIQR audit
# specifically: seed=42 is degenerate there -- _sample_initial_points_lhs
# (a "basic" LHS with no per-stratum jitter, just a fixed permuted grid of 5
# stratum-midpoint values per dimension) happens, for seed=42, to place the
# domain-midpoint value (which is exactly 0 for Ackley_5D's symmetric [-5,5]^5
# domain) in the same row across all 5 dimensions -- landing one "random"
# initial point exactly on Ackley's known global optimum [0,0,0,0,0], making
# regret ~1e-16 for the entire run regardless of variant. Seed=45 was used in
# its place for this specific audit only (confirmed no such collision, and
# DRO-Original shows genuine improving regret 5.56->0.76 over 10 iterations).
# The sampler itself was NOT changed -- Hartmann_6D's seeds (42/43/44,
# EXP6_SEEDS default) are unaffected by this issue (domain [0,1]^6 has no
# optimum at the domain midpoint) and its already-completed checkpoints
# remain valid.
EXP6_NAME = "roi_state_audit"
EXP6_BENCHMARKS = ["Hartmann_6D", "Ackley_5D"]
EXP6_VARIANTS = ["DRO-Original", "DRO-ROIState", "DRO-Slim8Dim", "DRO-ROISigmaIQR"]
EXP6_SEEDS = [42, 43, 44]

_EXP6_SHARED = dict(
    use_mes_reward=True, rtg_schema="floored", alpha_floor=0.5,
    rollout_acq_function="ei",
    gp_num_models=5, rollouts_per_iter=75, rollout_length=4,
    bo_iterations=20, initial_points=5,
    dt_hidden=128, dt_layers=4, dt_heads=4,
    gp_kernel="rbf", gp_ard=True,
)
EXP6_VARIANT_CONFIGS = {
    "DRO-Original": dict(use_roi_state=False, **_EXP6_SHARED),
    "DRO-ROIState": dict(use_roi_state=True, **_EXP6_SHARED),
    "DRO-Slim8Dim": dict(state_hyperparams_enabled=False, **_EXP6_SHARED),
    "DRO-SlimQuantiles": dict(state_hyperparams_enabled=False, use_roi_std_quantiles=True, **_EXP6_SHARED),
    "DRO-ROISigmaIQR": dict(use_roi_sigma_iqr=True, **_EXP6_SHARED),
}

# --- Stage 1 follow-up: quick Hartmann_6D flat-regret diagnostics ----------
# Four small (3 seed x 20 iter) probes into WHY every Stage 1 variant
# (DRO-Original/ROIState/Slim8Dim/ROISigmaIQR) plateaued flat on Hartmann_6D
# with no state-design fix so far. Each isolates one candidate mechanism:
#   DRO-Baseline:  exact current Stage 1 config, unmodified -- confirms the
#     flat-regret result still reproduces (sanity check for the other three).
#   DRO-Noise005:  adds N(0, 0.05^2) observation noise to Hartmann_6D
#     evaluations only (observation_noise_std, dro_runner.run_single_seed) --
#     tests whether GP posterior std collapsing to ~0 at a converged/corner
#     region (noiseless observations -> the GP can become arbitrarily
#     confident there) is creating a positive feedback loop the DT can't
#     escape.
#   DRO-RotateAcq: rollout_acq_function="rotate" cycles
#     ["ei","ucb","pi","mes"] by rollout step (_simulate_trajectory, see
#     acq_name_override on _optimize_acquisition) instead of using one fixed
#     acquisition function for every rollout step -- tests whether diverse
#     rollout queries (UCB explores even saturated regions, MES seeks
#     informative locations) fix training data diversity.
#   DRO-MES-EI: use_mes_reward=True + rollout_acq_function="ei" -- identical
#     to DRO-Baseline's actual config (Stage 1's shared settings already set
#     use_mes_reward=True); kept as specified for direct comparability, but
#     note this will not show a different result from DRO-Baseline by
#     construction, only a second 3-seed sample of the same configuration.
EXP7_NAME = "hartmann_diag"
EXP7_BENCHMARKS = ["Hartmann_6D"]
EXP7_VARIANTS = ["DRO-Baseline", "DRO-Noise005", "DRO-RotateAcq", "DRO-MES-EI"]
EXP7_SEEDS = [42, 43, 44]

_EXP7_SHARED = dict(
    use_mes_reward=True, rtg_schema="floored", alpha_floor=0.5,
    gp_num_models=5, rollouts_per_iter=75, rollout_length=4,
    bo_iterations=20, initial_points=5,
    dt_hidden=128, dt_layers=4, dt_heads=4,
    gp_kernel="rbf", gp_ard=True,
)
EXP7_VARIANT_CONFIGS = {
    "DRO-Baseline": dict(rollout_acq_function="ei", **_EXP7_SHARED),
    "DRO-Noise005": dict(rollout_acq_function="ei", observation_noise_std=0.05, **_EXP7_SHARED),
    "DRO-RotateAcq": dict(rollout_acq_function="rotate", **_EXP7_SHARED),
    "DRO-MES-EI": dict(rollout_acq_function="ei", **_EXP7_SHARED),
}

# --- entropy_joint RTG on Ackley_5D ----------------------------------------
# Tests whether rtg_schema="entropy_joint" (H(y*|D_tau), see EXP5's comment
# block and DirectRegretOptimization._simulate_trajectory_joint) works on a
# benchmark where DRO actually learns (Ackley_5D, unlike Hartmann_6D's flat
# plateau -- see hartmann_diag above). DRO-PerStep is the known-working
# rtg_schema="floored" baseline. DRO-EntropyJoint matches EXP5's Hartmann_6D
# entropy_joint config (rollout_length=8); DRO-EntropyJoint-L4 isolates
# whether that rollout_length matters on this benchmark specifically.
# rtg_schema="entropy_joint" always uses _compute_rtg_target's "dynamic"-style
# dispatch (plain batch_max_rtg, no floor) -- not a separate config key, see
# EXP5's comment. No changes to the entropy_joint mechanism itself; the only
# code addition anywhere in this task is a neg_rtg_frac diagnostic (fraction
# of a batch's per-rollout RTG[0] values that are negative), computed from
# batch_rtg0_list -- already assembled for _compute_rtg_target -- and logged
# alongside the existing rtg_target/batch_max_rtg fields.
# DRO-MES-jointMES added later: a targeted check of whether entropy_joint
# paired with MES acquisition (rollout_acq_function="mes") -- the actual
# candidate proposed for the EXP9 cluster matrix -- behaves differently from
# DRO-EntropyJoint above, which used the default EI acquisition (_EXP8_SHARED
# never sets rollout_acq_function). DRO-EntropyJoint (EI, L8) failed EXP9's
# planned rollout_length=8 by >1 baseline SE; this checks the untested
# MES-paired combination specifically before committing ~30 expensive
# 200-iteration cluster runs to EXP9's DRO-rotate-jointMES/DRO-MES-jointMES.
EXP8_NAME = "entropy_joint_ackley5d"
EXP8_BENCHMARKS = ["Ackley_5D"]
EXP8_VARIANTS = ["DRO-PerStep", "DRO-EntropyJoint", "DRO-EntropyJoint-L4", "DRO-MES-jointMES"]
EXP8_SEEDS = [42, 43, 44]

_EXP8_SHARED = dict(
    use_mes_reward=True, alpha_floor=0.5,
    gp_num_models=5, rollouts_per_iter=75,
    bo_iterations=30, initial_points=5,
    dt_hidden=128, dt_layers=4, dt_heads=4,
    gp_kernel="rbf", gp_ard=True,
)
EXP8_VARIANT_CONFIGS = {
    "DRO-PerStep": dict(rtg_schema="floored", rollout_length=4, **_EXP8_SHARED),
    "DRO-EntropyJoint": dict(rtg_schema="entropy_joint", rollout_length=8, **_EXP8_SHARED),
    "DRO-EntropyJoint-L4": dict(rtg_schema="entropy_joint", rollout_length=4, **_EXP8_SHARED),
    "DRO-MES-jointMES": dict(rtg_schema="entropy_joint", rollout_length=8,
                              rollout_acq_function="mes", **_EXP8_SHARED),
}

# --- mes_switching_v2: large-scale cluster experiment ----------------------
# 200-iteration, rollout_length=8 (uniform across variants -- MES may benefit
# from longer rollouts, per EXP8's finding), 5-seed x 3-benchmark comparison
# of acquisition-function source (rotate vs MES) x reward type (improvement
# vs MES) x DT-vs-no-DT (NaiveBO baselines).
#
# MES-as-ACQUISITION (rollout_acq_function="mes", the qMaxValueEntropy branch
# in _acquisition_function_value_botorch) intentionally still uses Gumbel
# sampling, NOT Thompson -- this is a different role from MES-as-REWARD
# (use_mes_reward=True, compute_mes_reward) and RTG's b_tau estimation
# (_simulate_trajectory_joint), both of which switched to Thompson sampling
# earlier this session because Gumbel's independence assumption biases a
# TRAINING LABEL that compounds across thousands of policy updates. A single
# greedy argmax over candidates doesn't have that compounding problem, and
# the original MES paper's own validation is that Gumbel-sampling bias
# doesn't meaningfully hurt acquisition-function performance -- so Gumbel is
# kept here deliberately, both for correctness (it's the already-tested,
# unmodified "mes" branch) and cost (Thompson would need a joint multivariate
# y* draw over ~1000 roi_candidates, an O(N^3) Cholesky Gumbel avoids
# entirely via per-candidate marginal CDFs).
#
# DRO-rotate-jointMES / DRO-MES-jointMES (rtg_schema="entropy_joint"):
# EXP8's Ackley_5D validation at rollout_length=8 with the DEFAULT (EI)
# rollout acquisition found entropy_joint underperforms per-step MES reward
# by >1 SE (only rollout_length=4 passed, which doesn't match this
# experiment's uniform rollout_length=8). A targeted follow-up specifically
# pairing entropy_joint with MES acquisition (EXP8's "DRO-MES-jointMES",
# 3 seeds/30 iters/Ackley_5D) landed WITHIN 1 SE of baseline -- better than
# the EI-paired version, though still numerically worse than per-step. The
# rotate-paired combination has no direct evidence either way. Both are
# included below by explicit choice, accepting that DRO-rotate-jointMES in
# particular is untested -- if either performs badly at cluster scale, the
# per-phase wall-clock / neg_rtg_frac / rollout_action_diversity diagnostics
# already logged for every variant should help explain why.
EXP9_NAME = "mes_switching_v2_cluster"
EXP9_BENCHMARKS = ["Ackley_2D", "Ackley_5D", "Ackley_10D"]
EXP9_VARIANTS = [
    "DRO-rotate-Impr", "DRO-rotate-MES", "DRO-MES-Impr", "DRO-MES-MES",
    "NaiveBO-EI", "NaiveBO-MES",
    "DRO-rotate-jointMES", "DRO-MES-jointMES",
]
EXP9_SEEDS = [42, 43, 44, 45, 46]

_EXP9_DRO_SHARED = dict(
    rtg_schema="floored", alpha_floor=0.5,
    gp_num_models=5, rollouts_per_iter=75, rollout_length=8,
    bo_iterations=200, initial_points=5,
    dt_hidden=128, dt_layers=4, dt_heads=4,
    gp_kernel="rbf", gp_ard=True,
)
_EXP9_NAIVEBO_SHARED = dict(
    is_naive_bo=True, bo_iterations=200, initial_points=5,
)
EXP9_VARIANT_CONFIGS = {
    "DRO-rotate-Impr": dict(use_mes_reward=False, rollout_acq_function="rotate", **_EXP9_DRO_SHARED),
    "DRO-rotate-MES":  dict(use_mes_reward=True,  rollout_acq_function="rotate", **_EXP9_DRO_SHARED),
    "DRO-MES-Impr":    dict(use_mes_reward=False, rollout_acq_function="mes",    **_EXP9_DRO_SHARED),
    "DRO-MES-MES":     dict(use_mes_reward=True,  rollout_acq_function="mes",    **_EXP9_DRO_SHARED),
    "NaiveBO-EI":  dict(naivebo_acq_function="ei",  **_EXP9_NAIVEBO_SHARED),
    "NaiveBO-MES": dict(naivebo_acq_function="mes", **_EXP9_NAIVEBO_SHARED),
    # Defined for completeness / easy opt-in -- not in EXP9_VARIANTS, see
    # comment block above.
    "DRO-rotate-jointMES": dict(use_mes_reward=True, rtg_schema="entropy_joint",
                                 rollout_acq_function="rotate",
                                 **{k: v for k, v in _EXP9_DRO_SHARED.items() if k != "rtg_schema"}),
    "DRO-MES-jointMES": dict(use_mes_reward=True, rtg_schema="entropy_joint",
                              rollout_acq_function="mes",
                              **{k: v for k, v in _EXP9_DRO_SHARED.items() if k != "rtg_schema"}),
}

EXPERIMENTS = {
    EXP1_NAME: dict(benchmarks=EXP1_BENCHMARKS, variants=EXP1_VARIANTS, variant_configs=EXP1_VARIANT_CONFIGS),
    EXP2_NAME: dict(benchmarks=EXP2_BENCHMARKS, variants=EXP2_VARIANTS, variant_configs=EXP2_VARIANT_CONFIGS),
    EXP3_NAME: dict(benchmarks=EXP3_BENCHMARKS, variants=EXP3_VARIANTS, variant_configs=EXP3_VARIANT_CONFIGS),
    EXP4_NAME: dict(benchmarks=EXP4_BENCHMARKS, variants=EXP4_VARIANTS, variant_configs=EXP4_VARIANT_CONFIGS),
    EXP5_NAME: dict(benchmarks=EXP5_BENCHMARKS, variants=EXP5_VARIANTS, variant_configs=EXP5_VARIANT_CONFIGS),
    EXP6_NAME: dict(benchmarks=EXP6_BENCHMARKS, variants=EXP6_VARIANTS, variant_configs=EXP6_VARIANT_CONFIGS),
    EXP7_NAME: dict(benchmarks=EXP7_BENCHMARKS, variants=EXP7_VARIANTS, variant_configs=EXP7_VARIANT_CONFIGS),
    EXP8_NAME: dict(benchmarks=EXP8_BENCHMARKS, variants=EXP8_VARIANTS, variant_configs=EXP8_VARIANT_CONFIGS),
    EXP9_NAME: dict(benchmarks=EXP9_BENCHMARKS, variants=EXP9_VARIANTS, variant_configs=EXP9_VARIANT_CONFIGS),
}


def run_experiment(exp_name, benchmarks, variants, seeds, variant_configs):
    """
    variant_configs: dict mapping variant_name -> kwargs for run_single_seed
    """
    setup_dirs(exp_name)
    total = len(benchmarks) * len(variants) * len(seeds)
    completed_count = skipped_count = failed_count = 0
    global_start = time.perf_counter()

    log_global(exp_name,
        f"EXPERIMENT STARTED  variants={variants}  "
        f"benchmarks={benchmarks}  seeds={seeds}  total={total}")

    for benchmark in benchmarks:
        for variant_name in variants:
            for seed in seeds:

                if is_completed(exp_name, benchmark, variant_name, seed):
                    log_global(exp_name, f"SKIPPED {benchmark} {variant_name} seed{seed}")
                    skipped_count += 1
                    print_progress(completed_count, skipped_count, failed_count, total, global_start)
                    continue

                log_global(exp_name, f"STARTED {benchmark} {variant_name} seed{seed}")

                try:
                    run_single_seed(
                        exp_name=exp_name,
                        benchmark_name=benchmark,
                        variant_name=variant_name,
                        seed=seed,
                        **variant_configs[variant_name]
                    )
                    # save_result is called inside run_single_seed
                    completed_count += 1
                    log_global(exp_name, f"COMPLETED {benchmark} {variant_name} seed{seed}")

                except Exception as e:
                    failed_count += 1
                    log_global(exp_name, f"FAILED {benchmark} {variant_name} seed{seed} error={str(e)}")
                    print(f"  FAILED: {benchmark} {variant_name} seed{seed}: {e}")

                print_progress(completed_count, skipped_count, failed_count, total, global_start)

        # After all variants/seeds for this benchmark finish, print a partial
        # summary immediately so progress can be monitored on long runs.
        print(f"\n=== PARTIAL RESULTS: {benchmark} ===")
        all_res = load_all_results(exp_name, [benchmark], variants, seeds)
        for v in variants:
            done = [all_res[(benchmark, v, s)]["regret_curve"][-1]
                    for s in seeds
                    if (benchmark, v, s) in all_res and all_res[(benchmark, v, s)]["regret_curve"]]
            if done:
                m, se = np.mean(done), np.std(done) / np.sqrt(len(done))
                print(f"  {v}: {m:.4f} +/- {se:.4f}  (N={len(done)})")
        print("=" * 40 + "\n")

    log_global(exp_name,
        f"EXPERIMENT FINISHED  completed={completed_count}  "
        f"skipped={skipped_count}  failed={failed_count}")

    return load_all_results(exp_name, benchmarks, variants, seeds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the MES-reward or RTG-schema DRO experiment.")
    parser.add_argument("--experiment", required=True, choices=list(EXPERIMENTS.keys()))
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                         help="Override the default 10 seeds (42-51), e.g. --seeds 42 43 44 for a quick subset.")
    parser.add_argument("--benchmarks", type=str, nargs="+", default=None,
                         help="Override the default benchmark list for this experiment.")
    parser.add_argument("--variants", type=str, nargs="+", default=None,
                         help="Override the default variant list for this experiment.")
    args = parser.parse_args()

    spec = EXPERIMENTS[args.experiment]
    run_experiment(
        exp_name=args.experiment,
        benchmarks=args.benchmarks or spec["benchmarks"],
        variants=args.variants or spec["variants"],
        seeds=args.seeds or SEEDS,
        variant_configs=spec["variant_configs"],
    )
