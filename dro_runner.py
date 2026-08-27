"""
Checkpoint-aware single-run wrapper (Step 3, Change 5), completed here in
Step 6 since it needs the benchmark registry (benchmarks.py) to resolve a
benchmark_name string into an actual objective function + domain.
"""
import os
import json
import time
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from checkpoint import (is_completed, load_result, log_global, save_result,
                         has_resume_checkpoint, delete_resume_checkpoint,
                         RESULTS_ROOT)
from benchmarks import get_benchmark
from naive_bo import run_naive_bo
from src.policy.dro import DirectRegretOptimization
from src.policy.mf_dro import DirectMFRegretOptimization

RESULT_KEYS = [
    "regret_curve", "best_observed", "mean_reward", "zero_frac",
    "rtg_target", "batch_max_rtg", "running_max_rtg", "iter_times",
]

MF_RESULT_KEYS = [
    "hf_regret_curve",
    "cost_curve",
    "fidelity_trace",
    "x_t_trace",
    "y_t_trace",
    "initial_hf_values",
    "cumulative_cost_curve",
    "lf_fraction",
    "rtg_target",
    "btg_target",
    "fid_mean_per_iter",
    "fid_std_per_iter",
    "L_loc_per_iter",
    "L_fid_per_iter",
    "neg_rtg_frac_per_iter",
    "action_reward_corr_per_iter",
    "rtg_frac_between_traj_var_per_iter",
    "rtg_gpbelief_corr_per_iter",
    "grad_coherency_per_iter",
    "query_dist_to_xstar_per_iter",
    "query_dist_to_x2_per_iter",
    "p_pred_inference_per_iter",
    "diag_frac_rollout_near_xstar_per_iter",
    "gp_refinement_log",
]

# Separate '.mf.json'/'.mf.done' path scheme (not checkpoint.py's own
# _json_path/_flag_path) -- MF-DRO runs are keyed by the SAME
# (exp_name, benchmark_base_name, variant_name, seed) tuple that an SF-DRO
# run could plausibly use (e.g. benchmark_base_name="Currin_2D" is also a
# valid SF benchmark_name), so reusing checkpoint.py's unmarked path
# functions would risk an MF result and an SF result silently colliding on
# the exact same file, each overwriting/misreading the other's
# differently-shaped JSON.


def _mf_json_path(exp_name, benchmark, variant, seed):
    return os.path.join(RESULTS_ROOT, exp_name, "checkpoints",
                         f"{benchmark}__{variant}__seed{seed}.mf.json")


def _mf_flag_path(exp_name, benchmark, variant, seed):
    return os.path.join(RESULTS_ROOT, exp_name, "checkpoints",
                         f"{benchmark}__{variant}__seed{seed}.mf.done")


def is_mf_completed(exp_name, benchmark_name, variant_name, seed):
    """Same pattern as is_completed, MF-marked path."""
    return os.path.exists(_mf_flag_path(exp_name, benchmark_name, variant_name, seed))


def save_mf_result(exp_name, benchmark_name, variant_name, seed, result):
    """Same pattern as save_result but uses MF_RESULT_KEYS' shape and an
    MF-marked path (JSON before flag, same crash-safety ordering)."""
    os.makedirs(os.path.join(RESULTS_ROOT, exp_name, "checkpoints"), exist_ok=True)
    json_path = _mf_json_path(exp_name, benchmark_name, variant_name, seed)
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    open(_mf_flag_path(exp_name, benchmark_name, variant_name, seed), 'w').close()


def load_mf_result(exp_name, benchmark_name, variant_name, seed):
    """Same pattern as load_result, MF-marked path."""
    with open(_mf_json_path(exp_name, benchmark_name, variant_name, seed)) as f:
        return json.load(f)

# HISTORY: dro.yaml's original gp.lengthscale_min/max (0.1, 10.0) were only ever
# exercised on Ackley/Rosenbrock/Levy-scale domains (width ~10-65), as an
# *absolute* range not scaled to domain size. Reusing them unchanged for e.g.
# Eggholder (width 1024) broke the GP (confirmed: zero improving points found
# in an entire 50-iteration run). The initial fix scaled these bounds
# proportionally to each benchmark's raw domain width.
#
# That fix is now REVERTED: src/policy/dro.py's GP ensemble was separately
# fixed to normalize inputs to [0,1]^d (via BoTorch's Normalize input_transform)
# before the kernel ever sees them, and to standardize outputs -- this was the
# deeper root cause (raw, unstandardized Y values of very different scale per
# benchmark meant 50 Adam steps never converged past a degenerate, flat
# posterior; confirmed by direct inspection). Once inputs are normalized
# internally, EVERY benchmark's kernel operates on a common [0,1]^d
# representation, so the ORIGINAL fixed (0.1, 10.0) bounds are already
# universally appropriate -- scaling them further by each benchmark's *raw*
# domain width double-compensates. Confirmed harmful directly: for Hartmann_6D
# (raw width 1), domain-width scaling produced a starting lengthscale of
# 0.0015 *in the normalized [0,1] space itself*, so tiny that gradients
# vanished and Adam never moved from that initial value in 50 steps (verified:
# lengthscale after "training" was bit-identical to its initial value) --
# producing the same degenerate flat-everywhere posterior this was meant to fix,
# and explaining why Hartmann_6D/Michalewicz showed zero improvement on every
# single seed even after the normalization fix.
_REFERENCE_LENGTHSCALE_MIN = 0.1
_REFERENCE_LENGTHSCALE_MAX = 10.0


def _scaled_lengthscale_bounds(benchmark_spec):
    # Fixed, not actually scaled by benchmark anymore -- see history above.
    # Name/signature kept as-is to avoid touching the (single) call site.
    return _REFERENCE_LENGTHSCALE_MIN, _REFERENCE_LENGTHSCALE_MAX


# Location (not value) of each benchmark's known global optimum --
# diagnostic-only (Track 1's mu_at_true_opt), so this is deliberately not part
# of the general BENCHMARKS registry. Left absent (None) for every other
# benchmark. Ackley's global optimum is at the origin for any dimension/bounds.
_KNOWN_OPTIMAL_X = {
    "Hartmann_6D": [0.2017, 0.1500, 0.4769, 0.2753, 0.3116, 0.6573],
    "Ackley_2D": [0.0, 0.0],
    "Ackley_5D": [0.0, 0.0, 0.0, 0.0, 0.0],
    "Ackley_10D": [0.0] * 10,
}


def _build_dro_config(exp_name, benchmark_name, variant_name, seed,
                       use_mes_reward, rtg_schema, alpha_floor, alpha_inference,
                       lambda_rtg, rtg_warmup, benchmark_spec,
                       gp_num_models, rollouts_per_iter, rollout_length,
                       bo_iterations, initial_points,
                       dt_hidden, dt_layers, dt_heads, dt_lr,
                       gp_kernel, gp_ard, verbose, mes_k=10,
                       rollout_acq_function="ei", use_roi_state=False,
                       use_roi_std_quantiles=False, state_hyperparams_enabled=True,
                       use_roi_sigma_iqr=False, use_awr=False, awr_temperature=None,
                       rollout_teacher="argmax", softmax_temperature=0.5,
                       use_gp_refinement=False, gp_refinement_steps=30,
                       gp_refinement_lr=0.05, gp_refinement_beta=2.0,
                       gp_refinement_variant="single",
                       num_epochs=100, propose_mode="dt", rollout_ga_steps=10,
                       uniform_ensemble=False):
    lengthscale_min, lengthscale_max = _scaled_lengthscale_bounds(benchmark_spec)
    cfg = OmegaConf.create({
        "seed": seed,
        "save_dir": os.path.join("results", exp_name, "dro_artifacts"),
        "verbose": verbose,
        "device": "cpu",
        "name": "dro",
        "exp_name": exp_name,
        "variant_name": variant_name,
        "benchmark_name": benchmark_name,
        "use_mes_reward": use_mes_reward,
        "mes_k": mes_k,
        # "argmax" default (bit-for-bit existing _optimize_acquisition
        # behavior). "softmax" routes _simulate_trajectory's rollout action
        # selection through _optimize_acquisition_softmax instead -- see
        # that method's docstring in src/policy/dro.py.
        "rollout_teacher": rollout_teacher,
        "softmax_temperature": softmax_temperature,
        # UCB refinement (warm-started gradient ascent from the DT's own
        # proposal, single-fidelity GP): isolates whether location quality
        # is the bottleneck when fidelity is not a factor. False default,
        # bit-for-bit unchanged pipeline when off -- see
        # DirectRegretOptimization._refine_proposal_ucb.
        "use_gp_refinement": use_gp_refinement,
        "gp_refinement_steps": gp_refinement_steps,
        "gp_refinement_lr": gp_refinement_lr,
        "gp_refinement_beta": gp_refinement_beta,
        # "single" (Variant A, default) | "ensemble" (B) | "twostage" (C) |
        # "restarts" (D) -- see the four _refine_proposal_ucb* methods.
        "gp_refinement_variant": gp_refinement_variant,
        # "dt" (default, unchanged pipeline) | "multistart_ucb_nodt"
        # (Diagnostic 1) | "naivebo_lognormal" (Diagnostic 2) -- bypasses
        # rollout simulation + DT training entirely when not "dt". See
        # DirectRegretOptimization._propose_next_candidate_no_dt.
        "propose_mode": propose_mode,
        # Only read when rollout_teacher=="gradient_ascent" (see
        # DirectRegretOptimization._select_x_tau_gradient_ascent).
        "rollout_ga_steps": rollout_ga_steps,
        # Failure-mode-1 diagnostic: forces identical (median) starting
        # lengthscale across all GP ensemble members instead of the diverse
        # np.linspace grid. False default -- bit-for-bit unchanged linspace
        # init when off. See DirectRegretOptimization._initialize_models.
        "uniform_ensemble": uniform_ensemble,
        "use_roi_state": use_roi_state,
        "use_roi_std_quantiles": use_roi_std_quantiles,
        "state_hyperparams_enabled": state_hyperparams_enabled,
        "use_roi_sigma_iqr": use_roi_sigma_iqr,
        "rtg_schema": rtg_schema,
        "alpha_floor": alpha_floor,
        "alpha_inference": alpha_inference,
        "lambda_rtg": lambda_rtg,
        "rtg_warmup": rtg_warmup,
        "known_optimal_value": benchmark_spec["known_optimal_value"],
        "gp": {
            # noise_constraint raised 1e-4->1e-2, matching KennedyOHaganGP's
            # own noise_lb fix -- see src/policy/dro.py's
            # DRO_LENGTHSCALE_PRIOR_LOC/SCALE module docstring.
            "kernel": gp_kernel, "noise_constraint": 1e-2,
            "lengthscale_min": lengthscale_min, "lengthscale_max": lengthscale_max,
            "num_models": gp_num_models, "verbose": False, "retrain": False,
            "ard": gp_ard,
        },
        "acquisition": {
            # Pinned per-variant rollout query-selection function ("ei" or
            # "mes" for the mes_switching experiment). Defaults to "ei" so any
            # caller that doesn't pass rollout_acq_function explicitly gets a
            # sane, standard acquisition function rather than erroring --
            # NOT a re-creation of the old shared "rotate_acq" default (that
            # branch still exists in _acquisition_function_value_botorch for
            # other uses, per instructions, but is never selected here unless
            # a caller explicitly passes rollout_acq_function="rotate_acq").
            "function": rollout_acq_function, "kappa": 2.0, "ucb_lcb_kappa": 6.0, "xi": 0.01,
            "early_stop_threshold": 1e-4, "constrain_ucb_lcb": True,
        },
        "transformer": {
            "hidden_size": dt_hidden, "num_layers": dt_layers, "num_heads": dt_heads,
            "dropout": 0.1, "lr": dt_lr, "weight_decay": 1e-5, "batch_size": 32,
            "num_epochs": num_epochs, "max_seq_length": 20,
            # AWR (advantage-weighted regression): False default, bit-for-bit
            # unchanged existing loss when off. awr_temperature=None means
            # compute adaptively per-batch (median |RTG|); set a float to
            # pin it for ablation. See _train_decision_transformer.
            "use_awr": use_awr, "awr_temperature": awr_temperature,
        },
        "simulation": {
            "num_rollouts": rollouts_per_iter, "max_rollout_length": rollout_length,
            # False (not the prior True): with True, a simulated rollout step
            # terminates the whole rollout as soon as reward < early_stop_threshold
            # -- which for sparse improvement reward is most steps (reward is
            # exactly 0 unless that step happens to beat observed_best), truncating
            # rollouts to length 1 far more often than not and starving the DT of
            # multi-step trajectories to train on. Confirmed via cfg inspection
            # this was True for every variant in both experiments (not something
            # that varied by reward type or RTG schema).
            "early_stop": False, "early_stop_threshold": 1e-4, "verbose": False,
        },
        "bo": {
            "max_iterations": bo_iterations,
            "input_dim": benchmark_spec["dim"],
            "domain_min": benchmark_spec["domain_min"],
            "domain_max": benchmark_spec["domain_max"],
            "initial_points": initial_points,
            "objective": "maximize",
            "initial_sampling_method": "lhs",
        },
    })
    if benchmark_name in _KNOWN_OPTIMAL_X:
        cfg.known_optimal_x = _KNOWN_OPTIMAL_X[benchmark_name]
    return cfg


def run_single_seed(exp_name, benchmark_name, variant_name, seed,
                     use_mes_reward=False, rtg_schema="fixed", alpha_floor=0.5,
                     alpha_inference=None, lambda_rtg=1.0, is_naive_bo=False,
                     gp_num_models=5, rollouts_per_iter=75, rollout_length=4,
                     bo_iterations=50, initial_points=5,
                     dt_hidden=128, dt_layers=4, dt_heads=4, dt_lr=1e-4,
                     gp_kernel="rbf", gp_ard=True, rtg_warmup=3, verbose=False,
                     mes_k=10, rollout_acq_function="ei", naivebo_acq_function="ei",
                     use_roi_state=False, use_roi_std_quantiles=False, state_hyperparams_enabled=True,
                     use_roi_sigma_iqr=False, observation_noise_std=0.0,
                     **kwargs):
    """
    Skip if already completed (checkpoint.py), otherwise run the full BO loop
    for this (benchmark, variant, seed), save the result, and return it.
    Dispatches to NaiveBO (no DT component) when is_naive_bo=True, else to
    DirectRegretOptimization with the given hyperparameters.

    observation_noise_std: default 0.0, no effect on any existing caller. When
    > 0, every objective evaluation gets additive N(0, observation_noise_std^2)
    noise -- a diagnostic knob (Hartmann_6D's flat-regret investigation: does
    noise prevent GP sigma collapse at converged/corner regions and break the
    resulting positive feedback loop). Applied by wrapping objective_function
    itself, so it's invisible to DirectRegretOptimization/run_naive_bo -- both
    already treat the objective as a (possibly noisy) black box.
    """
    if is_completed(exp_name, benchmark_name, variant_name, seed):
        log_global(exp_name, f"SKIPPED {benchmark_name} {variant_name} seed{seed}")
        return load_result(exp_name, benchmark_name, variant_name, seed)

    log_global(exp_name, f"STARTED {benchmark_name} {variant_name} seed{seed}")
    t_start = time.perf_counter()

    benchmark_spec = get_benchmark(benchmark_name)
    objective_function = benchmark_spec["make_objective"]()
    if observation_noise_std > 0:
        _clean_objective = objective_function
        def objective_function(x, _clean=_clean_objective, _std=observation_noise_std):
            y = _clean(x)
            return y + _std * torch.randn_like(y)

    if is_naive_bo:
        def _iter_cb(t, regret, best, iter_time):
            log_global(exp_name, f"ITER {t} {benchmark_name} {variant_name} seed{seed} regret={regret:.4f}")

        raw_result = run_naive_bo(
            objective_function=objective_function,
            domain_min=benchmark_spec["domain_min"],
            domain_max=benchmark_spec["domain_max"],
            dim=benchmark_spec["dim"],
            seed=seed,
            max_iterations=bo_iterations,
            initial_points=initial_points,
            known_optimal_value=benchmark_spec["known_optimal_value"],
            iter_callback=_iter_cb,
            naivebo_acq_function=naivebo_acq_function,
        )
        result = {k: raw_result[k] for k in RESULT_KEYS}
        result["acq_name_used"] = naivebo_acq_function
    else:
        cfg = _build_dro_config(
            exp_name, benchmark_name, variant_name, seed,
            use_mes_reward, rtg_schema, alpha_floor, alpha_inference,
            lambda_rtg, rtg_warmup, benchmark_spec,
            gp_num_models, rollouts_per_iter, rollout_length, bo_iterations,
            initial_points, dt_hidden, dt_layers, dt_heads, dt_lr,
            gp_kernel, gp_ard, verbose, mes_k=mes_k,
            rollout_acq_function=rollout_acq_function, use_roi_state=use_roi_state,
            use_roi_std_quantiles=use_roi_std_quantiles, state_hyperparams_enabled=state_hyperparams_enabled,
            use_roi_sigma_iqr=use_roi_sigma_iqr,
        )
        dro = DirectRegretOptimization(cfg, objective_function)
        if has_resume_checkpoint(exp_name, benchmark_name, variant_name, seed):
            # A prior process for this exact (benchmark, variant, seed) was
            # killed mid-run (e.g. a cluster job hitting its walltime limit)
            # -- continue from its last completed real iteration instead of
            # restarting from scratch. See DirectRegretOptimization.
            # load_checkpoint/save_checkpoint for exactly what's restored.
            start_iteration = dro.load_checkpoint()
            log_global(exp_name,
                f"RESUMED {benchmark_name} {variant_name} seed{seed} from iteration {start_iteration}")
            dro.run_optimization(start_iteration=start_iteration)
        else:
            dro.run_optimization()

        history = dro.iteration_log_history
        result = {
            "regret_curve": [d["regret"] for d in history],
            "best_observed": [d["best"] for d in history],
            "mean_reward": [d["mean_reward"] for d in history],
            "zero_frac": [d["zero_frac"] for d in history],
            "rtg_target": [d["rtg_target"] for d in history],
            "batch_max_rtg": [d["batch_max_rtg"] for d in history],
            "running_max_rtg": [d["running_max_rtg"] for d in history],
            "neg_rtg_frac": [d.get("neg_rtg_frac") for d in history],
            "gp_refit_time": [d.get("gp_refit_time") for d in history],
            "rollout_sim_time": [d.get("rollout_sim_time") for d in history],
            "dt_train_time": [d.get("dt_train_time") for d in history],
            "real_query_time": [d.get("real_query_time") for d in history],
            "rollout_action_diversity": [d.get("rollout_action_diversity") for d in history],
            "mu_proposed": [d.get("mu_proposed") for d in history],
            "sigma_proposed": [d.get("sigma_proposed") for d in history],
            "mu_at_true_opt": [d.get("mu_at_true_opt") for d in history],
            "corner_proximity": [d.get("corner_proximity") for d in history],
            "iter_times": [d["iter_time"] for d in history],
            "acq_name_used": rollout_acq_function,
        }
        # Per-iteration diagnostic: was the DT's inference-time RTG target
        # positive? Added for entropy_joint on Hartmann_6D, where
        # batch_max_rtg is expected to stay negative throughout (confirmed
        # via sanity check -- Hartmann_6D's b_tau stays below the
        # exp(-euler_gamma-1) zero-crossing even at n_initial=25). Cheap and
        # harmless to log for every schema, not just entropy_joint.
        result["rtg_target_sign_positive"] = [rtg_target > 0 for rtg_target in result["rtg_target"]]
        if rtg_schema == "quantile":
            result["L_pinball"] = [d.get("L_pinball") for d in history]
            result["L_loc"] = [d.get("L_loc") for d in history]
            result["Q_hat_inference"] = [d.get("Q_hat_inference") for d in history]
            result["calibration"] = [d.get("calibration") for d in history]
            result["quantile_spread"] = [d.get("quantile_spread") for d in history]

    run_time = time.perf_counter() - t_start
    save_result(exp_name, benchmark_name, variant_name, seed, result)
    # Only meaningful for the DRO path (is_naive_bo never creates one) --
    # a harmless no-op otherwise. Deleted AFTER save_result's JSON+.done
    # flag are both written, so a crash in between leaves the resume
    # checkpoint in place rather than losing it right before it might still
    # be needed.
    delete_resume_checkpoint(exp_name, benchmark_name, variant_name, seed)
    log_global(exp_name, f"COMPLETED {benchmark_name} {variant_name} seed{seed} time={run_time:.1f}s")
    return result


def _build_mf_dro_config(exp_name, benchmark_base_name,
                          variant_name, seed,
                          # DEFAULTS CHANGED 2026-08-26 (user-directed):
                          # M 10 -> 3, n_roi_candidates 200 -> 600, refinement
                          # left OFF. Evidence and its limits:
                          #   M: h81 measured M10/M3 wall-time at only 1.39x, so
                          #      shrinking M frees far less compute than the
                          #      premise assumed. Its statistical verdict was
                          #      still WITHHELD at 13/15 when this was set.
                          #   pool 600: measured 1.96x BASE cost. Buys 4.25 pts
                          #      on Borehole (n=3) and, on Hartmann at n=10,
                          #      8.91% -> 6.64% with sd 7.39 -> 3.22 -- but only
                          #      6/10 paired wins at p=0.5566, i.e. NOT a
                          #      demonstrated win. Adopted as an engineering
                          #      choice on cost-efficiency and variance, not as
                          #      a performance claim. Pool 1000 costs 5.87x for
                          #      1.8 pts more: ~3.5x worse per unit cost.
                          #   refinement: ~1.5x on its own, and h65 showed its
                          #      variance benefit does not generalise off
                          #      Borehole. Weakest of the three levers; OFF.
                          # Combined M3 + pool600 ~= 1.41x BASE overall.
                          M=3,
                          rollout_length=8,
                          # ITEM 3: raised 7 -> 20 (needed for the item-4
                          # gate measurement to be statistically meaningful
                          # at 10 ensemble-member groups).
                          rollouts_per_model=20,
                          bo_iterations=30,
                          initial_points=5,
                          dt_hidden=128,
                          dt_layers=4,
                          dt_heads=4,
                          dt_lr=1e-4,
                          num_epochs=100,
                          lambda_fid=1.0,
                          alpha_rtg=0.5,
                          alpha_btg=0.5,
                          max_seq_length=80,
                          minimum_hf_fraction=0.25,
                          real_hf_warmup=2,
                          cost_budget=None,
                          initial_hf=30,
                          initial_lf=30,
                          use_sequential_init=False,
                          use_rtg_grounding=False,
                          dkl_threshold=30,
                          bes_delta=0.0,
                          # DEFAULT REVERTED to the regression head. Change
                          # 1a had made candidate scoring the default; h45
                          # measured the two at matched settings (Hartmann
                          # 6D, cost budget 200, initial_hf=36/initial_lf=60)
                          # and the regression head won on 5/6 seeds, mean
                          # 0.3711 vs 0.4523. It is also the cheaper path:
                          # no 200-candidate pool is built at inference.
                          #
                          # Change 1a's argument is NOT refuted and is kept
                          # on record in mf_dro.py's __init__ -- MSE
                          # regression does average the teacher's argmax over
                          # repeated tau=0 states, and the STATE-DIAG line
                          # confirms the precondition holds (200 rollouts ->
                          # ~10 unique tau=0 states). h45's one catastrophic
                          # seed (1.1818 vs a 0.2308 median) is consistent
                          # with that failure mode firing occasionally rather
                          # than uniformly. Set use_candidate_scoring=True to
                          # restore the old head.
                          use_candidate_scoring=False,
                          rollout_policy="mes",
                          # ITEM 1: default switched to regret-based RTG.
                          rollout_reward="improvement",
                          use_lf_screened_init=False,
                          use_real_rollout_queries=False,
                          refit_hyperparams_in_rollout=False,
                          known_optimal_x=None,
                          known_secondary_x=None):
    """
    Config for MF-DRO. c_H and c_L from benchmark registry.
    NOTE: rollouts_per_model (7) NOT rollouts_per_iter (75).
    Total rollouts = M * rollouts_per_model = 70.
    """
    hf_spec = get_benchmark(benchmark_base_name + "_HF")
    lf_spec = get_benchmark(benchmark_base_name + "_LF")
    cfg = SimpleNamespace(
        exp_name=exp_name,
        benchmark_name=benchmark_base_name,
        variant_name=variant_name,
        seed=seed,
        M=M,
        rollout_length=rollout_length,
        rollouts_per_model=rollouts_per_model,
        bo_iterations=bo_iterations,
        initial_points=initial_points,
        dt_hidden=dt_hidden,
        dt_layers=dt_layers,
        dt_heads=dt_heads,
        dt_lr=dt_lr,
        num_epochs=num_epochs,
        lambda_fid=lambda_fid,
        alpha_rtg=alpha_rtg,
        alpha_btg=alpha_btg,
        minimum_hf_fraction=minimum_hf_fraction,
        real_hf_warmup=real_hf_warmup,
        cost_budget=cost_budget,
        initial_hf=initial_hf,
        use_sequential_init=use_sequential_init,
        use_rtg_grounding=use_rtg_grounding,
        dkl_threshold=dkl_threshold,
        bes_delta=bes_delta,
        use_candidate_scoring=use_candidate_scoring,
        rollout_policy=rollout_policy,
        rollout_reward=rollout_reward,
        use_lf_screened_init=use_lf_screened_init,
        use_real_rollout_queries=use_real_rollout_queries,
        refit_hyperparams_in_rollout=refit_hyperparams_in_rollout,
        initial_lf=initial_lf,
        max_seq_length=max_seq_length,
        c_H=hf_spec["cost"],
        c_L=lf_spec["cost"],
        true_opt=hf_spec["known_optimal_value"]
    )
    # known_optimal_x: explicit override takes precedence over the
    # _KNOWN_OPTIMAL_X auto-lookup-by-benchmark_base_name -- needed for
    # Ackley_10D specifically, whose name collides between the pre-existing
    # standalone SF-only entry ([-32.768,32.768]^10 domain, optimum at the
    # origin, _KNOWN_OPTIMAL_X's own [0.0]*10) and the newer MF pair
    # (Ackley_10D_HF/LF, [0,1]^10 domain, optimum at [0.5]*10) -- same
    # benchmark_base_name string, two different benchmarks, two different
    # true optimum locations. The dict itself is left untouched (still
    # correct for the SF-only entry); callers building configs for the MF
    # pair must pass known_optimal_x=[0.5]*10 explicitly.
    if known_secondary_x is not None:
        cfg.known_secondary_x = known_secondary_x
    if known_optimal_x is not None:
        cfg.known_optimal_x = known_optimal_x
    elif benchmark_base_name in _KNOWN_OPTIMAL_X:
        cfg.known_optimal_x = _KNOWN_OPTIMAL_X[benchmark_base_name]
    return cfg


def run_mf_single_seed(exp_name, benchmark_base_name,
                        variant_name, seed,
                        bo_iterations=30,
                        **config_overrides):
    """
    MF-DRO runner. Uses MF_RESULT_KEYS, save_mf_result, is_mf_completed.
    Dispatches to DirectMFRegretOptimization. Does not touch
    run_single_seed / RESULT_KEYS / save_result / load_result / is_completed.

    config_overrides: forwarded to _build_mf_dro_config (e.g. M,
    rollouts_per_model, num_epochs, initial_points -- used by the smoke test
    to shrink the run for speed).
    """
    if is_mf_completed(exp_name, benchmark_base_name,
                        variant_name, seed):
        print(f"SKIPPED: {benchmark_base_name} "
              f"{variant_name} seed{seed}")
        return load_mf_result(exp_name, benchmark_base_name,
                               variant_name, seed)

    hf_spec = get_benchmark(benchmark_base_name + "_HF")
    lf_spec = get_benchmark(benchmark_base_name + "_LF")
    f_hf = hf_spec["make_objective"]()
    f_lf = lf_spec["make_objective"]()
    d = hf_spec["dim"]
    # [2, d] BoTorch convention (row 0 = lower, row 1 = upper) -- matches
    # ko.fit()'s Normalize transform / every other bounds consumer in this
    # codebase (dro.py's self.bounds[0]/self.bounds[1], mf_baselines.py's
    # MultiFidelityBenchmark.bounds). Built from the registry's own
    # domain_min/domain_max rather than hardcoded [0,1], so this stays
    # correct for any future benchmark whose domain isn't [0,1]^d.
    bounds = torch.tensor(
        [hf_spec["domain_min"], hf_spec["domain_max"]], dtype=torch.float64
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    config = _build_mf_dro_config(
        exp_name, benchmark_base_name, variant_name, seed,
        bo_iterations=bo_iterations, **config_overrides
    )
    config.seed = seed

    mf_dro = DirectMFRegretOptimization(config, f_hf, f_lf, bounds)
    result = mf_dro.run()

    save_mf_result(exp_name, benchmark_base_name,
                    variant_name, seed, result)
    print(f"DONE: {benchmark_base_name} {variant_name} seed{seed} "
          f"final_regret={result['hf_regret_curve'][-1]:.4f}")
    return result
