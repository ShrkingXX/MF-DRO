import torch
import torch.optim as optim
import numpy as np
import gpytorch
import hydra
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from tqdm import tqdm

import warnings
import sys
import os
import math
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.model import DecisionTransformer
from src.objectives import Ackley, Rosenbrock, Levy
from src.policy.base import BaseBayesianOptimizer
from mes_reward import compute_mes_reward
from checkpoint import log_iter, log_global, resume_checkpoint_path
from quantile_rtg import compute_pinball_loss, interpolate_quantile
from gumbel_thompson import thompson_sample_y_star, fit_gumbel_to_samples

from botorch.models import SingleTaskGP # BoTorch standard GP model
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.exceptions import InputDataWarning
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel # Standard GPyTorch kernels
from gpytorch.constraints import GreaterThan
from gpytorch.priors import LogNormalPrior
from gpytorch.mlls import ExactMarginalLogLikelihood # Often used for training

# Same lengthscale-regularization convention as src/models/ko_gp.py's
# KennedyOHaganGP (see that module's LENGTHSCALE_PRIOR_LOC/SCALE docstring):
# a LogNormalPrior centered at 0.5 (geometric center of the useful RBF/Matern
# lengthscale range on normalized [0,1]^d inputs) replaces having no
# regularization at all on lengthscale here -- _construct_gp_model previously
# left lengthscale fully unconstrained beyond GPyTorch's default Positive
# transform, with noise_constraint=1e-4 (same value MF-DRO's KennedyOHaganGP
# used before it was identified there as encouraging near-interpolation --
# very short lengthscale, near-exact fit to each training point -- on sparse
# high-D data). Both fixes ported here unchanged.
DRO_LENGTHSCALE_PRIOR_LOC = math.log(0.5)
DRO_LENGTHSCALE_PRIOR_SCALE = 0.5
from botorch.acquisition import (
    ExpectedImprovement,
    UpperConfidenceBound,
    ProbabilityOfImprovement,
    qMaxValueEntropy,
)
from botorch.acquisition.analytic import LogExpectedImprovement

DEFAULT_DTYPE = torch.float64

class DirectRegretOptimization(BaseBayesianOptimizer):
    """
    Direct Regret Optimization using GP ensembles and Decision Transformer,
    inheriting from BaseBayesianOptimizer.
    """
    # Number of GP hyperparameters (lengthscale, outputscale) included per ensemble
    # member in the DT state vector. Shared by _get_state_dim and _extract_state so
    # the declared state dimension always matches what's actually assembled.
    GP_STATE_PARAMS_PER_MODEL = 2

    def __init__(self, config: DictConfig, objective_function):
        # Call parent constructor first
        super().__init__(config, objective_function)

        # --- Specific Initializations for Direct Regret ---

        # Configs specific to this method (assuming nested structure in main config)
        self.gp_config = config.gp # Expects a 'gp' sub-config
        self.transformer_config = config.transformer # Expects a 'transformer' sub-config
        self.simulation_config = config.simulation # Expects a 'simulation' sub-config
        self.acquisition_config = config.acquisition # Expects an 'acquisition' sub-config

        # --- Ablation / experiment-orchestration flags (all default to original DRO behavior) ---
        self.seed = getattr(config, 'seed', None)
        self.exp_name = getattr(config, 'exp_name', None)
        self.variant_name = getattr(config, 'variant_name', None)
        self.benchmark_name = getattr(config, 'benchmark_name', None)
        self.use_mes_reward = getattr(config, 'use_mes_reward', False)
        self.mes_k = getattr(config, 'mes_k', 10) # MC sample count for compute_mes_reward's Term2
        # When True, _extract_state appends [mean(mu_roi), mean(sigma_roi)] per
        # GP ensemble member (2*num_models extra slots) -- GP posterior
        # statistics over roi_candidates, giving the DT a location-sensitive
        # signal it otherwise entirely lacks (see Track 5 diagnostic finding:
        # the un-augmented state has zero location-sensitive GP information).
        # Default False: _get_state_dim()/_extract_state produce the original
        # 18-dim state, so every existing experiment is unaffected.
        self.use_roi_state = getattr(config, 'use_roi_state', False)
        # When True, _extract_state appends [min, p25, median, p75, max] of
        # sigma_roi (per-candidate posterior std over roi_candidates) per GP
        # ensemble member (5*num_models extra slots) -- a distributional view
        # of GP uncertainty, independent of and combinable with use_roi_state.
        # Default False: no effect on state_dim/_extract_state.
        self.use_roi_std_quantiles = getattr(config, 'use_roi_std_quantiles', False)
        # When True, _extract_state appends [mean(sigma_roi), IQR(sigma_roi)]
        # per GP ensemble member (2*num_models extra slots), both normalized
        # by that member's cached initial sigma_roi scale. Unlike use_roi_state
        # (which pairs mean_sigma_roi with mean_mu_roi), this pairs it with the
        # interquartile range of the per-candidate posterior std instead --
        # IQR distinguishes "uniformly uncertain everywhere" (low IQR) from
        # "some candidates saturated/low-std, others still high-std" (high
        # IQR), a distinction mean_sigma_roi alone cannot make. Independent of
        # and combinable with use_roi_state/use_roi_std_quantiles. Default
        # False: no effect on state_dim/_extract_state.
        self.use_roi_sigma_iqr = getattr(config, 'use_roi_sigma_iqr', False)
        # When False, _extract_state omits slots 0-9 (raw GP kernel
        # hyperparameters) entirely -- an ablation toggle to test whether they
        # carry real signal or are dead weight relative to spatial/ROI
        # features. Default True: original behavior, hyperparameters always
        # included, every existing experiment unaffected.
        self.state_hyperparams_enabled = getattr(config, 'state_hyperparams_enabled', True)
        # Per-ensemble-member reference sigma_roi, captured the first time
        # _extract_state computes ROI features for that model -- used to
        # normalize mean_sigma_roi onto a comparable scale to the rest of the
        # state (Bug 3 fix). None until first captured; list of length
        # num_models thereafter, each entry set once and never overwritten.
        self._initial_sigma_roi_per_model = None
        # Reference |best_value| for normalizing mean_mu_roi (use_roi_state),
        # captured once on the first _extract_state call that computes ROI
        # features and frozen thereafter -- same rationale and pattern as
        # _initial_sigma_roi_per_model. The prior implementation re-derived
        # abs(best_value) fresh every call, which is non-stationary (the scale
        # shifts every time a better point is found) and can approach zero
        # when best_value itself starts near zero (e.g. Rosenbrock's minimum
        # at 0), exploding mean_mu_roi/mu_denom to very large values.
        self._initial_mu_denom = None
        # Controls which acquisition function is used for query selection INSIDE
        # simulated rollouts (_optimize_acquisition). Decoupled from
        # acquisition_config.function (which still governs non-rollout callers,
        # e.g. direct/test invocations with acq_name=None). Valid values: "ei",
        # "ucb", "pi", "mes". Defaults to acquisition_config.function for
        # backward compatibility -- every existing variant that doesn't set this
        # explicitly behaves exactly as before.
        self.rollout_acq_function = getattr(
            config, 'rollout_acq_function', getattr(self.acquisition_config, 'function', 'ei')
        )
        # "argmax" (default, bit-for-bit unchanged): _simulate_trajectory's
        # rollout action selection uses _optimize_acquisition as always.
        # "softmax": routes through _optimize_acquisition_softmax instead --
        # see that method's docstring.
        self.rollout_teacher = getattr(config, 'rollout_teacher', 'argmax')
        self.softmax_temperature = getattr(config, 'softmax_temperature', 0.5)
        self.rtg_schema = getattr(config, 'rtg_schema', 'fixed')
        self.alpha_floor = getattr(config, 'alpha_floor', 0.5)
        # MC sample count for Thompson-sampling the Gumbel scale b_tau, used only
        # by rtg_schema=="joint" (see _simulate_trajectory_joint).
        self.rtg_joint_k_thompson = getattr(config, 'rtg_joint_k_thompson', 100)
        self.alpha_inference = getattr(config, 'alpha_inference', None)
        self.lambda_rtg = getattr(config, 'lambda_rtg', 1.0)
        self.rtg_warmup = getattr(config, 'rtg_warmup', 3)
        if self.rtg_schema not in ("fixed", "dynamic", "floored", "quantile", "joint", "entropy_joint"):
            raise ValueError(f"Unknown rtg_schema: {self.rtg_schema}")
        # True global minimum of the *raw* (pre-negation) objective, used to compute
        # simple regret for logging. Defaults to 0.0, matching every benchmark
        # currently in this codebase (Ackley/Rosenbrock/Levy all have optimal_value=0
        # by construction); benchmarks with a non-zero known optimum (e.g. Eggholder)
        # must pass this explicitly or logged regret will be offset by a constant.
        self.known_optimal_value = getattr(config, 'known_optimal_value', 0.0)
        # Location (not value) of the known global optimum, shape [d]. Optional,
        # diagnostic-only (Track 1's mu_at_true_opt) -- when None, that
        # diagnostic is skipped entirely rather than computed against a
        # meaningless default location.
        self.known_optimal_x = getattr(config, 'known_optimal_x', None)

        # Persistent RTG-schema state: initialized once here, updated every real BO
        # iteration in _propose_next_candidate, never reset mid-run.
        self.running_max_rtg = 0.0
        # Set at the end of _propose_next_candidate, consumed (and cleared) by
        # _update_data_and_best so per-iteration logging fires exactly once per real
        # BO step and never during initial-point sampling.
        self._pending_log = False
        self._last_iter_diagnostics = {}
        # Per-phase wall-clock breakdown (mes_switching_v2 cluster experiment
        # cost audit). _last_gp_refit_time is set by _update_models (runs
        # before _propose_next_candidate each real iteration, so it's always
        # current by the time _last_iter_diagnostics is built there);
        # _last_rollout_sim_time/_last_dt_train_time/_last_real_query_time and
        # _last_rollout_action_diversity are set inside _propose_next_candidate
        # itself. All default to 0.0/None here only so a log_dict built before
        # the first real iteration (shouldn't normally happen) has a
        # well-defined value rather than an AttributeError.
        self._last_gp_refit_time = 0.0
        self._last_rollout_sim_time = 0.0
        self._last_dt_train_time = 0.0
        self._last_real_query_time = 0.0
        self._last_rollout_action_diversity = None
        # Track 1 diagnostics (mu_proposed/sigma_proposed/mu_at_true_opt/
        # corner_proximity), set at the end of _propose_next_candidate. Kept
        # separate from _last_iter_diagnostics only because that's where the
        # spec for this put it -- merged into the same log_dict either way.
        self._last_diagnostics = {}
        # Every real iteration's logged dict, in order -- lets callers (e.g.
        # run_single_seed) build a checkpoint-ready result without re-parsing
        # the .log file written by log_iter.
        self.iteration_log_history = []
        # Populated by _train_decision_transformer (L_loc/L_pinball of the final
        # training epoch) and merged into _last_iter_diagnostics when quantile RTG
        # is active.
        self._last_train_diagnostics = {}
        # Q_hat [M] from the most recent _compute_quantile_rtg_target call, or None
        # when that wasn't invoked this iteration (non-quantile schema, or still
        # within warmup).
        self._last_q_hat = None
        # Running real-BO-history buffer: one (state, action) pair appended per
        # real iteration in _propose_next_candidate. Used to build the sliding
        # window for quantile-schema inference (Step 5); irrelevant otherwise.
        self.real_history_states = []
        self.real_history_actions = []

        # Initialize GP Ensemble (models will be created but empty)
        self.gp_ensemble = [] # List of dicts: {'model': ..., 'likelihood': ...}
        # Note: _initialize_models will populate this based on data later

        # Initialize Decision Transformer
        # State dim needs careful calculation based on actual GP params + extras
        state_dim = self._get_state_dim()
        action_dim = self.bo_config.input_dim
        self.decision_transformer = DecisionTransformer(
            self.transformer_config, state_dim, action_dim
        ).to(self.device, self.dtype)
        # Causal masking + the quantile head's forward path are only engaged when
        # rtg_schema=="quantile"; every other schema gets the DT's original
        # full-bidirectional-attention behavior unchanged.
        self.decision_transformer.use_quantile_rtg = (self.rtg_schema == "quantile")

        # Initialize Optimizer for the Transformer
        self.optimizer = optim.Adam(
            self.decision_transformer.parameters(),
            lr=self.transformer_config.lr,
            weight_decay=getattr(self.transformer_config, 'weight_decay', 0.0) # Optional weight decay
        )

        # Override initial sampling method if specified in config
        self.initial_sampling_method = getattr(self.bo_config, 'initial_sampling_method', 'lhs') # Default to LHS for this class


    def _update_data_and_best(self, new_x: torch.Tensor, new_y_val: float):
        """
        Extends the base class's data/best-tracking with per-iteration checkpoint
        logging (Step 3, Change 4). Only fires for real BO iterations -- gated by
        _pending_log, which _propose_next_candidate sets and this method clears, so
        initial-point sampling (which also calls this method) is never logged here.
        """
        t_now = time.perf_counter()
        iter_time = t_now - getattr(self, '_iter_t_start', t_now)

        super()._update_data_and_best(new_x, new_y_val)

        if self._pending_log:
            n_initial = getattr(self.bo_config, 'initial_points', 0)
            current_iter = self.data_x.shape[0] - n_initial
            # self.best_observed_value (base class) is set directly from the
            # objective function's return value, which BoTorch returns as a 0-d
            # Tensor -- so best_observed_value is a Tensor whenever a new best was
            # just found this iteration, and a plain float only on iterations where
            # it's still the -inf/+inf sentinel from __init__. Normalize to a plain
            # float here so regret/best are always JSON-serializable for
            # checkpointing (str formatting silently tolerates 0-d Tensors, which is
            # why this went unnoticed until save_result's json.dump hit it).
            best_observed = self.best_observed_value.item() if torch.is_tensor(self.best_observed_value) \
                else self.best_observed_value
            simple_regret = (-best_observed - self.known_optimal_value) if self.objective_mode == "maximize" \
                else (best_observed - self.known_optimal_value)
            diag = self._last_iter_diagnostics
            track1_diag = self._last_diagnostics

            log_dict = {
                "iter": current_iter,
                "regret": simple_regret,
                "best": best_observed,
                "mean_reward": diag.get("mean_reward"),
                "zero_frac": diag.get("zero_frac"),
                "rtg_target": diag.get("rtg_target"),
                "batch_max_rtg": diag.get("batch_max_rtg"),
                "running_max_rtg": diag.get("running_max_rtg"),
                "neg_rtg_frac": diag.get("neg_rtg_frac"),
                "gp_refit_time": diag.get("gp_refit_time"),
                "rollout_sim_time": diag.get("rollout_sim_time"),
                "dt_train_time": diag.get("dt_train_time"),
                "real_query_time": diag.get("real_query_time"),
                "rollout_action_diversity": diag.get("rollout_action_diversity"),
                "mu_proposed": track1_diag.get("mu_proposed"),
                "sigma_proposed": track1_diag.get("sigma_proposed"),
                "mu_at_true_opt": track1_diag.get("mu_at_true_opt"),
                "corner_proximity": track1_diag.get("corner_proximity"),
                "iter_time": iter_time,
            }
            if self.decision_transformer.use_quantile_rtg:
                log_dict["L_pinball"] = diag.get("L_pinball")
                log_dict["L_loc"] = diag.get("L_loc")
                log_dict["Q_hat_inference"] = diag.get("Q_hat_inference")
                log_dict["calibration"] = diag.get("calibration")
                log_dict["quantile_spread"] = diag.get("quantile_spread")

            # Always accumulate in-memory (used by run_single_seed to build the
            # checkpoint result dict), independent of whether file-based
            # checkpoint logging below is configured.
            self.iteration_log_history.append(log_dict)

            if self.exp_name and self.benchmark_name and self.variant_name:
                log_iter(self.exp_name, self.benchmark_name, self.variant_name, self.seed, log_dict)
                log_global(self.exp_name,
                    f"ITER {current_iter} {self.benchmark_name} {self.variant_name} "
                    f"seed{self.seed} regret={simple_regret:.4f} rtg_target={diag.get('rtg_target', float('nan')):.4f}")
                # Resumable in-progress checkpoint, saved every real iteration.
                # log_iter (above) already gives per-iteration partial-progress
                # visibility for free (a human-readable line appended to disk
                # immediately) -- this is the separate piece that was actually
                # missing: enough state to CONTINUE training (not just report
                # progress) after a process kill, e.g. a cluster job hitting its
                # walltime limit. See save_checkpoint's docstring for exactly
                # what's captured and why.
                self.save_checkpoint()

        self._pending_log = False


    def save_checkpoint(self):
        """
        Snapshot enough state to resume this run's real-BO-iteration loop from
        exactly where it left off after a process kill (e.g. a cluster job's
        walltime limit) -- called automatically after every real iteration
        (from _update_data_and_best, mirroring log_iter/log_global's gating on
        self.exp_name/benchmark_name/variant_name all being set). Separate
        from checkpoint.py's save_result/load_result, which only persist the
        FINAL summary once a run completes.

        The decision transformer is warm-started and continually trained
        across the WHOLE run (self.optimizer is created once in __init__ and
        reused every _train_decision_transformer call, accumulating Adam's
        momentum/variance buffers over all bo_iterations) -- not retrained
        from scratch each iteration. So resuming correctly requires the
        optimizer's state_dict, not just the model weights, or a resumed run's
        training dynamics would diverge from an uninterrupted one even before
        accounting for RNG.

        GP ensemble state is deliberately NOT included: _update_models()
        always rebuilds every ensemble member from scratch off
        self.data_x/self.data_y (see its docstring for why -- input/outcome
        transforms are baked in at construction time), so restoring
        data_x/data_y is sufficient; the resumed run_optimization() loop's
        first _update_models() call reconstructs identical GPs with no
        special-casing needed.

        RNG state (torch + numpy) is saved so a resumed run's random draws
        (rollout sampling, GP init, etc.) continue the same stream a single
        uninterrupted process would have produced, rather than silently
        restarting from a fresh unseeded state -- this project has leaned
        on bit-identical reproducibility (same seed+config -> same result)
        repeatedly, and a resume boundary shouldn't quietly break that.

        Written atomically (temp file + os.replace) so a kill mid-write never
        leaves a corrupt checkpoint that a subsequent resume attempt would
        crash trying to load.
        """
        path = resume_checkpoint_path(self.exp_name, self.benchmark_name, self.variant_name, self.seed)
        tmp_path = path + ".tmp"
        checkpoint = {
            "data_x": self.data_x.detach().clone(),
            "data_y": self.data_y.detach().clone(),
            "dt_state_dict": self.decision_transformer.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_observed_value": self.best_observed_value,
            "running_max_rtg": self.running_max_rtg,
            "real_history_states": self.real_history_states,
            "real_history_actions": self.real_history_actions,
            "_last_q_hat": self._last_q_hat,
            "_initial_sigma_roi_per_model": self._initial_sigma_roi_per_model,
            "_initial_mu_denom": self._initial_mu_denom,
            "iteration_log_history": self.iteration_log_history,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
        }
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)


    def load_checkpoint(self) -> int:
        """
        Restore state saved by save_checkpoint. Returns the number of real
        iterations already completed -- pass this straight through as
        run_optimization(start_iteration=...) to continue rather than restart.

        Caller must construct this DirectRegretOptimization with the SAME
        cfg/objective_function as the original run first (architecture,
        hyperparameters, exp_name/benchmark_name/variant_name/seed) -- this
        only overwrites the mutable run state on top of that already-built
        skeleton, it does not reconstruct the config itself.
        """
        path = resume_checkpoint_path(self.exp_name, self.benchmark_name, self.variant_name, self.seed)
        checkpoint = torch.load(path, map_location=self.device)
        self.data_x = checkpoint["data_x"].to(self.device, self.dtype)
        self.data_y = checkpoint["data_y"].to(self.device, self.dtype)
        self.decision_transformer.load_state_dict(checkpoint["dt_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_observed_value = checkpoint["best_observed_value"]
        self.running_max_rtg = checkpoint["running_max_rtg"]
        self.real_history_states = checkpoint["real_history_states"]
        self.real_history_actions = checkpoint["real_history_actions"]
        self._last_q_hat = checkpoint["_last_q_hat"]
        self._initial_sigma_roi_per_model = checkpoint["_initial_sigma_roi_per_model"]
        self._initial_mu_denom = checkpoint["_initial_mu_denom"]
        self.iteration_log_history = checkpoint["iteration_log_history"]
        torch.set_rng_state(checkpoint["torch_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        n_initial = getattr(self.bo_config, 'initial_points', 0)
        completed_real_iters = self.data_x.shape[0] - n_initial
        return completed_real_iters


    def sample_initial_points(self, method=None):
        """ Overrides base sampler to use the method preferred by this class (e.g., LHS)"""
        if method is None:
             method = self.initial_sampling_method

        if method.lower() == 'lhs':
             self._sample_initial_points_lhs()
        else:
             # Fallback to parent implementation for 'sobol', 'random', etc.
             super().sample_initial_points(method=method)


    def _sample_initial_points_lhs(self):
        """ Sample initial points using Latin Hypercube Sampling (basic implementation) """
        n_initial = getattr(self.bo_config, 'initial_points', 5)
        dim = self.bo_config.input_dim

        if n_initial <= 0: return
        if self.data_x.shape[0] > 0: return # Already have data

        print(f"Sampling {n_initial} initial points using Latin Hypercube Sampling...")

        # Basic LHS: grid + shuffle permutation per dimension
        points = torch.zeros((n_initial, dim), device=self.device, dtype=self.dtype)
        for i in range(dim):
            # Create evenly spaced points in [0, 1] range first
            grid_coords = torch.linspace(0.0, 1.0, n_initial + 1)[:-1] + 0.5 / n_initial
            perm = torch.randperm(n_initial, device=self.device)
            points[:, i] = grid_coords[perm] # Apply permutation

        # Scale points to the actual domain
        domain_min = self.bounds[0]
        domain_max = self.bounds[1]
        initial_x = domain_min + (domain_max - domain_min) * points

        # Evaluate points
        initial_y = torch.zeros(n_initial, device=self.device, dtype=self.dtype)
        for i in range(n_initial):
            y_val = self.objective_function(initial_x[i])
            if isinstance(y_val, torch.Tensor):
                initial_y[i] = y_val
            else:
                initial_y[i] = torch.tensor(y_val, device=self.device, dtype=self.dtype)

        # Update data and best observation
        for i in range(n_initial):
             self._update_data_and_best(initial_x[i].unsqueeze(0), initial_y[i].item())

        if self.verbose:
            best_val, _ = self.get_best()
            print(f"Completed LHS initial sampling. Initial best value: {best_val:.4f}")


    def _get_state_dim(self):
        """Calculate the dimension of the state vector for the Decision Transformer."""
        # Must match the per-model param count actually assembled in _extract_state
        # (previously this was derived from a throwaway ExactGPModel with a different
        # parameterization than the real ensemble's SingleTaskGP models, which could
        # silently desync the declared state_dim from the assembled state vector).
        num_models = self.gp_config.num_models
        # N * params (0 if state_hyperparams_enabled=False, an ablation toggle)
        # + best_value + iteration + best_position
        hyperparam_dim = num_models * self.GP_STATE_PARAMS_PER_MODEL if getattr(self, 'state_hyperparams_enabled', True) else 0
        state_dim = hyperparam_dim + 2 + self.bo_config.input_dim
        if getattr(self, 'use_roi_state', False):
            # [mean(mu_roi), mean(sigma_roi)] per ensemble member.
            state_dim += 2 * num_models
        if getattr(self, 'use_roi_std_quantiles', False):
            # [min, p25, median, p75, max] of sigma_roi per ensemble member.
            state_dim += 5 * num_models
        if getattr(self, 'use_roi_sigma_iqr', False):
            # [mean(sigma_roi), IQR(sigma_roi)] per ensemble member.
            state_dim += 2 * num_models
        # Either ROI-feature flag above being True is sufficient to resize the
        # DT's constructed input_dim too, since __init__ passes this method's
        # return value straight through as state_dim to the DecisionTransformer
        # constructor.
        if getattr(self.gp_config, 'verbose', False):
            print(f"Calculated state dimension: {state_dim}")
        return state_dim


    def _initialize_models(self):
        """
        Set up the GP ensemble slots (one per intended varying initial lengthscale).
        Actual model construction + fitting happens in _update_models, which
        rebuilds each ensemble member from scratch every call (see there for why).
        """
        print("Initializing BoTorch GP ensemble...")
        self.gp_ensemble = [] # Clear any previous models
        if self.data_x is None or self.data_y is None:
            # BoTorch models require training data at initialization
            print(
                "WARN: Training data (self.data_x, self.data_y) is None. "
                "Cannot initialize models. Ensure data is loaded or passed."
            )
            return

        min_scale = self.gp_config.lengthscale_min
        max_scale = self.gp_config.lengthscale_max
        num_models = self.gp_config.num_models
        # Generate the initial lengthscales to try for each model
        initial_lengthscales = np.linspace(min_scale, max_scale, num_models)

        # Failure-mode-1 diagnostic (uniform_ensemble ablation): identical
        # starting lengthscale for every member instead of the diverse
        # linspace grid, to test whether ensemble basin-diversity is what
        # drives the DT's multimodal-action-averaging failure. Noise is
        # already uniform across members regardless of this flag --
        # self.gp_config.noise_constraint is a single, non-member-varying
        # value in _construct_gp_model, nothing to change there. False
        # default -- bit-for-bit unchanged linspace init when off.
        if getattr(self.config, 'uniform_ensemble', False):
            fixed_ls = float(np.median(initial_lengthscales))
            initial_lengthscales = np.full(num_models, fixed_ls)

        for i, initial_ls in enumerate(initial_lengthscales):
            if self.verbose: print(f"Initializing model {i} with initial lengthscale: {initial_ls:.4f}")
            self.gp_ensemble.append({
                'model': None, 'likelihood': None, 'id': i,
                'initial_lengthscale': float(initial_ls),
            })

        if self.verbose: print(f"Successfully initialized {len(self.gp_ensemble)} GP ensemble slots.")
        if self.verbose: print("Proceeding to initial model fitting (_update_models)...")
        self._update_models()


    def _construct_gp_model(self, initial_lengthscale: float) -> SingleTaskGP:
        """
        Build a fresh SingleTaskGP on the current self.data_x/self.data_y, with
        input/outcome transforms so kernel hyperparameters (lengthscale,
        outputscale, noise) are learned in a normalized [0,1]^d / standardized
        Y=N(0,1) space regardless of the benchmark's raw domain/value scale.
        Without this, e.g. Eggholder's raw Y range (hundreds to low thousands)
        vs. a near-unit-scale default outputscale/noise initialization meant the
        MLL fit never converged to anything but a flat, uninformative posterior
        in 50 Adam steps -- confirmed by direct inspection (posterior mean/std
        were ~constant across 2000 random domain points).

        input_transform/outcome_transform are applied transparently by BoTorch
        for model.posterior(...) (and model(x) for the *input* side) -- callers
        elsewhere in this class keep passing/receiving raw-scale values with no
        changes needed, EXCEPT direct model(x) calls for MLL loss (must compare
        against model.train_targets, which is already-standardized) and direct
        model(x)+likelihood(...) sampling (must go through model.posterior(...,
        observation_noise=True) instead, since raw model(x) skips the outcome
        un-transform that .posterior() applies) -- both fixed at their call sites.
        """
        kernel_type = getattr(self.gp_config, 'kernel', 'rbf').lower()
        # noise_constraint default raised 1e-6->1e-2: matches
        # KennedyOHaganGP's own noise_lb fix (see module docstring above) --
        # light regularization against the near-zero-noise/near-interpolation
        # MLE attractor on sparse data, not expected to matter once data is
        # denser.
        noise_constraint_val = getattr(self.gp_config, 'noise_constraint', 1e-2)
        use_ard = getattr(self.gp_config, 'ard', False)
        input_dim = self.data_x.shape[-1]

        likelihood = GaussianLikelihood(noise_constraint=GreaterThan(noise_constraint_val))
        ard_num_dims = input_dim if use_ard else None
        if kernel_type == 'rbf':
            base_kernel = RBFKernel(ard_num_dims=ard_num_dims)
        elif kernel_type == 'matern':
            matern_nu = getattr(self.gp_config, 'matern_nu', 2.5)
            base_kernel = MaternKernel(nu=matern_nu, ard_num_dims=ard_num_dims)
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")
        # LogNormalPrior lengthscale regularization (module docstring above)
        # -- soft pull toward 0.5 without hard-blocking the optimizer from
        # going shorter/longer if the data genuinely supports it.
        base_kernel.register_prior(
            'lengthscale_prior',
            LogNormalPrior(loc=DRO_LENGTHSCALE_PRIOR_LOC, scale=DRO_LENGTHSCALE_PRIOR_SCALE),
            'lengthscale',
        )
        covar_module = ScaleKernel(base_kernel)
        covar_module.base_kernel.initialize(lengthscale=float(initial_lengthscale))

        train_x = self.data_x.to(device=self.device, dtype=self.dtype)
        train_y = self.data_y.to(device=self.device, dtype=self.dtype).squeeze(-1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=InputDataWarning)
            model = SingleTaskGP(
                train_X=train_x,
                train_Y=train_y.reshape(-1, 1),
                likelihood=likelihood,
                covar_module=covar_module,
                input_transform=Normalize(d=input_dim, bounds=self.bounds),
                outcome_transform=Standardize(m=1),
            )
            model = model.to(device=self.device, dtype=self.dtype)
        return model


    def _update_models(self):
        """Update all GP models in the ensemble with current data and retrain."""
        # Marks the start of a real BO iteration for iter_time logging (see
        # _update_data_and_best); harmless when called from _initialize_models
        # since no logging happens until _propose_next_candidate sets _pending_log.
        self._iter_t_start = time.perf_counter()
        _gp_refit_t_start = time.perf_counter()
        retrain= getattr(self.gp_config, 'retrain', False) # Default to True
        if retrain:
            # Warm-starting via set_train_data is architecturally incompatible with
            # this class's GP construction: _construct_gp_model's input/outcome
            # transforms are baked into the model at construction time (BoTorch
            # does not re-apply them on set_train_data), and _initialize_models
            # always populates fresh ensemble slots with gp['model']=None before
            # the first _update_models() call -- so this path would either crash
            # immediately (AttributeError on the None model) or, if patched to
            # tolerate that, silently skip the transform re-application and train
            # on the wrong scale. Rebuilding from scratch (the `else` branch) is
            # the only correct approach given how this class manages transforms;
            # raise clearly here rather than let either failure mode happen silently.
            raise NotImplementedError(
                "gp.retrain=True is not supported -- see comment above _update_models's "
                "retrain branch. Use the default retrain=False (rebuilds each ensemble "
                "member from scratch every call, which is required for correct "
                "input/outcome transform handling)."
            )
        else:
            if not self.gp_ensemble:
                print("Warning: GP ensemble not initialized. Cannot update.")
                # Try to initialize if data exists? Or rely on _initialize_models being called first.
                if self.data_x.shape[0] > 0:
                    print("Attempting to initialize GP ensemble now...")
                    self._initialize_models() # This might recursive call _update_models, be careful
                    if not self.gp_ensemble: # If still failed
                        raise RuntimeError("Failed to initialize GP models.")
                else:
                    self._last_gp_refit_time = time.perf_counter() - _gp_refit_t_start
                    return # Cannot update without data

        if self.verbose: print(f"Updating {len(self.gp_ensemble)} GP models...")

        # Each ensemble member is rebuilt from scratch every call (rather than
        # warm-started via set_train_data on a persisting model), for two
        # reasons: (1) it's the only correct way to apply input/outcome
        # transforms via _construct_gp_model, since BoTorch bakes the outcome
        # transform into train_targets at construction time rather than
        # re-applying it on set_train_data; (2) it keeps every ensemble member
        # anchored to its own designated initial_lengthscale each iteration,
        # preserving the ensemble's intended hyperparameter diversity rather
        # than letting warm-started members gradually converge toward each
        # other over many iterations.
        for gp_dict in self.gp_ensemble:
            model = self._construct_gp_model(gp_dict['initial_lengthscale'])

            # Train the model
            model.train()
            model.likelihood.train()

            # Use Exact MLL for optimization
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

            # Use Adam or LBFGS for optimization (Adam often more robust)
            gp_optimizer = torch.optim.Adam(model.parameters(), lr=getattr(self.gp_config,'lr', 0.1)) # Configurable LR
            max_iter = getattr(self.gp_config, 'train_iter', 50) # Configurable iterations

            train_x = self.data_x.to(device=self.device, dtype=self.dtype)
            for i in range(max_iter):
                gp_optimizer.zero_grad()
                output = model(train_x)
                # model.train_targets is already standardized (outcome_transform
                # applied at construction); comparing against raw self.data_y here
                # would be the same mismatched-scale bug this method now fixes.
                loss = -mll(output, model.train_targets)
                loss.backward()
                gp_optimizer.step()

            # Set to eval mode
            model.eval()
            model.likelihood.eval()

            gp_dict['model'] = model
            gp_dict['likelihood'] = model.likelihood
        self._last_gp_refit_time = time.perf_counter() - _gp_refit_t_start
        if self.verbose: print("GP models updated and trained.")


    # --- Acquisition Function Logic (Internal Helper) ---

    def _acquisition_function_value_botorch(
            self, x: torch.Tensor, gp_idx: int, observed_best: float, acq_name: str = None,
            candidate_set: torch.Tensor = None,
        ) -> torch.Tensor:
        """
        Compute acquisition function value using BoTorch for a given point x.

        Args:
            x: A single point tensor [D] or [1, D] representing the candidate point.
            gp_idx: The index of the Gaussian Process model in the ensemble to use.
            observed_best: The best objective function value observed so far.
            candidate_set: Required when acq_name=="mes" (or "rotate_acq" happens to
                draw "mes"). qMaxValueEntropy uses this to Gumbel-sample the global
                maximum y* -- it must be a domain-covering set of candidates, NOT x
                itself. x is often a single point [1, D] (e.g. the call site scoring
                best_x_overall), which gives qMaxValueEntropy no spread at all to
                estimate y* meaningfully: with a single-point candidate_set, the
                Gumbel CDF-of-max degenerates to the CDF of one Gaussian, at that one
                point. Callers should pass _optimize_acquisition's roi_candidates,
                which is drawn from a uniform, domain-wide candidate pool.

        Returns:
            A tensor containing the acquisition function value for the point x.
        """
        if not hasattr(self, 'gp_ensemble') or not self.gp_ensemble:
            raise RuntimeError("GP Ensemble (`self.gp_ensemble`) not initialized.")
        if gp_idx >= len(self.gp_ensemble):
            raise IndexError(
                f"gp_idx {gp_idx} out of bounds for ensemble size {len(self.gp_ensemble)}"
            )
        if not hasattr(self, 'acquisition_config'):
             raise RuntimeError("Acquisition config (`self.acquisition_config`) not found.")

        # --- 1. Prepare Input and Model ---
        x = x.to(device=self.device, dtype=self.dtype) # Ensure correct device/dtype
        if x.ndim == 1:
            x = x.unsqueeze(0) # Shape: [1, D]

        # BoTorch analytic acquisition functions expect input shape (b x q x d) or (q x d).
        # We have a single point (q=1), so reshape to [1, 1, D]
        # where the first dimension is the batch dimension ('b') for the acquisition function.
        if x.shape[0] == 1:
            X = x.unsqueeze(0) # Shape: [1, 1, D]
        else:
            X = x.reshape(x.shape[0], 1, x.shape[1]) # Shape: [b, 1, D]

        # Get the specific GP model from the ensemble
        model = self.gp_ensemble[gp_idx]['model']

        # Ensure the model is in evaluation mode
        model.eval()
        # No need to explicitly call likelihood.eval() here, BoTorch handles it via the model.

        # --- 2. Get Acquisition Function Configuration ---
        # Default to 'ei' if not specified
        if acq_name is None:
            acq_name = getattr(self.acquisition_config, 'function', 'ei').lower()
        # Assuming maximization objective, change if minimizing
        maximize_objective = getattr(self.acquisition_config, 'maximize', True)

        # --- 3. Instantiate BoTorch Acquisition Function ---
        if acq_name == "ei":
            # Note: BoTorch EI's 'best_f' corresponds to 'observed_best'.
            # The 'xi' parameter from the original code often acts as a trade-off.
            # In BoTorch, this can sometimes be incorporated by adjusting best_f.
            xi = getattr(self.acquisition_config, 'xi', 0.01)
            effective_best_f = observed_best + xi if maximize_objective else observed_best - xi
            acq_func = LogExpectedImprovement(
                model=model,
                best_f=effective_best_f,
                maximize=maximize_objective,
            )
        elif acq_name == "ucb":
            kappa = getattr(self.acquisition_config, 'kappa', 2.5)
            # BoTorch UCB uses 'beta' which often corresponds to 'kappa' or kappa^2
            # depending on the literature. Check BoTorch docs for the exact definition used.
            # As of recent versions, beta is typically the direct trade-off parameter (like kappa).
            acq_func = UpperConfidenceBound(
                model=model,
                beta=kappa, 
                maximize=maximize_objective,
                # objective=None
            )
        elif acq_name == "pi":
            xi = getattr(self.acquisition_config, 'xi', 0.01)
            effective_best_f = observed_best + xi if maximize_objective else observed_best - xi
            acq_func = ProbabilityOfImprovement(
                model=model,
                best_f=effective_best_f,
                maximize=maximize_objective,
                # objective=None
            )
        # --- Max-value Entropy Search (MES) ---
        elif acq_name == "mes":
            if candidate_set is None:
                raise ValueError(
                    "MES acquisition requires a domain-covering candidate_set "
                    "(e.g. _optimize_acquisition's roi_candidates) -- x itself "
                    "must not be reused for this, see docstring."
                )

            # Get necessary MES parameters from config
            num_mv_samples = getattr(self.acquisition_config, 'mes_num_mv_samples', 10)

            acq_func = qMaxValueEntropy(
                model=model,
                candidate_set=candidate_set,
                num_mv_samples=num_mv_samples,
                maximize=maximize_objective,
                # use_posterior_mean=False, # Optional optimization
            )

        # --- GIBBON Placeholder ---
        elif acq_name == "gibbon":
            raise NotImplementedError(
                "GIBBON acquisition function requires custom implementation based on "
                "the specific algorithm/paper and is not a standard BoTorch class."
            )
        elif acq_name == "rotate_acq":
            # randomly rotate the acquisition function. "mes" deliberately
            # excluded: MES is used only as a *reward* signal (use_mes_reward),
            # not as a rollout acquisition function -- the acq_name="mes" branch
            # above is kept correct/callable (e.g. for direct testing) but is no
            # longer one of rotate_acq's automatic choices.
            acq_names = ["ei", "ucb", "pi"]
            acq_name = np.random.choice(acq_names)
            return self._acquisition_function_value_botorch(
                x, gp_idx, observed_best, acq_name=acq_name, candidate_set=candidate_set,
            )
        else:
            raise ValueError(f"Unknown acquisition function: {acq_name}")

        # --- 4. Compute Acquisition Value ---
        acq_value = acq_func(X) # Input shape [1, 1, D] -> Output shape [1]

        # --- 5. Return Result ---
        # Return the tensor containing the value (original function returned a tensor)
        return acq_value # Shape [1]

    # --- Helper Function ---
    def _get_posterior_mean_stddev(self, x: torch.Tensor, gp_idx: int):
        """
        Helper function to get posterior mean and standard deviation for input x.

        Args:
            x (torch.Tensor): Input points (shape [b, D] or [D]).
            gp_idx (int): Index of the GP model in the ensemble.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Posterior mean and standard deviation.
        """
        if not self.gp_ensemble:
            raise RuntimeError("GP Ensemble not initialized.")
        if gp_idx >= len(self.gp_ensemble):
             raise IndexError(f"gp_idx {gp_idx} out of bounds for ensemble size {len(self.gp_ensemble)}")

        x = x.to(self.device, self.dtype) # Ensure correct device/dtype
        if x.ndim == 1: x = x.unsqueeze(0) # Ensure batch dimension [1, D]

        gp = self.gp_ensemble[gp_idx]
        model = gp['model']
        likelihood = gp['likelihood'] # Included for completeness, though BoTorch model handles it

        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Use model.posterior(x) for BoTorch models - gives GPyTorch posterior object
            posterior = model.posterior(x)
            mean = posterior.mean.squeeze(-1) # Squeeze trailing dimension [b, 1] -> [b]
            variance = posterior.variance.squeeze(-1) # Squeeze trailing dimension [b, 1] -> [b]
            # Clamp variance for numerical stability before sqrt
            stddev = variance.clamp_min(1e-9).sqrt()

        return mean, stddev

    def _make_fantasy_model(self, model, x_new: torch.Tensor, y_new_raw: torch.Tensor):
        """
        Correctly condition a SingleTaskGP (built by _construct_gp_model, which
        uses input_transform=Normalize + outcome_transform=Standardize) on one
        new observation, by rebuilding a fresh GP on the augmented raw data and
        reusing the original's frozen kernel/likelihood hyperparameters (no MLL
        refitting) -- rather than using condition_on_observations/
        get_fantasy_model, which are broken for this model configuration.

        ROOT CAUSE (two independent bugs found and ruled out en route to this):
        1. Y-scale mismatch was NOT the issue -- BoTorch's condition_on_observations
           already applies outcome_transform to Y internally (confirmed by reading
           botorch/models/gpytorch.py's BatchedMultiOutputGPyTorchModel.
           condition_on_observations, which calls self.outcome_transform(Y, noise)
           before conditioning).
        2. The real issue: GPyTorch's ExactGP.get_fantasy_model builds the
           augmented (train + fantasy) joint distribution via
           `super(ExactGP, self).__call__(*full_inputs)`, which calls
           SingleTaskGP.forward(x) DIRECTLY -- and forward() only applies
           input_transform when self.training is True. Every caller here has
           the model in .eval() mode when conditioning, so this computation
           uses RAW domain-scale coordinates against ARD lengthscales fit for
           NORMALIZED [0,1]^d space -- cross-covariance underflows to exactly
           0.0, so condition_on_observations's mean shift is bit-identical to
           pre-conditioning regardless of Y (confirmed empirically).
           Temporarily flipping model.training=True (bypassing model.train(),
           which would wipe the prediction_strategy cache get_fantasy_model
           requires) does make forward() apply the transform -- but produces
           a DIFFERENT wrong answer (confirmed by comparing against a from-
           scratch ground-truth GP built on the augmented data: wrong sign in
           some trials, e.g. y_new above the current mean yet the "fixed"
           posterior mean at that point moving further away from it). The
           interaction between GPyTorch's fantasy-update caching and BoTorch's
           InputTransform.forward's own transform_on_train/eval/fantasize
           flags (gated on the *transform's own* .training, not the parent
           model's) is more tangled than a single attribute flip resolves --
           not worth reverse-engineering further given a simple, verifiably
           correct alternative exists.

        THIS METHOD instead rebuilds a plain SingleTaskGP from scratch each
        call: recovers the model's current RAW-scale train_X/train_Y (via
        untransform, so this composes correctly across repeated/chained
        calls), appends (x_new, y_new_raw), and constructs a new model with
        the SAME frozen covar_module/likelihood.noise as the input model (no
        re-optimization) plus fresh Normalize/Standardize transforms fit on
        the augmented data. This mirrors how _update_models already rebuilds
        every ensemble member from scratch each real BO iteration (never
        warm-starting) -- see that method's docstring for why that's the
        correct approach given how transforms are handled in this class.

        Verified bit-identical (to 1e-6) against a from-scratch ground-truth
        GP built independently on the same augmented data, across 5 trials
        with varying Y, including cases where condition_on_observations-based
        fixes produced wrong-signed shifts.

        x_new:     [1, d] input location (raw domain scale)
        y_new_raw: scalar or [1, 1] observed value (raw objective scale)

        Returns: fresh fantasy-conditioned model with correctly updated
        posterior mean + variance, in eval mode.
        """
        y_new_raw = y_new_raw.reshape(-1).to(model.train_targets.device, model.train_targets.dtype)

        with torch.no_grad():
            raw_train_x = model.input_transform.untransform(model.train_inputs[0])
            raw_train_y = model.outcome_transform.untransform(model.train_targets.unsqueeze(-1))[0].reshape(-1)

        aug_x = torch.cat([raw_train_x, x_new.reshape(1, -1)], dim=0)
        aug_y = torch.cat([raw_train_y, y_new_raw], dim=0)

        input_dim = aug_x.shape[-1]
        new_likelihood = GaussianLikelihood(
            noise_constraint=GreaterThan(getattr(self.gp_config, 'noise_constraint', 1e-6))
        )
        new_likelihood.noise = model.likelihood.noise.detach().clone()
        new_covar_module = ScaleKernel(RBFKernel(ard_num_dims=input_dim) if isinstance(model.covar_module.base_kernel, RBFKernel)
                                        else MaternKernel(nu=model.covar_module.base_kernel.nu, ard_num_dims=input_dim))
        new_covar_module.load_state_dict(model.covar_module.state_dict())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=InputDataWarning)
            fantasy_model = SingleTaskGP(
                train_X=aug_x, train_Y=aug_y.reshape(-1, 1),
                likelihood=new_likelihood, covar_module=new_covar_module,
                input_transform=Normalize(d=input_dim, bounds=self.bounds),
                outcome_transform=Standardize(m=1),
            )
            fantasy_model = fantasy_model.to(device=self.device, dtype=self.dtype)
        fantasy_model.eval()
        return fantasy_model

    # --- Optimize Acquisition Function (Internal Helper for Simulation) ---
    def _optimize_acquisition(self, gp_idx: int, observed_best: float, for_rollout: bool = False,
                               acq_name_override: str = None) -> tuple:
        """
        Optimize acquisition function for a specific GP using random search + refinement,
        with an option to constrain the search space based on UCB >= max(LCB).

        acq_name_override: when given, used instead of self.rollout_acq_function for
        this call only (self.rollout_acq_function itself is left untouched). Lets a
        caller resolve a per-call acquisition function without needing every other
        _optimize_acquisition call site to also handle it -- e.g. rollout_acq_function
        == "rotate" (see _simulate_trajectory), which cycles ["ei","ucb","pi","mes"]
        by rollout step. Only _simulate_trajectory currently resolves "rotate" into a
        concrete acq_name; other callers passing "rotate" straight through via
        self.rollout_acq_function (no override) would hit
        _acquisition_function_value_botorch's "Unknown acquisition function" error --
        not exercised by any existing caller, since "rotate" is rollout-only by design.

        Returns: (best_x [1, D], roi_candidates [N_roi, D]) where roi_candidates
        is the UCB>=max(LCB) filtered candidate set (or the full broad-search
        set if that constraint is disabled). The broad-search candidates
        (x_samples) are always uniform over the entire domain regardless of
        constrain_ucb_lcb, so roi_candidates itself has non-vanishing coverage
        everywhere the UCB/LCB filter admits -- including narrow regions far
        from the current best -- and is safe to reuse directly for MES's
        Gumbel y* sampling (no separate gumbel_candidates set needed).

        for_rollout: this method is called ONLY from within simulated rollouts
        (_simulate_trajectory / _simulate_trajectory_joint) -- the real
        candidate returned to the caller each BO iteration comes directly
        from the trained Decision Transformer's forward pass, never from this
        method. Profiling (profile_dro_iteration.py) found this method's
        refinement loop (not the broad search) dominates its cost, and it is
        called ~300x per real BO iteration (once per rollout step), making it
        ~38% of total wall-clock. Since its only consumer is simulated
        rollout data used to train the DT (not a real-time decision), a
        cheaper budget here trades rollout fidelity for speed -- set
        for_rollout=True to use the separate rollout_opt_samples/
        rollout_opt_restarts/rollout_refinement_samples config keys (each
        falls back to the standard key's value if unset, so this is a no-op
        unless those keys are explicitly configured).
        """
        effective_acq_name = acq_name_override if acq_name_override is not None else self.rollout_acq_function
        # --- Configurable Parameters ---
        if for_rollout:
            num_samples = getattr(self.acquisition_config, 'rollout_opt_samples',
                                   getattr(self.acquisition_config, 'opt_samples', 1000))
            num_restarts = getattr(self.acquisition_config, 'rollout_opt_restarts',
                                    getattr(self.acquisition_config, 'opt_restarts', 10))
            refinement_samples = getattr(self.acquisition_config, 'rollout_refinement_samples',
                                          getattr(self.acquisition_config, 'refinement_samples', 100))
        else:
            num_samples = getattr(self.acquisition_config, 'opt_samples', 1000)
            num_restarts = getattr(self.acquisition_config, 'opt_restarts', 10)
            refinement_samples = getattr(self.acquisition_config, 'refinement_samples', 100)
        refinement_noise = getattr(self.acquisition_config, 'refinement_noise', 0.05)
        # --- Constraint Parameters ---
        constrain_ucb_lcb = getattr(self.acquisition_config, 'constrain_ucb_lcb', True)
        # Use kappa from UCB acquisition if available, otherwise default (or add specific config)
        kappa_constraint = getattr(self.acquisition_config, 'ucb_lcb_kappa',
                                   getattr(self.acquisition_config, 'kappa', 2.5))


        dim = self.bo_config.input_dim
        domain_min, domain_max = self.bounds[0], self.bounds[1]
        best_x_overall = None # Keep track of the best valid point found

        # Ensure observed_best is a scalar float or tensor for BoTorch acq functions
        if isinstance(observed_best, torch.Tensor) and observed_best.numel() > 1:
             raise ValueError("observed_best must be a scalar or scalar tensor for acquisition optimization.")

        # --- 1. Broad Random Search ---
        with torch.no_grad():
            # Candidates are uniform over the entire domain regardless of
            # constrain_ucb_lcb. The UCB >= max(LCB) filter below is what
            # defines the ROI and does all the focusing work -- candidate
            # *generation* no longer tries to pre-focus around the current
            # best (that was the local-Gaussian/hybrid approach, which could
            # still collapse or dilute coverage of narrow good regions; see
            # Section "Structural Diagnosis" in the report for why).
            x_samples = domain_min + (domain_max - domain_min) * torch.rand(
                num_samples, dim, device=self.device, dtype=self.dtype
            )

            max_lcb = -float('inf') # Initialize max LCB

            # --- Apply UCB >= max(LCB) constraint IF enabled ---
            if constrain_ucb_lcb:
                # Calculate LCB and UCB for all random samples
                mean_samples, stddev_samples = self._get_posterior_mean_stddev(x_samples, gp_idx)
                lcb_samples = mean_samples - kappa_constraint * stddev_samples
                ucb_samples = mean_samples + kappa_constraint * stddev_samples

                # Find the maximum LCB value among the samples
                max_lcb = torch.max(lcb_samples).item()

                # Create a mask for valid points (UCB >= max_LCB)
                valid_mask = (ucb_samples >= max_lcb)

                if self.verbose and valid_mask.sum() == 0:
                    print(f"Warning: No points satisfied UCB >= max(LCB) constraint (max LCB: {max_lcb:.4f}). Optimization might be difficult.")
                elif self.verbose == 2:
                    #  pass
                     print(f"Constraint Info GP ({gp_idx}): max(LCB)={max_lcb:.4f}. Found {valid_mask.sum()}/{num_samples} valid points in broad search.")

            # ROI candidates: the UCB>=max(LCB) filtered region already computed above
            # (or the full broad-search set if the constraint is disabled). Exposed so
            # callers (MES reward) can reuse it instead of recomputing a candidate set.
            if constrain_ucb_lcb and valid_mask.sum() > 0:
                roi_candidates = x_samples[valid_mask]
            else:
                roi_candidates = x_samples

            # Evaluate the PRIMARY acquisition function. candidate_set is only
            # consulted by the "mes" branch (incl. when "rotate_acq" happens
            # to draw "mes"). roi_candidates is now itself drawn from the uniform,
            # domain-wide x_samples (filtered by UCB>=max(LCB)), so it no
            # longer needs a separate gumbel_candidates set for this purpose.
            # acq_name=self.rollout_acq_function: _optimize_acquisition is called
            # only from within simulated rollouts, so query selection here should
            # use the per-variant rollout acquisition function, not whatever
            # acquisition_config.function happens to default to.
            acq_values = self._acquisition_function_value_botorch(
                x_samples, gp_idx, observed_best, acq_name=effective_acq_name, candidate_set=roi_candidates
            )

            # If constraint is active, filter out invalid points by setting acq value low
            if constrain_ucb_lcb:
                # Use a very small number instead of -inf to avoid issues with argmax if all are invalid
                acq_values[~valid_mask] = -1e20 # Or -float('inf') if argmax handles it

            # Check if any valid points were found
            if constrain_ucb_lcb and valid_mask.sum() == 0:
                 # If no points are valid, we can't really optimize.
                 # Let's pick the point with the highest (invalid) UCB.
                 best_fallback_idx = torch.argmax(ucb_samples)
                 best_x_overall = x_samples[best_fallback_idx].clone()
                 print(f"WARN: No valid points found for constraint UCB >= max(LCB). Falling back to point with highest UCB: {best_x_overall}")
                 # Skip refinement if no valid points
                 num_restarts = 0
            else:
                 # Get top candidates based on acquisition values
                 # Ensure we don't request more restarts than valid points found
                 actual_num_restarts = min(num_restarts, num_samples if not constrain_ucb_lcb else valid_mask.sum().item())
                 if actual_num_restarts <=0 and num_restarts > 0:
                      # This case should be covered above, but as a safeguard
                      print("WARN: num_restarts > 0 but no valid points found. Setting restarts to 0.")
                      actual_num_restarts = 0

                 if actual_num_restarts > 0:
                    top_indices = torch.argsort(acq_values, descending=True)[:actual_num_restarts]
                    top_points = x_samples[top_indices]
                    # Initialize overall best with the best from broad search (that satisfies constraint if active)
                    best_x_overall = top_points[0].clone()
                 else:
                      # Handle case where restarts is 0 or no valid points for restarts
                      # Find the single best point according to acq_values (might be invalid if constraint active and all fail)
                      best_idx = torch.argmax(acq_values)
                      best_x_overall = x_samples[best_idx].clone()
                      # Set top_points empty to skip refinement loop
                      top_points = torch.empty((0, dim), device=self.device, dtype=self.dtype)
                      if constrain_ucb_lcb and not valid_mask[best_idx]:
                           print(f"WARN: Best point from broad search ({best_x_overall}) doesn't satisfy constraint, but was chosen as best overall.")


        # --- 2. Local Refinement around top candidates (if any) ---
        best_acq_value_refined = -float('inf') # Track best acq value found during refinement

        # Only refine if we have valid starting points
        if best_x_overall is not None and top_points.shape[0] > 0 :
             # Reuse the value already computed for this point during the broad
             # search (acq_values[top_indices[0]], since best_x_overall ==
             # top_points[0] == x_samples[top_indices[0]]) instead of a fresh
             # _acquisition_function_value_botorch call -- that call was a pure
             # redundant recomputation for deterministic acquisitions (EI/UCB/PI),
             # and for MES specifically it was worse than redundant: qMaxValueEntropy
             # re-samples its Gumbel y* draws on every instantiation, so the fresh
             # call could return a genuinely different value than the one that was
             # actually used to rank/select this point as the top broad-search
             # candidate in the first place. Reusing the stored value is both
             # cheaper and more internally consistent.
             initial_best_acq_val = acq_values[top_indices[0]].item()
             best_acq_value_refined = initial_best_acq_val

             with torch.no_grad():
                 for start_point in top_points:
                     # Generate local perturbations
                     noise = refinement_noise * (domain_max - domain_min) * (
                         torch.rand(refinement_samples, dim, device=self.device, dtype=self.dtype) * 2 - 1
                     )
                     perturbed_points = start_point.unsqueeze(0) + noise
                     perturbed_points = torch.min(torch.max(perturbed_points, domain_min), domain_max) # Clip

                     # --- Apply constraint check to refinement samples IF enabled ---
                     valid_refinement_mask = torch.ones(refinement_samples, dtype=torch.bool, device=self.device) # Default to all valid
                     if constrain_ucb_lcb:
                         mean_perturbed, stddev_perturbed = self._get_posterior_mean_stddev(perturbed_points, gp_idx)
                         ucb_perturbed = mean_perturbed + kappa_constraint * stddev_perturbed
                         valid_refinement_mask = (ucb_perturbed >= max_lcb)

                     # Evaluate primary acquisition function for perturbed points
                     local_acq_values = self._acquisition_function_value_botorch(
                         perturbed_points, gp_idx, observed_best, acq_name=effective_acq_name, candidate_set=roi_candidates
                         ) # Batch evaluation

                     # Filter out invalid refinement points
                     if constrain_ucb_lcb:
                         local_acq_values[~valid_refinement_mask] = -1e20 # Or -float('inf')

                     # Find best in this local search (among potentially filtered points)
                     if valid_refinement_mask.sum() > 0 or not constrain_ucb_lcb:
                         best_local_idx = torch.argmax(local_acq_values)
                         local_best_acq = local_acq_values[best_local_idx].item()

                         # Update overall best if improved AND the point is valid (or constraint inactive)
                         current_best_is_valid = True # Assume valid if constraint inactive
                         if constrain_ucb_lcb:
                              current_best_is_valid = valid_refinement_mask[best_local_idx].item()

                         if current_best_is_valid and local_best_acq > best_acq_value_refined:
                             best_acq_value_refined = local_best_acq
                             best_x_overall = perturbed_points[best_local_idx].clone()

        # --- Return the best point found ---
        if best_x_overall is None:
             # This should only happen if opt_samples is 0. Fallback: return a random point.
             print("Error: Optimization failed to find any candidate point. Returning random.")
             best_x_overall = domain_min + (domain_max - domain_min) * torch.rand(
                 1, dim, device=self.device, dtype=self.dtype
             )
             return best_x_overall, roi_candidates # [1, D], [N_roi, D]
        else:
            # Ensure shape [1, D]
            best_x_out = best_x_overall.unsqueeze(0) if best_x_overall.ndim == 1 else best_x_overall
            return best_x_out, roi_candidates

    def _optimize_acquisition_softmax(self, gp_idx: int, observed_best, acq_name: str = None) -> tuple:
        """
        Stochastic "softmax teacher" alternative to _optimize_acquisition,
        for _simulate_trajectory's rollout action selection only (never used
        for the real candidate, which always comes from the trained DT).
        Gated behind self.rollout_teacher == "softmax" (default "argmax" ->
        _optimize_acquisition, unchanged).

        Same broad random search (uniform over the full domain, num_samples
        = rollout_opt_samples/opt_samples) and the same
        _acquisition_function_value_botorch scoring as
        _optimize_acquisition's Step 1 -- but SAMPLES a candidate
        proportional to softmax(scores / self.softmax_temperature) instead
        of taking the argmax + local refinement. No refinement step: a
        hill-climbing local search doesn't fit a stochastic-sampling design
        (it would just re-collapse the sample back toward a single mode).

        Tests whether stochastic exploration alone (independent of reward
        scheme) produces more diverse, less mutually-coherent rollout
        gradients than the current deterministic argmax teacher -- see the
        cos_sim trajectory diagnostic this is built to feed.

        Returns: (chosen_x [1, D], candidate_pool [N, D]) -- same shape
        contract as _optimize_acquisition, so _simulate_trajectory's call
        site doesn't need to branch on return type.
        """
        effective_acq_name = acq_name if acq_name is not None else self.rollout_acq_function
        num_samples = getattr(self.acquisition_config, 'rollout_opt_samples',
                               getattr(self.acquisition_config, 'opt_samples', 1000))
        dim = self.bo_config.input_dim
        domain_min, domain_max = self.bounds[0], self.bounds[1]

        if isinstance(observed_best, torch.Tensor):
            observed_best_t = observed_best
        else:
            observed_best_t = torch.tensor(observed_best, device=self.device, dtype=self.dtype)

        with torch.no_grad():
            x_samples = domain_min + (domain_max - domain_min) * torch.rand(
                num_samples, dim, device=self.device, dtype=self.dtype
            )
            scores = self._acquisition_function_value_botorch(
                x_samples, gp_idx, observed_best_t, acq_name=effective_acq_name,
                candidate_set=x_samples,
            ).reshape(-1)
            probs = torch.nn.functional.softmax(scores / self.softmax_temperature, dim=0)
            chosen = torch.multinomial(probs, 1).item()

        return x_samples[chosen:chosen + 1], x_samples

    # --- State Extraction (Internal Helper) ---
    def _extract_state(self, current_data_x, current_data_y, current_step, roi_candidates=None) -> torch.Tensor:
        """
        Extract state representation for the transformer.

        roi_candidates: optional [N_roi, d] domain-spanning candidate set,
        needed whenever self.use_roi_state, self.use_roi_std_quantiles, or
        self.use_roi_sigma_iqr is True. All three flags reuse the SAME
        model.posterior(roi_candidates) call per ensemble member (no
        duplicate GP calls) to build three independent, optional feature
        blocks:
          - use_roi_state: appends [mean(mu_roi), mean(sigma_roi)] per member
            -- GP posterior mean/std averaged over roi_candidates (Track 5's
            original fix: the base state otherwise has zero location-sensitive
            GP information).
          - use_roi_std_quantiles: appends [min, p25, median, p75, max] of the
            per-candidate posterior std over roi_candidates, per member -- a
            distributional view of uncertainty instead of a single mean.
          - use_roi_sigma_iqr: appends [mean(sigma_roi), IQR(sigma_roi)] per
            member -- pairs mean_sigma_roi with the interquartile range of
            the per-candidate posterior std instead of mean_mu_roi, so the
            DT can distinguish uniform uncertainty from saturated-region
            uncertainty.
        When a flag is True but roi_candidates is None (some call sites
        don't have one available -- see _propose_next_candidate's per-rollout
        initial_state calls), that flag's slots are zero-padded instead of
        computed, preserving a well-defined, fixed-length state.

        self.state_hyperparams_enabled (default True) gates slots 0-9 (raw GP
        kernel hyperparameters) -- set False to ablate them out of the state
        entirely (a hypothesis test: are they dead weight, or carrying real
        signal). Default True preserves the original, always-present behavior.
        """
        # 1. GP Hyperparameters (only when state_hyperparams_enabled=True)
        gp_params_flat = []
        if not getattr(self, 'state_hyperparams_enabled', True):
            pass # Ablated out entirely -- contributes 0 dims.
        elif not self.gp_ensemble:
             num_models = self.gp_config.num_models
             gp_params_flat = [1.0] * (num_models * self.GP_STATE_PARAMS_PER_MODEL) # Default params
             print("Warning: Extracting state before GP ensemble is fully initialized. Using default params.")
        else:
            for gp_dict in self.gp_ensemble:
                model = gp_dict['model']
                # Extract LEARNED hyperparameters (needs care depending on GPyTorch version/model)
                params = []
                for param_name, param, constraint in model.named_parameters_and_constraints():
                     # Example: Extract common params like lengthscale, outputscale
                     if 'lengthscale' in param_name:
                          # With ARD (gp.ard=True), lengthscale has one value per input
                          # dimension rather than a single scalar; .mean() collapses it
                          # to one summary value for the fixed 2-slots-per-model state
                          # budget (GP_STATE_PARAMS_PER_MODEL). For a true scalar
                          # (non-ARD) lengthscale this is identical to param.item().
                          params.append(param.mean().item())
                     elif 'outputscale' in param_name:
                          params.append(param.mean().item())
                     # Add more param extraction logic as needed based on your GP model
                
                # Simple fallback if specific params aren't found easily: just grab all params
                if not params:
                     params = [p.item() for p in model.parameters()] # Less interpretable

                # Pad or truncate if param count varies unexpectedly? For now, assume fixed count.
                gp_param_dim_assumed = self.GP_STATE_PARAMS_PER_MODEL
                if len(params) < gp_param_dim_assumed:
                     params.extend([1.0] * (gp_param_dim_assumed - len(params))) # Pad with defaults
                elif len(params) > gp_param_dim_assumed:
                     params = params[:gp_param_dim_assumed] # Truncate

                gp_params_flat.extend(params)


        # 2. Best Observed Value so far
        if current_data_y.shape[0] == 0:
             best_value = 0.0 # Or some other default
        else:
             best_value = current_data_y.max().item() if self.objective_mode == "maximize" else current_data_y.min().item()

        # 3. Current Iteration/Step Number
        step_norm = float(current_step) / self.bo_config.max_iterations # Normalize step? Optional.

        # 4. Current Best Position (if available)
        best_position = torch.zeros(self.bo_config.input_dim, device=self.device, dtype=self.dtype)
        if current_data_x.shape[0] > 0:
            best_position = current_data_x[current_data_y.argmax() if self.objective_mode == "maximize" else current_data_y.argmin()]

        # 5. ROI-based GP posterior statistics (use_roi_state, use_roi_std_quantiles,
        # and/or use_roi_sigma_iqr). All three share ONE model.posterior(roi_candidates)
        # call per ensemble member -- no duplicate GP calls when more than one is on.
        #
        # use_roi_state: [mean(mu_roi), mean(sigma_roi)] per member, normalized
        # onto a comparable scale to the rest of the state vector: mean_mu_roi
        # is divided by the run's first-ever |best_value| (frozen and cached
        # in self._initial_mu_denom -- NOT re-derived from the current
        # best_value each call, which would be non-stationary and can approach
        # zero on benchmarks where best_value starts near 0, e.g. Rosenbrock);
        # mean_sigma_roi is divided by that ensemble member's OWN first-ever
        # sigma_roi (a per-model reference "prior std" scale, captured once
        # and cached in self._initial_sigma_roi_per_model). Both denominators
        # are floored away from zero to avoid inf/NaN.
        #
        # use_roi_std_quantiles: [min, p25, median, p75, max] of the
        # per-candidate posterior std over roi_candidates, per member -- raw
        # (un-normalized), same units as sigma_roi itself. NOT normalized like
        # use_roi_state's mean_sigma_roi is -- if these turn out to sit on an
        # inconsistent scale relative to the rest of the state (the same class
        # of issue Bug 3 found), that would need the same per-model reference-
        # scale treatment, not yet applied here since it wasn't specified.
        #
        # use_roi_sigma_iqr: [mean(sigma_roi), IQR(sigma_roi)] per member, both
        # normalized by the same per-model sigma_denom used above (same units,
        # same reference scale). IQR = p75(sigma_roi) - p25(sigma_roi), reusing
        # the quantile computation directly.
        #
        # The sigma_denom reference-scale cache (self._initial_sigma_roi_per_model)
        # is captured whenever ANY of these three flags needs mean_sigma_roi, not
        # just use_roi_state, so use_roi_sigma_iqr gets a consistent normalization
        # even when use_roi_state is off.
        roi_state_flat = []
        roi_std_quantile_flat = []
        roi_sigma_iqr_flat = []
        want_roi_state = getattr(self, 'use_roi_state', False)
        want_roi_std_quantiles = getattr(self, 'use_roi_std_quantiles', False)
        want_roi_sigma_iqr = getattr(self, 'use_roi_sigma_iqr', False)
        if want_roi_state or want_roi_std_quantiles or want_roi_sigma_iqr:
            num_models = self.gp_config.num_models
            if roi_candidates is not None and self.gp_ensemble:
                if self._initial_mu_denom is None:
                    self._initial_mu_denom = abs(best_value) + 1e-6
                mu_denom = self._initial_mu_denom
                if self._initial_sigma_roi_per_model is None:
                    self._initial_sigma_roi_per_model = [None] * len(self.gp_ensemble)
                quantile_levels = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=self.device, dtype=self.dtype)
                for m, gp_dict in enumerate(self.gp_ensemble):
                    model = gp_dict['model']
                    model.eval()
                    with torch.no_grad():
                        posterior = model.posterior(roi_candidates)
                        mu_roi = posterior.mean.reshape(-1)
                        sigma_roi = posterior.variance.clamp_min(1e-12).sqrt().reshape(-1)

                    mean_sigma_roi = sigma_roi.mean().item()
                    if self._initial_sigma_roi_per_model[m] is None:
                        self._initial_sigma_roi_per_model[m] = mean_sigma_roi
                    sigma_denom = self._initial_sigma_roi_per_model[m] + 1e-6

                    if want_roi_state:
                        mean_mu_roi = mu_roi.mean().item()
                        roi_state_flat.append(mean_mu_roi / mu_denom)
                        roi_state_flat.append(mean_sigma_roi / sigma_denom)

                    if want_roi_std_quantiles or want_roi_sigma_iqr:
                        quantiles = torch.quantile(sigma_roi, quantile_levels)
                        if want_roi_std_quantiles:
                            roi_std_quantile_flat.extend(quantiles.tolist())
                        if want_roi_sigma_iqr:
                            iqr = (quantiles[3] - quantiles[1]).item()
                            # Level (mean_sigma_roi) stays normalized by the frozen
                            # initial scale, consistent with use_roi_state's
                            # mean_sigma_roi above. The IQR component is instead
                            # normalized by the CURRENT mean_sigma_roi (not the
                            # frozen initial one) -- both shrink together as the GP
                            # converges, so dividing by the initial value collapses
                            # this ratio toward whatever it started at by late
                            # iterations, losing exactly the shape information
                            # (uniform vs. saturated-region uncertainty) it exists
                            # to provide. iqr/mean_sigma_roi is a scale-invariant
                            # relative-spread measure that stays informative
                            # throughout the run regardless of the overall
                            # uncertainty scale.
                            current_sigma_denom = mean_sigma_roi + 1e-6
                            roi_sigma_iqr_flat.append(mean_sigma_roi / sigma_denom)
                            roi_sigma_iqr_flat.append(iqr / current_sigma_denom)
            else:
                if want_roi_state:
                    roi_state_flat = [0.0] * (2 * num_models)
                if want_roi_std_quantiles:
                    roi_std_quantile_flat = [0.0] * (5 * num_models)
                if want_roi_sigma_iqr:
                    roi_sigma_iqr_flat = [0.0] * (2 * num_models)

        # Combine into state tensor
        # Ensure state_dim matches transformer input
        expected_state_dim = self._get_state_dim() # Recalculate or store
        state_list = gp_params_flat + [best_value, step_norm] + best_position.flatten().tolist() \
            + roi_state_flat + roi_std_quantile_flat + roi_sigma_iqr_flat

        # Validate length, pad/truncate if necessary
        if len(state_list) < expected_state_dim:
            state_list.extend([0.0] * (expected_state_dim - len(state_list)))
        elif len(state_list) > expected_state_dim:
            state_list = state_list[:expected_state_dim]

        state = torch.tensor(state_list, device=self.device, dtype=self.dtype)
        return state


    # --- Simulation Logic (Internal Helper) ---
    def _simulate_trajectory(self, gp_idx: int, initial_state: torch.Tensor, max_length: int,
                              initial_roi_candidates: torch.Tensor = None) -> dict:
        """
        Simulate a BO trajectory using one GP model from the ensemble.

        initial_roi_candidates: optional (use_roi_state=True callers only).
        ACTION SELECTION always uses a fresh per-step _optimize_acquisition
        call regardless of this argument -- unlike _simulate_trajectory_joint,
        this method never fantasy-conditions the GP (it's genuinely static
        for the whole rollout), so reusing one fixed candidate set for action
        selection here would make every step's argmax identical (same static
        model + same static candidates => same acquisition landscape every
        time observed_best doesn't change): confirmed empirically, all
        max_length actions came back bit-identical under that design. Only
        the ROI-STATE FEATURES (mean_mu_roi/mean_sigma_roi appended to the
        state, when self.use_roi_state) use initial_roi_candidates when given
        -- fixed across states[0] and every states[1:] in this rollout, so
        they're a consistent, non-noisy signal instead of pure candidate-
        sampling noise (the original always-fresh design's bug) without
        forcing action selection onto a degenerate fixed set (this fix's own
        earlier, since-reverted, mistake). When None (every existing,
        non-ROI-state caller), behavior is completely unchanged: fresh
        _optimize_acquisition every step for both actions and state,
        full broad-search + refinement budget, exactly as before.
        """
        if not self.gp_ensemble: raise RuntimeError("GP Ensemble not initialized.")
        gp_dict = self.gp_ensemble[gp_idx]
        model = gp_dict['model']
        likelihood = gp_dict['likelihood']

        # Track trajectory
        states = [initial_state]
        actions = []
        rewards = [] # Immediate reward (improvement)

        # Use copies of data for simulation to avoid modifying real data
        sim_data_x = self.data_x.clone()
        sim_data_y = self.data_y.clone()

        if sim_data_y.shape[0] > 0:
             # .item() here (not the raw 0-d Tensor): observed_best is later mixed
             # into Python-float arithmetic with sampled_y_item, and reassigned to
             # a plain float on the first improvement -- without this, observed_best
             # silently starts as a Tensor and only becomes a float after the first
             # improvement, so any reward computed on/before that transition (e.g.
             # `new_best - observed_best`) is a 0-d Tensor, and `rewards` ends up a
             # list mixing floats and 0-d Tensors, which torch.tensor(rewards, ...)
             # below cannot reliably convert.
             observed_best = (sim_data_y.max() if self.objective_mode == "maximize" else sim_data_y.min()).item()
        else:
             observed_best = -float('inf') if self.objective_mode == "maximize" else float('inf')

        current_step = self.data_x.shape[0] # Starting step number

        model.eval()
        likelihood.eval()

        for step in range(max_length):
            # 1. Select next point using acquisition function for *this specific GP*.
            # ALWAYS a fresh _optimize_acquisition call, regardless of
            # initial_roi_candidates -- unlike _simulate_trajectory_joint,
            # this method never fantasy-conditions the GP (it's genuinely
            # static all rollout long), so reusing one fixed candidate set for
            # ACTION SELECTION here would make every step's argmax identical
            # (same static model + same static candidates + usually-unchanged
            # observed_best) -- confirmed empirically: all max_length actions
            # came back bit-identical when this was tried. roi_candidates
            # (this step's fresh draw) is still used below for the MES reward,
            # matching the original design (reward evaluated over the same
            # candidate pool the action was chosen from).
            # rollout_acq_function == "rotate": cycle the rollout's query-selection
            # acquisition function by step index instead of using one fixed function
            # for the whole rollout -- UCB explores even saturated regions, MES seeks
            # informative locations, testing whether diverse rollout queries fix
            # training data diversity. Resolved here (not inside _optimize_acquisition
            # itself) since only rollout action selection should rotate; state/
            # diagnostic _optimize_acquisition calls elsewhere still use whatever
            # rollout_acq_function is configured, unchanged.
            step_acq_name = ["ei", "ucb", "pi", "mes"][step % 4] if self.rollout_acq_function == "rotate" else None
            _obs_best_t = observed_best if isinstance(observed_best, torch.Tensor) \
                else torch.tensor(observed_best, device=self.device, dtype=self.dtype)
            if self.rollout_teacher == "softmax":
                next_x_tensor, roi_candidates = self._optimize_acquisition_softmax(
                    gp_idx, _obs_best_t, acq_name=step_acq_name)
            elif self.rollout_teacher == "gradient_ascent":
                # Gradient-ascent rollout generation: trains the DT on
                # trajectories whose x_tau came from the SAME UCB
                # gradient-ascent mechanism validated at real inference
                # (Variant D), instead of the old acquisition-argmax
                # teacher -- tests whether the DT can learn to amortize
                # that ascent rather than needing it re-run at inference.
                _ga_beta = getattr(self.config, 'gp_refinement_beta', 2.0)
                _ga_steps = getattr(self.config, 'rollout_ga_steps', 10)
                x_ga = self._select_x_tau_gradient_ascent(model, _ga_beta, _ga_steps)
                next_x_tensor = x_ga.unsqueeze(0)
                d = self.bo_config.input_dim
                roi_candidates = torch.rand(200, d, device=self.device, dtype=self.dtype)
            else:
                next_x_tensor, roi_candidates = self._optimize_acquisition(
                    gp_idx, _obs_best_t, for_rollout=True, acq_name_override=step_acq_name)  # [1, D], [N_roi, D]
            actions.append(next_x_tensor.squeeze(0)) # Store as [D] tensor

            # 2. Sample simulated observation from GP posterior
            # Uses model.posterior(..., observation_noise=True) rather than
            # likelihood(model(x)): the model's outcome_transform (Standardize)
            # is only applied automatically by .posterior() -- calling model(x)
            # directly returns the *standardized*-scale latent distribution, which
            # would silently corrupt sampled_y (and everything computed from it:
            # rewards, best-so-far tracking, state extraction) into the wrong scale.
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                posterior = model.posterior(next_x_tensor, observation_noise=True)
                # Sample from the predictive distribution
                sampled_y = posterior.sample().reshape(-1) # Shape [1]

            # 3. Update simulated data
            sim_data_x = torch.cat([sim_data_x, next_x_tensor], dim=0)
            sim_data_y = torch.cat([sim_data_y, sampled_y], dim=0)

            # 4. Calculate reward and update best observed in simulation.
            # Running-best bookkeeping is needed regardless of reward type, since
            # _optimize_acquisition's EI/PI branches consume observed_best on later steps.
            sampled_y_item = sampled_y.item()
            if self.objective_mode == "maximize":
                improved = sampled_y_item > observed_best
            else: # Minimize
                improved = sampled_y_item < observed_best

            _diag_oracle_xstar = getattr(self, '_diag_oracle_reward_xstar', None)
            if _diag_oracle_xstar is not None:
                # Diagnostic-only oracle reward (action-reward informativity
                # test): reward = -||x_tau - x*||, guaranteed by construction
                # to be maximized exactly at the true optimum -- unlike
                # use_mes_reward/improvement, this can never be "insufficiently
                # informative" about location quality. RTG = backward cumsum
                # of this (below, unchanged) is then a literal, definite
                # regret-to-x* signal. Uses oracle knowledge of x* (real
                # deployments never have this) -- diagnostic only, gated
                # behind self._diag_oracle_reward_xstar (unset/None by
                # default, never active in the normal pipeline). Running-best
                # bookkeeping (observed_best) still updates normally so
                # _optimize_acquisition's EI/PI branches on later steps are
                # unaffected.
                reward = -(next_x_tensor.squeeze(0).detach().cpu()
                           - _diag_oracle_xstar.cpu()).norm().item()
                if improved:
                    observed_best = sampled_y_item
            elif self.use_mes_reward:
                # Dense MES reward: mutual information between y* and the observation
                # at this query point. roi_candidates is now itself drawn from a
                # uniform, domain-wide candidate pool (see _optimize_acquisition),
                # so it's safe to reuse directly here.
                reward = compute_mes_reward(next_x_tensor.squeeze(0), model, roi_candidates, K=self.mes_k).item()
                if improved:
                    observed_best = sampled_y_item
            else:
                # Original sparse improvement reward (unchanged when use_mes_reward=False).
                new_best = 0
                reward = 0
                if improved:
                    new_best = sampled_y_item
                    reward = (new_best - observed_best) if self.objective_mode == "maximize" \
                        else (observed_best - new_best) # Positive improvement (reduction if minimizing)
                    observed_best = new_best
            rewards.append(reward)

            # 5. Update state for the next step
            # ROI-state FEATURES use the FIXED initial_roi_candidates (when
            # given), decoupled from this step's fresh action-selection
            # roi_candidates above -- consistent across states[0] and every
            # states[1:] within a rollout, without forcing action selection
            # onto a static, degenerate candidate set (see note at step 1).
            current_step += 1
            state_roi = initial_roi_candidates if initial_roi_candidates is not None else roi_candidates
            new_state = self._extract_state(sim_data_x, sim_data_y, current_step, roi_candidates=state_roi)
            states.append(new_state)

            # 6. Check early stopping for simulation rollouts
            early_stop_thresh = getattr(self.simulation_config, 'early_stop_threshold', 1e-6)
            if getattr(self.simulation_config, 'early_stop', False) and reward < early_stop_thresh:
                 if getattr(self.simulation_config, 'verbose', False): 
                     print(f" Sim Rollout Early Stop: Reward {reward:.2e} < {early_stop_thresh:.2e}")
                 break

        # Compute final simple regret for this trajectory (using simulated data)
        final_regret = self._compute_simulated_regret(sim_data_y)

        return {
            # Ensure all are tensors on the correct device
            'states': torch.stack(states).to(self.device, self.dtype),
            'actions': torch.stack(actions).to(self.device, self.dtype),
            'rewards': torch.tensor(rewards, device=self.device, dtype=self.dtype),
            'final_regret': final_regret
        }


    def _simulate_trajectory_joint(self, gp_idx: int, initial_state: torch.Tensor, max_length: int) -> dict:
        """
        Joint-RTG variant of _simulate_trajectory, used when self.rtg_schema
        is "joint" (log-ratio: log(b_tau/b_T)) or "entropy_joint" (pure
        entropy: log(b_tau)+euler_gamma+1, no b_T needed -- see the branch
        near the end of this method). Mirrors validation/phase2_rtg_correlation.py's
        simulate_with_gumbel_b (the version validated in Phase 2 v4 / Phase 3 v4,
        after fixing the roi_candidates variable-swap bug found there):

        - roi_candidates is computed ONCE, from the real (unconditioned) model,
          before the rollout loop -- and reused unchanged for x_tau selection,
          Thompson sampling, and the MES reward's Term 2. _optimize_acquisition
          is never called again inside the loop.
        - The GP is fantasy-conditioned at each step via
          condition_on_observations(), so the posterior actually updates within
          a rollout (unlike _simulate_trajectory, which reuses the same static
          real-data-fit model throughout -- fine for per-step reward RTG, but
          meaningless for joint RTG, which needs b_tau to genuinely shrink as
          more simulated data is conditioned on).
        - b_tau (the Gumbel scale of the domain max y*) is fit via Thompson
          sampling BEFORE conditioning on that step's own observation, i.e. from
          D_tau not D_{tau+1}, at every step including step T (b_T, after all
          max_length conditioning steps).
        - joint_rtg[tau] = log(b_tau / b_T), left unclamped (Phase 2 found
          clamping at 0 distorts the distribution before it's used/compared).

        Since _optimize_acquisition / _acquisition_function_value_botorch /
        _get_posterior_mean_stddev all hard-read self.gp_ensemble[gp_idx]['model']
        with no way to pass a model in directly, the fantasy model is installed
        by temporarily swapping that dict entry before each acquisition call and
        restoring the real model immediately after (try/finally) -- this never
        mutates gp_ensemble's persistent state past this method's return.
        """
        if not self.gp_ensemble: raise RuntimeError("GP Ensemble not initialized.")
        gp_dict = self.gp_ensemble[gp_idx]
        real_model = gp_dict['model'] # persistent, real-data-fit model -- never mutated
        real_model.eval()

        states = [initial_state]
        actions = []
        rewards = []
        gumbel_b_values = []

        sim_data_x = self.data_x.clone()
        sim_data_y = self.data_y.clone()

        if sim_data_y.shape[0] > 0:
            observed_best = sim_data_y.max() if self.objective_mode == "maximize" else sim_data_y.min()
        else:
            observed_best = -float('inf') if self.objective_mode == "maximize" else float('inf')
        if not isinstance(observed_best, torch.Tensor):
            observed_best = torch.tensor(observed_best, device=self.device, dtype=self.dtype)

        current_step = self.data_x.shape[0]
        current_model = real_model # D_0: fantasy model starts as the real, unconditioned model

        # --- roi_candidates computed ONCE, from D_0, before the rollout loop ---
        # rollout_acq_function=="rotate" is resolved to a fixed "ei" for this
        # ONE call -- it establishes a shared, broad candidate POOL (reused for
        # x_tau selection, Thompson sampling, and the MES reward's Term 2), not
        # a per-step action-selection choice, so there's no step index to
        # rotate by yet. The actual per-step rotation happens in
        # _select_x_tau below, mirroring _simulate_trajectory's
        # acq_name_override pattern -- "rotate" itself is never passed to
        # _acquisition_function_value_botorch, which doesn't recognize it.
        _initial_acq_override = "ei" if self.rollout_acq_function == "rotate" else None
        _, roi_candidates = self._optimize_acquisition(
            gp_idx, observed_best, for_rollout=True, acq_name_override=_initial_acq_override
        )

        def _select_x_tau(model, acq_name_override=None):
            """Select x_tau by scoring the FIXED roi_candidates under `model`'s
            acquisition function -- direct argmax, no fresh candidates."""
            gp_dict['model'] = model
            try:
                acq_values = self._acquisition_function_value_botorch(
                    roi_candidates, gp_idx, observed_best,
                    acq_name=acq_name_override if acq_name_override is not None else self.rollout_acq_function,
                    candidate_set=roi_candidates
                )
            finally:
                gp_dict['model'] = real_model
            best_idx = torch.argmax(acq_values)
            return roi_candidates[best_idx].unsqueeze(0)

        K_thompson = self.rtg_joint_k_thompson

        for step in range(max_length):
            # --- Step D_tau: select x_tau and compute b_tau BEFORE conditioning ---
            # Same "rotate" resolution as _simulate_trajectory: cycle
            # ["ei","ucb","pi","mes"] by rollout step index.
            step_acq_name = ["ei", "ucb", "pi", "mes"][step % 4] if self.rollout_acq_function == "rotate" else None
            next_x_tensor = _select_x_tau(current_model, step_acq_name)

            thompson_samples = thompson_sample_y_star(current_model, roi_candidates, K=K_thompson)
            _, b_tau = fit_gumbel_to_samples(thompson_samples)
            gumbel_b_values.append(b_tau)

            actions.append(next_x_tensor.squeeze(0))

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                posterior = current_model.posterior(next_x_tensor, observation_noise=True)
                sampled_y = posterior.sample().reshape(-1)

            sim_data_x = torch.cat([sim_data_x, next_x_tensor], dim=0)
            sim_data_y = torch.cat([sim_data_y, sampled_y], dim=0)

            sampled_y_item = sampled_y.item()
            observed_best_item = observed_best.item()
            if self.objective_mode == "maximize":
                improved = sampled_y_item > observed_best_item
            else:
                improved = sampled_y_item < observed_best_item

            # Per-step reward, computed from the SAME fantasy model at D_tau.
            if self.use_mes_reward:
                reward = compute_mes_reward(next_x_tensor.squeeze(0), current_model, roi_candidates, K=self.mes_k).item()
                if improved:
                    observed_best = sampled_y.reshape(())
            else:
                reward = 0.0
                if improved:
                    new_best_item = sampled_y_item
                    reward = (new_best_item - observed_best_item) if self.objective_mode == "maximize" \
                        else (observed_best_item - new_best_item)
                    observed_best = sampled_y.reshape(())
            rewards.append(reward)

            current_step += 1
            new_state = self._extract_state(sim_data_x, sim_data_y, current_step, roi_candidates=roi_candidates)
            states.append(new_state)

            # --- Now condition on (x_tau, y_tau) to get D_{tau+1} ---
            # _make_fantasy_model, not raw condition_on_observations: the latter
            # silently produces a bit-identical (zero-shift) posterior mean with
            # this class's Normalize+Standardize-transformed GPs (see
            # _make_fantasy_model's docstring for the full root-cause analysis).
            with torch.no_grad():
                current_model = self._make_fantasy_model(
                    current_model, next_x_tensor, sampled_y.reshape(1, 1)
                )

        if self.rtg_schema == "entropy_joint":
            # Pure entropy formulation: RTG[tau] = H(y* | D_tau) =
            # log(b_tau) + euler_gamma + 1. No b_T needed -- this is the
            # whole point of this schema (saves the extra K=100-sample
            # Thompson draw "joint" spends on b_T alone). Not clamped: if
            # b_tau < exp(-euler_gamma - 1) the formula goes slightly
            # negative, which is a Gumbel-MLE fit artifact at finite K, not
            # a real negative entropy -- reported honestly via a neg_frac
            # diagnostic rather than silently floored.
            _EULER_GAMMA = 0.5772156649015329
            joint_rtg = torch.tensor(
                [math.log(b) + _EULER_GAMMA + 1.0 for b in gumbel_b_values],
                device=self.device, dtype=self.dtype,
            )
        else:
            # "joint": log-ratio formulation -- needs b_T, the Gumbel scale
            # after all max_length conditioning steps, using the SAME fixed
            # roi_candidates. (Unchanged from before.)
            thompson_samples_T = thompson_sample_y_star(current_model, roi_candidates, K=K_thompson)
            _, b_T = fit_gumbel_to_samples(thompson_samples_T)

            b_T_safe = max(b_T, 1e-9)
            joint_rtg = torch.tensor(
                [math.log(b / b_T_safe) for b in gumbel_b_values],
                device=self.device, dtype=self.dtype,
            )

        final_regret = self._compute_simulated_regret(sim_data_y)

        return {
            'states': torch.stack(states).to(self.device, self.dtype),
            'actions': torch.stack(actions).to(self.device, self.dtype),
            'rewards': torch.tensor(rewards, device=self.device, dtype=self.dtype),
            'joint_rtg': joint_rtg,
            'final_regret': final_regret,
        }


    def _compute_simulated_regret(self, sim_data_y):
        """Compute simple regret based on simulated observations."""
        if sim_data_y.shape[0] == 0: return 0.0 # Or NaN?

        if self.objective_mode == "maximize":
            best_sim_value = sim_data_y.max().item()
            # Regret for maximization is often defined relative to a known optimum (if available)
            # Or simply use negative best value (lower is better regret)
            return -best_sim_value
        else:
            best_sim_value = sim_data_y.min().item()
            # Regret for minimization is often best value itself (lower is better)
            return best_sim_value


    # --- Decision Transformer Training ---
    def _train_decision_transformer(self, trajectories: list):
        """Train the Decision Transformer on simulated trajectories."""
        if not trajectories:
             print("No trajectories provided for training.")
             return

        # Prepare data for transformer (padding, masking, return-to-go)
        # Prepare batches
        all_states = []
        all_actions = []
        all_rewards = []
        all_timesteps = []
        all_masks = []
        
        max_len = 0
        _diag_grad_active = getattr(self, '_diag_grad_instrumentation', False) \
            or getattr(self, '_diag_coherence_check', False)
        _zero_reward_count = 0
        _total_reward_count = 0
        for traj in trajectories:
            # Get length of this trajectory
            traj_len = min(len(traj['states']) - 1, self.config.transformer.max_seq_length)
            max_len = max(max_len, traj_len)

            # Add to lists
            all_states.append(traj['states'][:traj_len])
            all_actions.append(traj['actions'][:traj_len])

            if self.rtg_schema in ("joint", "entropy_joint"):
                # joint_rtg[tau] (log(b_tau/b_T), or for "entropy_joint",
                # log(b_tau)+gamma+1) is already the full RTG label at each
                # step (not a per-step reward to be summed) -- precomputed in
                # _simulate_trajectory_joint, so the backward-cumsum-of-rewards
                # below does not apply here.
                rtg = traj['joint_rtg'][:traj_len]
            else:
                # For return-to-go, we want the cumulative future reward at each step
                traj_rewards = traj['rewards'][:traj_len]
                rtg = torch.zeros_like(traj_rewards)
                for i in range(traj_len):
                    rtg[i] = traj_rewards[i:].sum()  # Sum of future rewards
                if _diag_grad_active:
                    # zero_reward_frac (matches the original exp1 plot's
                    # "fraction of zero-reward rollout steps"): raw PER-STEP
                    # reward, not the cumulative rtg computed above.
                    _zero_reward_count += int((traj_rewards == 0).sum().item())
                    _total_reward_count += traj_rewards.numel()
            all_rewards.append(rtg)
            
            # Create timestep tensor (long, to match padded_timesteps and the
            # torch.long dtype used for timesteps at inference)
            timesteps = torch.arange(traj_len, device=self.device, dtype=torch.long)
            all_timesteps.append(timesteps)
            
            # Create mask (1 for real data, 0 for padding)
            mask = torch.ones(traj_len, device=self.device, dtype=torch.bool)
            all_masks.append(mask)

        if not all_actions: # Check if any valid trajectories remained
             print("No valid actions found in trajectories for training.")
             return

        # Pad sequences to same length
        padded_states = torch.zeros((len(trajectories), max_len, all_states[0].shape[-1]), device=self.device, dtype=self.dtype)
        padded_actions = torch.zeros((len(trajectories), max_len, self.config.bo.input_dim), device=self.device, dtype=self.dtype)
        padded_rewards = torch.zeros((len(trajectories), max_len), device=self.device, dtype=self.dtype)
        padded_timesteps = torch.zeros((len(trajectories), max_len), device=self.device, dtype=torch.long)
        padded_masks = torch.zeros((len(trajectories), max_len), device=self.device, dtype=torch.bool)
        
        for i in range(len(trajectories)):
            traj_len = len(all_states[i])
            padded_states[i, :traj_len] = all_states[i]
            padded_actions[i, :traj_len] = all_actions[i]
            padded_rewards[i, :traj_len] = all_rewards[i]
            padded_timesteps[i, :traj_len] = all_timesteps[i]
            padded_masks[i, :traj_len] = all_masks[i]

        # Training loop
        self.decision_transformer.train()
        use_quantile = self.decision_transformer.use_quantile_rtg
        num_epochs = self.config.transformer.num_epochs

        # Gradient-norm instrumentation (RTG-insensitivity diagnosis, see
        # _diag_sf_grad_instrumentation.py): epoch-0 loss captured for the
        # print-line below; grad_rtg/grad_state captured from the LAST batch
        # of the LAST epoch's backward() (before clip_grad_norm_, which
        # would otherwise mask the raw imbalance the diagnostic is checking
        # for). Gated behind self._diag_grad_instrumentation (unset/False by
        # default -- never runs/prints in the normal pipeline).
        _diag_grad = getattr(self, '_diag_grad_instrumentation', False)
        _L_loc_epoch0 = None
        _grad_rtg_last = None
        _grad_state_last = None
        _awr_mean_weight_last = None
        _awr_max_weight_last = None

        epoch_iterator = tqdm(range(num_epochs), desc="Training Decision Transformer", disable=not self.verbose)
        for epoch in epoch_iterator:
            # Create batch indices
            batch_size = min(self.config.transformer.batch_size, len(trajectories))
            indices = torch.randperm(len(trajectories))

            total_loss = 0
            total_L_loc = 0.0
            total_L_pinball = 0.0
            num_batches = 0

            for i in range(0, len(trajectories), batch_size):
                batch_indices = indices[i:i+batch_size]

                # Get batch data
                batch_states = padded_states[batch_indices]
                batch_actions = padded_actions[batch_indices]
                batch_rewards = padded_rewards[batch_indices]
                batch_timesteps = padded_timesteps[batch_indices]
                batch_masks = padded_masks[batch_indices]

                if use_quantile:
                    # RTG dropout: with p=0.3 per batch, zero the RTG *input* so the
                    # quantile head learns from state/action context rather than
                    # leaning on RTG token values -- robust to the zero placeholder
                    # used at inference. Does NOT affect the training label below
                    # (batch_rewards), only what's fed into the DT's forward pass.
                    forward_rewards = batch_rewards
                    if torch.rand(()).item() < 0.3:
                        forward_rewards = torch.zeros_like(batch_rewards)

                    predicted_actions, Q_hat = self.decision_transformer(
                        batch_states, batch_actions, forward_rewards, batch_timesteps,
                        batch_masks, return_quantiles=True,
                    )
                    L_loc = torch.nn.functional.mse_loss(
                        predicted_actions[batch_masks], batch_actions[batch_masks]
                    )
                    alpha_levels = self.decision_transformer.quantile_head.alpha_levels
                    L_pinball = compute_pinball_loss(
                        Q_hat[batch_masks], batch_rewards[batch_masks], alpha_levels
                    )
                    loss = L_loc + self.lambda_rtg * L_pinball
                else:
                    # Forward pass (unchanged when quantile RTG is disabled)
                    predicted_actions = self.decision_transformer(
                        batch_states,
                        batch_actions,
                        batch_rewards,
                        batch_timesteps,
                        batch_masks
                    )
                    use_awr = getattr(self.config.transformer, 'use_awr', False)
                    if use_awr:
                        # AWR (advantage-weighted regression): exp(RTG/T)
                        # per-example weights, so examples with higher RTG
                        # (more information remaining / better outcome) pull
                        # harder on the loss than examples with low/negative
                        # RTG -- directly opposes the convergence-to-a-
                        # single-state-only-minimum mechanism confirmed by
                        # the cos_sim trajectory diagnostic (a plain
                        # unweighted MSE loss has no reason to keep
                        # discriminating by RTG once it finds a state-only
                        # solution good enough for the AVERAGE example).
                        # SF-DRO has no fidelity head (single-fidelity, MF-
                        # only concept) -- weighting applies to the sole
                        # (location) loss term, not a location+fidelity sum.
                        awr_temp_cfg = getattr(self.config.transformer, 'awr_temperature', None)
                        if awr_temp_cfg is not None:
                            temperature = torch.tensor(
                                float(awr_temp_cfg), device=self.device, dtype=self.dtype)
                        else:
                            rtg_valid = batch_rewards[batch_masks]
                            if rtg_valid.numel() > 0:
                                temperature = rtg_valid.abs().median().clamp(min=1e-4)
                            else:
                                temperature = torch.tensor(1.0, device=self.device, dtype=self.dtype)

                        weights = (batch_rewards / temperature).exp().clamp(max=20.0)  # [B,T]
                        vm = batch_masks.float()
                        loss = (
                            torch.nn.functional.mse_loss(
                                predicted_actions, batch_actions, reduction='none'
                            ).mean(dim=-1) * weights * vm
                        ).sum() / vm.sum().clamp_min(1)

                        if getattr(self, '_diag_coherence_check', False):
                            valid_weights = weights[batch_masks]
                            _awr_mean_weight_last = valid_weights.mean().item()
                            _awr_max_weight_last = valid_weights.max().item()
                    else:
                        # Compute loss (MSE on action prediction) -- bit-for-
                        # bit unchanged from before use_awr existed.
                        loss = torch.nn.functional.mse_loss(
                            predicted_actions[batch_masks],
                            batch_actions[batch_masks]
                        )
                    L_loc, L_pinball = loss, None

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient-norm capture -- BEFORE clip_grad_norm_ (below),
                # which would rescale away exactly the imbalance this is
                # meant to measure. Only meaningful/captured on the last
                # epoch (overwritten each batch, so holds the LAST batch's
                # values once the epoch's batch loop finishes).
                if _diag_grad and epoch == num_epochs - 1:
                    _grad_rtg_last = self.decision_transformer.reward_embedding.weight.grad.norm().item()
                    _grad_state_last = self.decision_transformer.state_embedding.weight.grad.norm().item()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.decision_transformer.parameters(),
                    1.0
                )

                self.optimizer.step()

                total_loss += loss.item()
                total_L_loc += L_loc.item()
                if L_pinball is not None:
                    total_L_pinball += L_pinball.item()
                num_batches += 1

            epoch_iterator.set_postfix(loss=total_loss / num_batches)

            if _diag_grad and epoch == 0:
                _L_loc_epoch0 = total_L_loc / num_batches

            if epoch == num_epochs - 1: # Diagnostics from the final epoch only
                self._last_train_diagnostics = {"L_loc": total_L_loc / num_batches}
                if _diag_grad:
                    t = len(self.real_history_states)
                    grad_ratio = _grad_state_last / max(_grad_rtg_last, 1e-8)
                    print(f"iter {t} | L_loc[e0]={_L_loc_epoch0:.4f} | "
                          f"L_loc[e{num_epochs-1}]={total_L_loc / num_batches:.4f} | "
                          f"grad_rtg={_grad_rtg_last:.4f} grad_state={_grad_state_last:.4f} "
                          f"ratio={grad_ratio:.1f}x", flush=True)
                if use_quantile:
                    self._last_train_diagnostics["L_pinball"] = total_L_pinball / num_batches

                # Multi-checkpoint coherence + RTG-sensitivity probe (extends
                # the single-iteration coherence check): fires at whichever
                # real BO iterations are listed in self._diag_checkpoint_iters
                # (default {5}, matching the original single-checkpoint
                # behavior), AFTER this iteration's training completes (not
                # before, unlike the original version) so L_loc_final,
                # mean_cos_sim and probe_spread all reflect the SAME
                # end-of-iteration-t snapshot. Read-only: zero_grad() before
                # and after, no optimizer.step() in this block.
                _diag_ckpt_iters = getattr(self, '_diag_checkpoint_iters', {5})
                if getattr(self, '_diag_coherence_check', False) and len(self.real_history_states) in _diag_ckpt_iters:
                    t = len(self.real_history_states)
                    L_loc_final = total_L_loc / num_batches

                    self.decision_transformer.eval()
                    grads = []
                    for i in range(len(trajectories)):
                        self.optimizer.zero_grad()
                        s_i = padded_states[i:i + 1]
                        a_i = padded_actions[i:i + 1]
                        r_i = padded_rewards[i:i + 1]
                        ts_i = padded_timesteps[i:i + 1]
                        m_i = padded_masks[i:i + 1]
                        if not m_i.any():
                            continue
                        pred_i = self.decision_transformer(s_i, a_i, r_i, ts_i, m_i)
                        loss_i = torch.nn.functional.mse_loss(pred_i[m_i], a_i[m_i])
                        loss_i.backward()
                        g_i = self.decision_transformer.reward_embedding.weight.grad.detach().flatten().clone()
                        grads.append(g_i)
                    self.optimizer.zero_grad()

                    G = torch.stack(grads)
                    G_norm = G / G.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    cos_matrix = G_norm @ G_norm.T
                    Nc = cos_matrix.shape[0]
                    mean_cos_sim = ((cos_matrix.sum() - cos_matrix.diagonal().sum())
                                     / (Nc * (Nc - 1))).item()

                    # RTG-sensitivity probe: same design as MF-DRO's
                    # propose_mf sweep, adapted to SF-DRO's own single-
                    # timestep real-inference call (see
                    # _propose_next_candidate: state_seq/dummy_action/
                    # target_rtg/dummy_timestep, sequence length 1). No
                    # single real target_rtg is in scope inside this method,
                    # so the reference is this batch's own mean valid RTG --
                    # a representative target for iteration t.
                    valid_rtg_for_probe = padded_rewards[padded_masks]
                    ref_rtg = valid_rtg_for_probe.mean().item()
                    probe_state = padded_states[0:1, 0:1]  # [1,1,state_dim]
                    dummy_action = torch.zeros(
                        (1, 1, self.config.bo.input_dim), device=self.device, dtype=self.dtype)
                    dummy_timestep = torch.zeros((1, 1), device=self.device, dtype=torch.long)
                    probe_preds = []
                    with torch.no_grad():
                        for mult in (0.25, 0.5, 1.0, 2.0, 4.0):
                            target_rtg = torch.tensor(
                                [[ref_rtg * mult]], device=self.device, dtype=self.dtype)
                            pred = self.decision_transformer(
                                probe_state, dummy_action, target_rtg, dummy_timestep,
                                attention_mask=None)
                            probe_preds.append(pred.reshape(-1))
                    probe_t = torch.stack(probe_preds)
                    probe_spread = (probe_t - probe_t.mean(dim=0, keepdim=True)).norm(dim=1).mean().item()

                    self.decision_transformer.train()
                    _mw = f"{_awr_mean_weight_last:.3f}" if _awr_mean_weight_last is not None else "n/a"
                    _xw = f"{_awr_max_weight_last:.3f}" if _awr_max_weight_last is not None else "n/a"
                    _zrf = (_zero_reward_count / _total_reward_count) if _total_reward_count > 0 else float('nan')
                    print(f"{self.benchmark_name} | iter {t} | L_loc_final={L_loc_final:.4f} | "
                          f"mean_cos_sim={mean_cos_sim:.3f} | mean_weight={_mw} | "
                          f"max_weight={_xw} | zero_reward_frac={_zrf:.4f} | "
                          f"probe_spread={probe_spread:.4f} (N={Nc})", flush=True)
    

    # --- RTG Target Selection (Internal Helper) ---
    def _compute_rtg_target(self, batch_rtg0_list: list, current_real_state: torch.Tensor = None) -> tuple:
        """
        Compute the inference-time RTG target per self.rtg_schema, updating the
        persistent running_max_rtg tracker along the way.

        batch_rtg0_list: RTG[0] (= sum of all rewards) of every rollout generated
        this iteration.
        current_real_state: only used by rtg_schema=="quantile" (past warmup) to
        build the sliding-window inference sequence.
        Returns: (rtg_target, batch_max_rtg)
        """
        # NOTE: "entropy_joint" deliberately does NOT enter this percentile
        # branch (check below is == "joint", not a tuple) -- it falls through
        # to the plain max() in the else clause, matching "dynamic"'s
        # treatment. See rtg_target dispatch below for why.
        if self.rtg_schema == "joint" and batch_rtg0_list:
            # joint_rtg[0] = log(b_0/b_T) is a log-ratio of two independently
            # Thompson/Gumbel-MLE-fit scale parameters (K=100 samples each) --
            # a much noisier, heavier-tailed quantity than a bounded per-step
            # reward sum. Diagnosed empirically (validation/phase4_step3_diagnosis.py):
            # at early BO stages the batch is centered near 0 with std >> mean
            # (CV~5), so raw max() over 75 rollouts mostly picks up a rare MLE
            # noise spike, not a genuinely achievable target -- and because
            # running_max_rtg never decays, that spike permanently inflates the
            # floor for every later iteration, producing a stuck DT policy
            # (confirmed: DRO-MES-Joint went flat for 28-30/30 iterations on
            # 2/3 seeds in the initial 3-seed comparison). Using the 90th
            # percentile instead of the raw max is far less sensitive to a
            # single noisy rollout while still selecting an ambitious
            # (not median/typical) target, matching the same "high but
            # learnable" intent as the raw max used for other schemas.
            batch_max_rtg = float(np.percentile(batch_rtg0_list, 90))
        else:
            batch_max_rtg = max(batch_rtg0_list) if batch_rtg0_list else 0.0
        self.running_max_rtg = max(self.running_max_rtg, batch_max_rtg)
        self._last_q_hat = None

        if self.rtg_schema == "fixed":
            rtg_target = 1.0
        elif self.rtg_schema in ("dynamic", "entropy_joint"):
            # No floor: can collapse to near-zero (or, for entropy_joint on a
            # narrow-y*-distribution benchmark like Hartmann_6D, stay negative
            # throughout -- confirmed correct via sanity check, not a bug) in
            # late iterations. Intentional -- this is one of the designs being
            # compared in the RTG schema study. entropy_joint explicitly
            # reuses "dynamic"'s plain batch_max_rtg (no floor, no percentile
            # trimming) rather than "joint"'s floored/percentile treatment,
            # per explicit user instruction.
            rtg_target = batch_max_rtg
        elif self.rtg_schema in ("floored", "joint"):
            # "joint" reuses the identical floored formula: batch_rtg0_list is
            # already joint_rtg[0] per rollout (see _propose_next_candidate),
            # so running_max_rtg/batch_max_rtg here are tracking joint-RTG scale,
            # not per-step reward scale. Phase 3 confirmed the two are
            # dimensionally compatible (log-ratio of Gumbel scales lands in the
            # same order of magnitude as summed per-step reward), so no separate
            # floor formula is needed -- only running_max_rtg's own reset at
            # __init__ (self.running_max_rtg = 0.0) matters when switching
            # schemas, so a fresh DRO instance never floors against a stale
            # per-step-RTG scale.
            rtg_target = max(batch_max_rtg, self.alpha_floor * self.running_max_rtg)
        elif self.rtg_schema == "quantile":
            # t = number of real BO iterations completed so far (1-indexed for the
            # iteration about to run), since real_history_states is only appended
            # to at the end of _propose_next_candidate.
            t = len(self.real_history_states) + 1
            if self.decision_transformer.use_quantile_rtg and t > self.rtg_warmup and current_real_state is not None:
                rtg_target = self._compute_quantile_rtg_target(current_real_state, batch_max_rtg)
            else:
                # Falls back to floored for the first rtg_warmup iterations.
                rtg_target = max(batch_max_rtg, self.alpha_floor * self.running_max_rtg)
        else:
            raise ValueError(f"Unknown rtg_schema: {self.rtg_schema}")

        return rtg_target, batch_max_rtg

    def _compute_quantile_rtg_target(self, current_real_state: torch.Tensor, batch_max_rtg: float) -> float:
        """
        Self-predicted RTG target for rtg_schema=="quantile": builds a sliding
        window of the last self.simulation_config.max_rollout_length real BO
        steps, ending at the current (not-yet-acted-upon) state, runs it through
        the DT, and interpolates the predicted quantiles at self.alpha_inference.

        Uses h_action[-2] (not h_action[-1]) to query the quantile head, to match
        how the head is trained: forward()'s one-step-shift trick pairs
        h_action[tau-1] with R_true[tau], i.e. Q_hat[tau] is predicted from the
        hidden state *before* tau. h_action[-1] is h^a_{L-1}, which during
        training is the predictor for R_L -- one position beyond this window, not
        the RTG at the current/last position. h_action[-2] = h^a_{L-2} is the
        correct predictor for R_{L-1}, the quantity we actually want here. Falls
        back to the floored target when the window is too short (L=1) to have a
        second-to-last action token at all.

        Sets self._last_q_hat to the raw [M] prediction for logging (calibration,
        spread), on top of returning the scalar target.
        """
        W = self.simulation_config.max_rollout_length
        n_prior = max(W - 1, 0)
        prior_states = self.real_history_states[-n_prior:] if n_prior > 0 else []
        prior_actions = self.real_history_actions[-n_prior:] if n_prior > 0 else []

        states_window = prior_states + [current_real_state] # length <= W, ends at "now"
        action_dim = self.bo_config.input_dim

        if prior_actions:
            actions_tensor = torch.stack(prior_actions)
        else:
            actions_tensor = torch.empty(0, action_dim, device=self.device, dtype=self.dtype)
        # The action for the current (last) state hasn't been chosen yet -- pad
        # with one zero action, per the spec's inference recipe.
        pad_action = torch.zeros(1, action_dim, device=self.device, dtype=self.dtype)
        actions_seq = torch.cat([actions_tensor, pad_action], dim=0).unsqueeze(0) # [1, L, action_dim]

        states_seq = torch.stack(states_window).unsqueeze(0) # [1, L, state_dim]
        L = states_seq.shape[1]
        rtgs_seq = torch.zeros(1, L, device=self.device, dtype=self.dtype) # placeholder, irrelevant at inference
        timesteps_seq = torch.arange(L, device=self.device, dtype=torch.long).unsqueeze(0)

        self.decision_transformer.eval()
        with torch.no_grad():
            h_action = self.decision_transformer.get_action_hidden_states(states_seq, actions_seq, rtgs_seq, timesteps_seq)
            if h_action.shape[1] < 2:
                # No second-to-last action token available in this short a
                # window -- fall back to the same floored target used during
                # rtg_warmup.
                return max(batch_max_rtg, self.alpha_floor * self.running_max_rtg)
            Q_hat_last = self.decision_transformer.quantile_head(h_action[:, -2, :]).squeeze(0) # [M]

        # Quantile crossing guard: independently-trained quantile heads aren't
        # guaranteed monotonic in alpha, especially early in training / with
        # little data. Sorting ascending (Chernozhukov et al. 2010 rearrangement)
        # restores Q_hat[0.1] <= ... <= Q_hat[0.9] by construction before
        # interpolating, since alpha_levels is itself ascending.
        Q_hat_last, _ = torch.sort(Q_hat_last)

        alpha_levels = self.decision_transformer.quantile_head.alpha_levels.tolist()
        alpha_inference = self.alpha_inference if self.alpha_inference is not None else 0.5
        rtg_target = interpolate_quantile(Q_hat_last, alpha_inference, alpha_levels)

        self._last_q_hat = Q_hat_last.tolist()
        return rtg_target.item()

    # --- Main Logic for Proposing Next Candidate ---
    def _make_synthetic_expert_trajectory(self, x_star, rollout_length):
        """
        Synthetic-expert-demonstration trajectory (diagnostic only), ported
        from _synthetic_expert_worker.py's MF-DRO regime: linearly
        interpolates from a random x_start to the TRUE known optimum x_star
        over rollout_length steps, evaluating the TRUE objective at each
        point (no GP sampling) -- an unambiguously-correct demonstration.
        Tests whether the DT can learn a working policy given clean data at
        all; if it still can't, the problem is DT training/capacity, not
        the KO-GP/MES rollout generator or reward informativity.

        Reward is per-step true-value improvement, normalized so that
        _train_decision_transformer's backward cumsum of rewards reproduces
        exactly RTG[tau] = (y_final - y_tau) / (y_final - y_0) -- the
        reference script's own formula ("cumulative reward" toward the true
        optimum, by construction perfectly correlated with proximity to the
        objective's actual value along this path).

        best_value/best_position/step-count state slots (inside
        _extract_state) are seeded from the REAL accumulated self.data_x/
        self.data_y (exactly matching _simulate_trajectory's own
        sim_data_x = self.data_x.clone() convention, and matching what
        _propose_next_candidate's real-inference current_real_state uses),
        with the synthetic expert path appended on top -- NOT an isolated
        local history starting at x_start alone. That isolated-history
        version (the reference script's original choice) was found to
        create a severe train/inference state mismatch: training's
        best_value there is the max of only 1-4 draws (mean ~0.66, 90%
        range [0.09, 1.76] on Hartmann_6D) while real inference's
        best_value is the max of ~36+ draws (mean ~1.67, actual observed
        1.86 -- above the training distribution's 95th percentile), AND the
        two are correlated oppositely: training's best_value/best_position
        are always coupled (best-so-far is always near x*, since the
        synthetic path walks straight there) while real data's are
        essentially independent (i.i.d. LHS draws) -- an input combination
        (high value, far from x*) never seen once during training. Grounding
        in self.data_x/self.data_y eliminates this: states[0] here is
        BIT-IDENTICAL to current_real_state at inference (same data, same
        _extract_state call), so only the synthetic path's own few appended
        points can differ. The GP-hyperparameter block still reflects the
        REAL, currently-fitted self.gp_ensemble (unavoidable/appropriate --
        there is no synthetic GP here).
        """
        d = self.bo_config.input_dim
        domain_min, domain_max = self.bounds[0], self.bounds[1]
        x_start = domain_min + (domain_max - domain_min) * torch.rand(
            d, device=self.device, dtype=self.dtype
        )

        T = rollout_length
        x_star_t = x_star.to(device=self.device, dtype=self.dtype)
        xs = [x_start + (x_star_t - x_start) * (tau / max(T - 1, 1)) for tau in range(T)]

        def _eval(x):
            y = self.objective_function(x)
            return y.item() if isinstance(y, torch.Tensor) else float(y)

        ys = [_eval(x) for x in xs]

        sim_x = list(self.data_x.clone())
        sim_y = list(self.data_y.clone())
        states = []
        real_step0 = self.data_x.shape[0]
        for tau in range(T):
            cur_x = torch.stack(sim_x)
            cur_y = torch.stack(sim_y) if sim_y and isinstance(sim_y[0], torch.Tensor) else torch.tensor(sim_y, device=self.device, dtype=self.dtype)
            states.append(self._extract_state(cur_x, cur_y, real_step0 + tau, roi_candidates=None))
            sim_x.append(xs[tau])
            sim_y.append(torch.tensor(ys[tau], device=self.device, dtype=self.dtype))
        # One more state after the final action, matching _simulate_trajectory's
        # own states-has-one-more-entry-than-actions convention
        # (_train_decision_transformer truncates to len(states)-1).
        cur_x = torch.stack(sim_x)
        cur_y = torch.stack(sim_y)
        states.append(self._extract_state(cur_x, cur_y, real_step0 + T, roi_candidates=None))

        y0, y_final = ys[0], ys[-1]
        denom = max(y_final - y0, 1e-8)  # matches the reference script's own flooring exactly
        rewards = [
            ((ys[tau + 1] - ys[tau]) / denom) if tau < T - 1 else 0.0
            for tau in range(T)
        ]

        return {
            'states': torch.stack(states).to(self.device, self.dtype),
            'actions': torch.stack(xs).to(self.device, self.dtype),
            'rewards': torch.tensor(rewards, device=self.device, dtype=self.dtype),
            'final_regret': self._compute_simulated_regret(cur_y),
        }

    def _select_x_tau_gradient_ascent(self, model, beta=2.0, steps=10):
        """
        Rollout-generation teacher (gated behind self.rollout_teacher ==
        "gradient_ascent"): single random-start UCB gradient ascent, used
        to pick x_tau during _simulate_trajectory instead of the old
        broad-random-search-then-local-polish _optimize_acquisition
        teacher. Cheap variant of _refine_ucb_core (steps=10 vs the 30
        used once per real iteration at inference) -- called up to
        rollouts_per_iter * rollout_length times per real BO iteration
        (75*4=300 here), so a much smaller step budget than inference-time
        refinement is needed to keep rollout generation's wall-clock
        reasonable.
        """
        d = self.bo_config.input_dim
        x_start = torch.rand(d, device=self.device, dtype=self.dtype)
        return self._refine_ucb_core(x_start, model, beta, steps)

    def _refine_ucb_core(self, x_init, model, beta, steps, lr=None):
        """
        Core single-GP UCB gradient ascent (Adam, warm-started from x_init,
        clamped to [0,1]^d each step) -- shared by every refinement variant
        below. Called from inside _propose_next_candidate's outer `with
        torch.no_grad():` block -- torch.enable_grad() is required to
        locally override that and actually build a graph for .backward().
        """
        lr = lr if lr is not None else self.config.gp_refinement_lr
        with torch.enable_grad():
            x = x_init.clone().detach().requires_grad_(True)
            opt = torch.optim.Adam([x], lr=lr)
            model.eval()
            for _ in range(steps):
                opt.zero_grad()
                with gpytorch.settings.fast_pred_var():
                    posterior = model.posterior(x.unsqueeze(0))
                    mu = posterior.mean.squeeze()
                    sigma = posterior.variance.clamp_min(1e-8).sqrt().squeeze()
                ucb = mu + beta * sigma
                (-ucb).backward()
                opt.step()
                with torch.no_grad():
                    x.clamp_(0.0, 1.0)
        return x.detach()

    def _ucb_value(self, x, model, beta):
        """UCB at x under `model`, no grad -- used for restart selection."""
        with torch.no_grad():
            posterior = model.posterior(x.reshape(1, -1))
            mu = posterior.mean.squeeze()
            sigma = posterior.variance.clamp_min(1e-8).sqrt().squeeze()
            return (mu + beta * sigma).item()

    def _refine_proposal_ucb(self, x_init):
        """
        VARIANT A (default): warm-start UCB gradient ascent from the DT's
        proposal, single GP (gp_ensemble[0]), fixed beta/steps from config.
        Diagnostic: isolates whether location quality is the bottleneck in
        the SF setting, where fidelity is not a factor -- every query is
        HF. Returns refined x in [0,1]^d. Gated behind
        self.config.use_gp_refinement at the call site -- bit-for-bit
        unchanged pipeline when off.
        """
        model = self.gp_ensemble[0]['model']
        beta = getattr(self.config, 'gp_refinement_beta', 2.0)
        return self._refine_ucb_core(x_init, model, beta, self.config.gp_refinement_steps)

    def _refine_proposal_ucb_ensemble(self, x_init):
        """
        VARIANT B: multi-ensemble UCB -- gradient ascent on the MEAN UCB
        score averaged across all M gp_ensemble members jointly (one
        shared x, one Adam trajectory), instead of a single member's GP.
        Tests whether a single ensemble member's posterior is an
        unreliable/noisy refinement target and averaging over the ensemble
        gives a more robust ascent direction.
        """
        beta = getattr(self.config, 'gp_refinement_beta', 2.0)
        steps = self.config.gp_refinement_steps
        with torch.enable_grad():
            x = x_init.clone().detach().requires_grad_(True)
            opt = torch.optim.Adam([x], lr=self.config.gp_refinement_lr)
            for gp_dict in self.gp_ensemble:
                gp_dict['model'].eval()
            for _ in range(steps):
                opt.zero_grad()
                ucb_sum = 0.0
                with gpytorch.settings.fast_pred_var():
                    for gp_dict in self.gp_ensemble:
                        posterior = gp_dict['model'].posterior(x.unsqueeze(0))
                        mu = posterior.mean.squeeze()
                        sigma = posterior.variance.clamp_min(1e-8).sqrt().squeeze()
                        ucb_sum = ucb_sum + (mu + beta * sigma)
                ucb_mean = ucb_sum / len(self.gp_ensemble)
                (-ucb_mean).backward()
                opt.step()
                with torch.no_grad():
                    x.clamp_(0.0, 1.0)
        return x.detach()

    def _refine_proposal_ucb_twostage(self, x_init):
        """
        VARIANT C: two-stage refinement, single GP (gp_ensemble[0]).
        Stage 1: high beta=5.0, 20 steps -- coarse, exploration-weighted
        ascent (uncertainty dominates, can move far from x_init). Stage 2:
        low beta=0.5, 20 steps, warm-started from stage 1's endpoint --
        fine, exploitation-weighted refinement (mean dominates, local
        polish). Tests whether decoupling "which basin" from "where in the
        basin" helps versus a single fixed-beta ascent throughout.
        """
        model = self.gp_ensemble[0]['model']
        x_stage1 = self._refine_ucb_core(x_init, model, beta=5.0, steps=20)
        x_stage2 = self._refine_ucb_core(x_stage1, model, beta=0.5, steps=20)
        return x_stage2

    def _refine_proposal_ucb_restarts(self, x_init, n_restarts=5):
        """
        VARIANT D: N=5 multi-start refinement, single GP (gp_ensemble[0]),
        same beta/steps as Variant A. Runs independent UCB ascent from the
        DT's own proposal PLUS (n_restarts-1) uniform-random starting
        points, then keeps whichever converged point has the HIGHEST final
        UCB value (evaluated under the same beta). Tests whether Variant
        A's single ascent trajectory is getting stuck in a poor local
        optimum of the UCB landscape.
        """
        model = self.gp_ensemble[0]['model']
        beta = getattr(self.config, 'gp_refinement_beta', 2.0)
        steps = self.config.gp_refinement_steps
        d = x_init.shape[-1]
        starts = [x_init] + [
            torch.rand(d, device=self.device, dtype=self.dtype) for _ in range(n_restarts - 1)
        ]
        best_x, best_ucb = None, -float('inf')
        for s in starts:
            x_ref = self._refine_ucb_core(s, model, beta, steps)
            u = self._ucb_value(x_ref, model, beta)
            if u > best_ucb:
                best_ucb, best_x = u, x_ref
        return best_x

    def _refine_proposal_ucb_restarts_pure(self, n_restarts=5):
        """
        DIAGNOSTIC 1 (D_nodt): same as _refine_proposal_ucb_restarts
        (Variant D) but with NO DT proposal seeding any start -- all
        n_restarts starts are uniform-random. Isolates whether Variant D's
        DT-seeded start contributes anything beyond pure multi-start UCB.
        """
        model = self.gp_ensemble[0]['model']
        beta = getattr(self.config, 'gp_refinement_beta', 2.0)
        steps = self.config.gp_refinement_steps
        d = self.bo_config.input_dim
        starts = [torch.rand(d, device=self.device, dtype=self.dtype) for _ in range(n_restarts)]
        best_x, best_ucb = None, -float('inf')
        for s in starts:
            x_ref = self._refine_ucb_core(s, model, beta, steps)
            u = self._ucb_value(x_ref, model, beta)
            if u > best_ucb:
                best_ucb, best_x = u, x_ref
        return best_x

    def _measure_ucb_peak_spread(self):
        """
        Failure-mode-1 diagnostic: for each ensemble member, find its UCB
        argmax over 500 random candidates, then return the mean pairwise
        L2 distance between all M peak locations. High value = members
        point to different basins (multimodal action distribution across
        the ensemble); low value = members agree on the same basin.
        Gated behind self._diag_ucb_spread (unset/None by default -- never
        runs/prints in the normal pipeline).
        """
        d = self.bo_config.input_dim
        peaks = []
        X_probe = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * torch.rand(
            500, d, device=self.device, dtype=self.dtype
        )
        for gp_dict in self.gp_ensemble:
            model = gp_dict['model']
            with torch.no_grad():
                post = model.posterior(X_probe)
                mu = post.mean.squeeze()
                sigma = post.variance.clamp_min(1e-8).sqrt().squeeze()
                ucb = mu + 2.0 * sigma
            peak_idx = ucb.argmax().item()
            peaks.append(X_probe[peak_idx])
        peaks = torch.stack(peaks)  # [M, d]
        diffs = peaks.unsqueeze(0) - peaks.unsqueeze(1)
        dists = diffs.norm(dim=-1)
        n = len(peaks)
        spread = dists.sum() / (n * (n - 1))
        return spread.item()

    def _propose_next_candidate_no_dt(self, propose_mode) -> torch.Tensor:
        """
        Bypasses rollout simulation and DT training entirely -- diagnostic
        modes isolating whether DRO's learned proposal contributes anything
        beyond the (already lognormal-prior-calibrated) GP itself
        (_construct_gp_model already has the LogNormalPrior fix).

        "multistart_ucb_nodt" (Diagnostic 1, D_nodt): 5 random-restart UCB
        ascent, same beta/steps as Variant D, but with NO DT proposal
        seeding any of the 5 starts (all uniform-random).

        "naivebo_lognormal" (Diagnostic 2): single-shot _optimize_acquisition
        (this class's own existing acquisition-optimization routine -- broad
        random search + local gradient refinement, no restarts, no DT) on
        gp_ensemble[0] with acq_name_override="ucb" -- standard NaiveBO on
        the calibrated GP.
        """
        _real_query_t_start = time.perf_counter()
        with torch.no_grad():
            observed_best = self.data_y.max() if self.objective_mode == "maximize" else self.data_y.min()

        if getattr(self, '_diag_ucb_spread', False):
            _spread = self._measure_ucb_peak_spread()
            self._diag_ucb_spread_log = getattr(self, '_diag_ucb_spread_log', [])
            self._diag_ucb_spread_log.append(_spread)
            print(f"iter {len(self._diag_ucb_spread_log) - 1}: ucb_peak_spread = {_spread:.4f} (mode={propose_mode})", flush=True)

        if propose_mode == 'multistart_ucb_nodt':
            next_action = self._refine_proposal_ucb_restarts_pure(n_restarts=5)
        else:  # naivebo_lognormal
            with torch.no_grad():
                next_action_batched, _ = self._optimize_acquisition(
                    0, observed_best, for_rollout=False, acq_name_override="ucb"
                )
            next_action = next_action_batched.squeeze(0)

        domain_min, domain_max = self.bounds[0], self.bounds[1]
        next_action = torch.min(torch.max(next_action.detach(), domain_min), domain_max)

        self._last_real_query_time = time.perf_counter() - _real_query_t_start
        self._last_iter_diagnostics = {"real_query_time": self._last_real_query_time}
        self._last_diagnostics = {}
        self._pending_log = True

        with torch.no_grad():
            current_real_state = self._extract_state(self.data_x, self.data_y, self.data_x.shape[0])
        self.real_history_states.append(current_real_state.detach().clone())
        self.real_history_actions.append(next_action.detach().clone())

        _diag_xstar = getattr(self, '_diag_xstar', None)
        if _diag_xstar is not None:
            _t = len(self.real_history_states) - 1
            _dist = (next_action.detach().cpu() - _diag_xstar.cpu()).norm().item()
            print(f"iter {_t}: dist(x_t, x*) = {_dist:.4f} (mode={propose_mode})", flush=True)

        return next_action.unsqueeze(0)

    def _propose_next_candidate(self) -> torch.Tensor:
        """
        Run simulations, train transformer, and use it to propose the next point.
        Returns tensor shape [1, input_dim].
        """
        _propose_mode = getattr(self.config, 'propose_mode', 'dt')
        if _propose_mode in ('multistart_ucb_nodt', 'naivebo_lognormal'):
            return self._propose_next_candidate_no_dt(_propose_mode)

        if getattr(self, '_diag_ucb_spread', False):
            _spread = self._measure_ucb_peak_spread()
            self._diag_ucb_spread_log = getattr(self, '_diag_ucb_spread_log', [])
            self._diag_ucb_spread_log.append(_spread)
            print(f"iter {len(self._diag_ucb_spread_log) - 1}: ucb_peak_spread = {_spread:.4f} (mode={_propose_mode})", flush=True)

        # 1. Simulate trajectories from current state using GP ensemble
        _rollout_sim_t_start = time.perf_counter()
        trajectories = []
        batch_rtg0_list = [] # RTG[0] per rollout, reset at the start of every iteration
        if self.verbose: print("Simulating trajectories...")
        num_rollouts = self.simulation_config.num_rollouts
        rollout_length = self.simulation_config.max_rollout_length

        use_joint_rtg = (self.rtg_schema in ("joint", "entropy_joint"))

        # Synthetic-expert-demonstration diagnostic (action-reward
        # informativity / DT-capacity test, ported from
        # _synthetic_expert_worker.py's MF-DRO regime): replaces ONLY this
        # trajectory-generation block with hand-designed, unambiguously-
        # correct trajectories -- _train_decision_transformer, the real
        # DT-inference call below, and everything else in this method stay
        # untouched. Gated behind self._diag_synthetic_expert_xstar
        # (unset/None by default -- never active in the normal pipeline).
        _diag_synth_xstar = getattr(self, '_diag_synthetic_expert_xstar', None)
        if _diag_synth_xstar is not None:
            for _ in range(num_rollouts):
                trajectory = self._make_synthetic_expert_trajectory(_diag_synth_xstar, rollout_length)
                trajectories.append(trajectory)
                batch_rtg0_list.append(trajectory['rewards'].sum().item())
            self._last_rollout_sim_time = time.perf_counter() - _rollout_sim_t_start
            self._last_rollout_action_diversity = None
        else:
          for gp_idx in tqdm(range(len(self.gp_ensemble)), desc="Simulating Rollouts", disable=not self.verbose):
             # Determine how many rollouts per GP model
             rollouts_per_gp = max(1, num_rollouts // len(self.gp_ensemble))
             for _ in range(rollouts_per_gp):
                  # Get current state based on *real* data. For use_roi_state
                  # with the non-joint trajectory function, roi_candidates is
                  # computed ONCE per rollout here and reused for BOTH the
                  # initial state (states[0]) and every step inside
                  # _simulate_trajectory (states[1:]) -- fixes the states[0]
                  # zero-padding / per-step re-randomization bugs found in
                  # Stage 1. use_joint_rtg is intentionally untouched here:
                  # _simulate_trajectory_joint already computes and reuses its
                  # own roi_candidates internally with the correct design.
                  if (self.use_roi_state or self.use_roi_std_quantiles or self.use_roi_sigma_iqr) and not use_joint_rtg:
                      _, roi_candidates_for_rollout = self._optimize_acquisition(
                          gp_idx, self.data_y.max(), for_rollout=True
                      )
                  else:
                      roi_candidates_for_rollout = None
                  current_state = self._extract_state(
                      self.data_x, self.data_y, self.data_x.shape[0],
                      roi_candidates=roi_candidates_for_rollout,
                  )
                  if use_joint_rtg:
                      trajectory = self._simulate_trajectory_joint(gp_idx, current_state, rollout_length)
                      trajectories.append(trajectory)
                      batch_rtg0_list.append(trajectory['joint_rtg'][0].item())
                  else:
                      trajectory = self._simulate_trajectory(
                          gp_idx, current_state, rollout_length,
                          initial_roi_candidates=roi_candidates_for_rollout,
                      )
                      trajectories.append(trajectory)
                      batch_rtg0_list.append(trajectory['rewards'].sum().item())

        # Handle remaining rollouts if num_rollouts not divisible by num_models
        # (skipped entirely under the synthetic-expert diagnostic above --
        # that branch already generated exactly num_rollouts trajectories).
        if _diag_synth_xstar is None:
          remaining_rollouts = num_rollouts % len(self.gp_ensemble)
          for gp_idx in range(remaining_rollouts):
            if (self.use_roi_state or self.use_roi_std_quantiles or self.use_roi_sigma_iqr) and not use_joint_rtg:
                _, roi_candidates_for_rollout = self._optimize_acquisition(
                    gp_idx, self.data_y.max(), for_rollout=True
                )
            else:
                roi_candidates_for_rollout = None
            current_state = self._extract_state(
                self.data_x, self.data_y, self.data_x.shape[0],
                roi_candidates=roi_candidates_for_rollout,
            )
            if use_joint_rtg:
                trajectory = self._simulate_trajectory_joint(gp_idx, current_state, rollout_length)
                trajectories.append(trajectory)
                batch_rtg0_list.append(trajectory['joint_rtg'][0].item())
            else:
                trajectory = self._simulate_trajectory(
                    gp_idx, current_state, rollout_length,
                    initial_roi_candidates=roi_candidates_for_rollout,
                )
                trajectories.append(trajectory)
                batch_rtg0_list.append(trajectory['rewards'].sum().item())

        self._last_rollout_sim_time = time.perf_counter() - _rollout_sim_t_start

        # Rollout action diversity: mean pairwise distance among each
        # rollout's OWN action sequence (its rollout_length steps),
        # averaged across the batch -- a WITHIN-trajectory metric, not
        # across-rollout. Purpose: rollout_acq_function="rotate" is meant to
        # diversify the DT's training data by cycling the acquisition
        # function used AT EACH STEP WITHIN a rollout ([ei,ucb,pi,mes][step
        # % 4]) -- this metric tests that mechanism directly.
        #
        # Deliberately NOT the across-rollout diversity of just the first
        # action (an earlier version of this metric): under "rotate", step 0
        # always resolves to acq_names[0 % 4] == "ei" for EVERY rollout (step
        # starts at 0), so a first-action-only metric is structurally
        # identical between "rotate" and a fixed acq_name="ei" variant and
        # can never detect rotate's effect at all -- confirmed empirically
        # (bit-identical values on the very first rollout step, sanity_check_
        # mes_switching_v2.py's check 2b) before switching to this
        # within-trajectory formulation, which is sensitive to steps 1+ where
        # rotate actually diverges from a fixed acquisition function.
        #
        # None when rollout_length==1 (a single-step trajectory has no
        # internal diversity to measure) or there are no trajectories at all.
        traj_diversities = []
        for traj in trajectories:
            acts = traj['actions'] # [rollout_length, D]
            if acts.shape[0] > 1:
                pairwise_dists = torch.cdist(acts, acts) # [rollout_length, rollout_length]
                off_diag_mask = ~torch.eye(acts.shape[0], dtype=torch.bool, device=acts.device)
                traj_diversities.append(pairwise_dists[off_diag_mask].mean().item())
        self._last_rollout_action_diversity = float(np.mean(traj_diversities)) if traj_diversities else None

        # 1b. Reward diagnostics + RTG target for this iteration (used for logging
        # and for the DT's inference-time target below). current_real_state is
        # computed once here (it only depends on self.data_x/self.data_y, which
        # don't change during rollout simulation) and reused below both for the
        # quantile sliding-window inference and the actual action proposal --
        # previously this was recomputed a second time for no reason.
        #
        # roi_candidates_for_state: NEW, separate from every rollout's own
        # (ephemeral, per-rollout) roi_candidates -- none of those survive past
        # their rollout, so without this call current_real_state would have no
        # roi_candidates at all and get zero-padded ROI-state slots, unlike the
        # rollout-training states which DO have real ones. gp_idx=0 (any single
        # member suffices for a domain-spanning candidate set) and
        # for_rollout=False (full optimization budget, since this feeds the
        # real inference-time decision, not a cheap simulated rollout step).
        if self.use_roi_state or self.use_roi_std_quantiles or self.use_roi_sigma_iqr:
            _, roi_candidates_for_state = self._optimize_acquisition(0, self.data_y.max(), for_rollout=False)
        else:
            roi_candidates_for_state = None
        current_real_state = self._extract_state(
            self.data_x, self.data_y, self.data_x.shape[0], roi_candidates=roi_candidates_for_state
        )
        all_step_rewards = [r for traj in trajectories for r in traj['rewards'].tolist()]
        if all_step_rewards:
            mean_reward = float(np.mean(all_step_rewards))
            zero_frac = float(np.mean([1.0 if r == 0 else 0.0 for r in all_step_rewards]))
        else:
            mean_reward = 0.0
            zero_frac = 0.0
        rtg_target, batch_max_rtg = self._compute_rtg_target(batch_rtg0_list, current_real_state)
        # neg_rtg_frac: fraction of this iteration's batch_rtg0_list (one
        # RTG[0] per rollout) that's negative -- diagnostic only for
        # joint/entropy_joint schemas, where RTG[0] can legitimately be
        # negative (a Gumbel-fit artifact at finite K, or genuine negative
        # differential entropy for entropy_joint -- see _simulate_trajectory_joint).
        # None for non-joint schemas, where batch_rtg0_list holds summed
        # per-step rewards instead (not comparable / not meaningful to call
        # "negative RTG" for those).
        neg_rtg_frac = (
            float(np.mean([1.0 if v < 0 else 0.0 for v in batch_rtg0_list]))
            if use_joint_rtg and batch_rtg0_list else None
        )
        self._last_iter_diagnostics = {
            "mean_reward": mean_reward,
            "zero_frac": zero_frac,
            "rtg_target": rtg_target,
            "batch_max_rtg": batch_max_rtg,
            "running_max_rtg": self.running_max_rtg,
            "neg_rtg_frac": neg_rtg_frac,
            # Per-phase wall-clock breakdown (cost audit) + rollout action
            # diversity (rotate's diversification mechanism, tested directly).
            # gp_refit_time/rollout_sim_time/rollout_action_diversity are
            # already known at this point; dt_train_time/real_query_time are
            # appended below once those phases actually run.
            "gp_refit_time": self._last_gp_refit_time,
            "rollout_sim_time": self._last_rollout_sim_time,
            "rollout_action_diversity": self._last_rollout_action_diversity,
        }
        if self.decision_transformer.use_quantile_rtg and self._last_q_hat is not None:
            alpha_levels = self.decision_transformer.quantile_head.alpha_levels.tolist()
            q_hat = self._last_q_hat
            self._last_iter_diagnostics["Q_hat_inference"] = q_hat
            self._last_iter_diagnostics["quantile_spread"] = q_hat[-1] - q_hat[0]
            self._last_iter_diagnostics["calibration"] = [
                float(np.mean([1.0 if r < q_hat[j] else 0.0 for r in batch_rtg0_list]))
                for j in range(len(alpha_levels))
            ] if batch_rtg0_list else [float('nan')] * len(alpha_levels)
        self._pending_log = True

        # 2. Train Decision Transformer on simulated trajectories
        _dt_train_t_start = time.perf_counter()
        if self.verbose: print("Training Decision Transformer...")
        self._train_decision_transformer(trajectories)
        if self.decision_transformer.use_quantile_rtg:
            self._last_iter_diagnostics.update(self._last_train_diagnostics)
        self._last_dt_train_time = time.perf_counter() - _dt_train_t_start
        self._last_iter_diagnostics["dt_train_time"] = self._last_dt_train_time

        # 3. Use trained Decision Transformer to propose the next candidate
        _real_query_t_start = time.perf_counter()
        if self.verbose: print("Proposing candidate using Decision Transformer...")
        self.decision_transformer.eval() # Set to evaluation mode

        with torch.no_grad():
            # Transformer expects sequence input. Create sequence of length 1.
            state_seq = current_real_state.unsqueeze(0).unsqueeze(0) # [1, 1, StateDim]

            dummy_action = torch.zeros((1, 1, self.bo_config.input_dim), device=self.device, dtype=self.dtype)
            target_rtg = torch.tensor([[rtg_target]], device=self.device, dtype=self.dtype) # [1, 1], per self.rtg_schema
            dummy_timestep = torch.zeros((1, 1), device=self.device, dtype=torch.long) # It should dummy as there is only one decision to be make.

            # Get predicted action
            predicted_action_seq = self.decision_transformer(
                 state_seq, dummy_action, target_rtg, dummy_timestep, attention_mask=None
                 ) # Output: [1, 1, ActionDim]

            # Extract the predicted action for the current step
            next_action = predicted_action_seq.squeeze(0).squeeze(0) # [ActionDim]

            # 4. Ensure the proposed action is within bounds
            domain_min, domain_max = self.bounds[0], self.bounds[1]
            next_action = torch.min(torch.max(next_action, domain_min), domain_max)

            # UCB refinement (warm-started from the DT's own proposal):
            # isolates whether location quality is the bottleneck when
            # fidelity is not a factor (SF-DRO always queries HF). Off by
            # default -- bit-for-bit unchanged when use_gp_refinement is
            # unset/False.
            x_dt = next_action.detach().clone()
            if getattr(self.config, 'use_gp_refinement', False):
                _refine_variant = getattr(self.config, 'gp_refinement_variant', 'single')
                if _refine_variant == 'ensemble':
                    next_action = self._refine_proposal_ucb_ensemble(next_action)
                elif _refine_variant == 'twostage':
                    next_action = self._refine_proposal_ucb_twostage(next_action)
                elif _refine_variant == 'restarts':
                    next_action = self._refine_proposal_ucb_restarts(next_action)
                else:
                    next_action = self._refine_proposal_ucb(next_action)

            # Decisive diagnostic: where does the DT actually propose at
            # REAL inference, relative to the true optimum -- and, when UCB
            # refinement is on, how much does refinement move it and does
            # it actually raise UCB? Gated behind self._diag_xstar (unset/
            # None by default -- never runs/prints in the normal pipeline).
            # t = len(self.real_history_states) BEFORE this iteration's own
            # append below, matching this file's existing "current real
            # iteration index" convention. Per-iteration records are stored
            # in self._diag_refine_log (list of dicts) for the worker
            # script to zip with iteration_log_history's regret/improved
            # afterward -- regret for this iteration isn't known yet here
            # (it's computed after the real y-evaluation, later in the
            # caller's loop).
            _diag_xstar = getattr(self, '_diag_xstar', None)
            if _diag_xstar is not None:
                _t = len(self.real_history_states)
                _xstar_cpu = _diag_xstar.cpu()
                x_dt_dist = (x_dt.cpu() - _xstar_cpu).norm().item()
                x_refined_dist = (next_action.detach().cpu() - _xstar_cpu).norm().item()
                with torch.no_grad():
                    _model = self.gp_ensemble[0]['model']
                    ucb_beta = getattr(self.config, 'gp_refinement_beta', 2.0)
                    _post_dt = _model.posterior(x_dt.reshape(1, -1))
                    ucb_at_dt = (_post_dt.mean + ucb_beta * _post_dt.variance.clamp_min(1e-8).sqrt()).item()
                    _post_ref = _model.posterior(next_action.detach().reshape(1, -1))
                    ucb_at_refined = (_post_ref.mean + ucb_beta * _post_ref.variance.clamp_min(1e-8).sqrt()).item()
                closer = x_refined_dist < x_dt_dist
                if not hasattr(self, '_diag_refine_log'):
                    self._diag_refine_log = []
                self._diag_refine_log.append({
                    'iter': _t, 'x_dt_dist': x_dt_dist, 'x_refined_dist': x_refined_dist,
                    'ucb_at_dt': ucb_at_dt, 'ucb_at_refined': ucb_at_refined, 'closer': closer,
                })
                _n_closer = sum(1 for r in self._diag_refine_log if r['closer'])
                _frac_closer = _n_closer / len(self._diag_refine_log)
                print(f"iter {_t}: x_DT_dist_to_xstar={x_dt_dist:.4f} "
                      f"x_refined_dist_to_xstar={x_refined_dist:.4f} "
                      f"ucb_at_x_DT={ucb_at_dt:.4f} ucb_at_x_refined={ucb_at_refined:.4f} "
                      f"frac_closer_running={_frac_closer:.3f}", flush=True)

        self._last_real_query_time = time.perf_counter() - _real_query_t_start
        self._last_iter_diagnostics["real_query_time"] = self._last_real_query_time

        # Record this real iteration's (state, action) for future quantile-schema
        # sliding-window inference (see _compute_quantile_rtg_target).
        self.real_history_states.append(current_real_state.detach().clone())
        self.real_history_actions.append(next_action.detach().clone())

        # Track 1 diagnostics -- cheap, two posterior calls. Purely for
        # analysis (Stage 1/2 ROI-state experiments); no effect on the
        # proposed action itself.
        with torch.no_grad():
            model = self.gp_ensemble[0]['model']

            post_proposed = model.posterior(next_action.reshape(1, -1))
            mu_proposed = post_proposed.mean.item()
            sigma_proposed = post_proposed.variance.sqrt().item()

            if self.known_optimal_x is not None:
                x_opt = torch.tensor(
                    self.known_optimal_x, device=self.device, dtype=self.dtype
                ).reshape(1, -1)
                mu_at_true_opt = model.posterior(x_opt).mean.item()
            else:
                mu_at_true_opt = float('nan')

            # Corner proximity: distance to the domain_min corner, in
            # normalized [0,1]^d coordinates, divided by the normalized
            # diameter sqrt(d). Domain-generic (previously hardcoded to the
            # RAW-scale origin [0,...,0], which only happened to coincide with
            # the domain_min corner for [0,1]^d benchmarks like Hartmann_6D --
            # for a domain like Ackley_5D's [-5,5]^5, raw [0,...,0] is the
            # domain CENTER, not a corner, which would have silently measured
            # the wrong thing).
            domain_min_diag, domain_max_diag = self.bounds[0], self.bounds[1]
            domain_range_diag = (domain_max_diag - domain_min_diag).clamp_min(1e-12)
            next_action_norm = (next_action.reshape(-1) - domain_min_diag) / domain_range_diag
            corner_norm = torch.zeros_like(next_action_norm)
            corner_proximity = (
                1.0 - torch.norm(next_action_norm - corner_norm).item()
                / math.sqrt(self.bo_config.input_dim)
            )
            # corner_proximity = 1.0 means next_action IS the domain_min corner
            # corner_proximity = 0.0 means next_action is maximally far

        self._last_diagnostics = {
            'mu_proposed': mu_proposed,
            'sigma_proposed': sigma_proposed,
            'mu_at_true_opt': mu_at_true_opt,
            'corner_proximity': corner_proximity,
        }

        # Return shape [1, input_dim]
        return next_action.unsqueeze(0)


@hydra.main(config_path='../../config', config_name="experiment", version_base=None)
def main(cfg: DictConfig):
    # --- Configuration Setup (Common for all trials) ---
    input_dim = cfg.test_function.input_dim
    domain_min = cfg.test_function.domain_min
    domain_max = cfg.test_function.domain_max
    if isinstance(domain_min, (int, float)):
        domain_min = [float(domain_min)] * input_dim
        domain_max = [float(domain_max)] * input_dim
    else:
        domain_min = [float(b) for b in domain_min]
        domain_max = [float(b) for b in domain_max]
    bounds_list = list(zip(domain_min, domain_max))

    if cfg.test_function.name == "Ackley":
        objective_function = Ackley(dim=input_dim, bounds=bounds_list, negate=True)
    elif cfg.test_function.name == "Rosenbrock":
        objective_function = Rosenbrock(dim=input_dim, bounds=bounds_list, negate=True)
    elif cfg.test_function.name == "Levy":
        objective_function = Levy(dim=input_dim, bounds=bounds_list, negate=True)
    else:
        raise ValueError(f"Unsupported objective function specified: {cfg.experiment.objective}") # Corrected cfg path

    # Override the cfg.method.bo parameters
    cfg.method.bo.domain_min = domain_min # Use processed list
    cfg.method.bo.domain_max = domain_max # Use processed list
    cfg.method.bo.input_dim = input_dim
    cfg.method.bo.initial_points = cfg.experiment_params.initial_points
    cfg.method.bo.objective = cfg.experiment_params.objective 
    cfg.method.bo.max_iterations = cfg.experiment_params.max_iterations


    print("--- Experiment Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------------------")

    n_trials = cfg.experiment_params.num_trials
    all_trials_results = []
    all_trials_all_y = []
    all_trials_best_y = []

    if n_trials <= 0:
        print("Warning: cfg.experiment_params.n_trials is <= 0. No trials will be run.")
        return None # Or raise error

    cfg.method.name = 'dro'
    save_prefix = f"{cfg.method.name}_{cfg.test_function.name}"
    

    # --- Run Multiple Trials ---
    iterator = tqdm(range(n_trials), desc="Trials", unit="trial")
    for trial_idx in iterator:
        # Set up the optimization problem for this trial
        # Re-instantiate to ensure independent runs if the optimizer has state
        cfg.method.seed = cfg.experiment_params.seed_start + trial_idx
        dro = DirectRegretOptimization(cfg.method, objective_function)

        # Run optimization
        result = dro.run_optimization()
        all_trials_results.append(result)
        all_trials_all_y.append(result['all_y'])
        all_trials_best_y.append(result['best_y'])

        print(f"\nTrial {trial_idx + 1} Complete!")
        print(f"Best solution found: {result['best_x']}")
        print(f"Best objective value: {result['best_y']}")

        # --- Save Individual Trial Results ---
        trial_save_dir = cfg.save_dir if hasattr(cfg, 'save_dir') else "."
        os.makedirs(trial_save_dir, exist_ok=True) # Ensure directory exists

        # Save individual optimization progress plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(result['all_y'])), result['all_y'], marker='o', linestyle='-') # Use plot for clarity
        plt.xlabel('Iteration (incl. initial points)')
        plt.ylabel('Objective Value')
        plt.title(f'Optimization Progress (Trial {trial_idx + 1})')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(trial_save_dir, f"{save_prefix}_optimization_progress_trial_{trial_idx}.png"))
        plt.close() # Close plot to free memory

        # Save individual contour plot (if applicable)
        if cfg.method.bo.input_dim == 2:
            plt.figure(figsize=(10, 8))
            # Create grid for contour
            x1_lin = np.linspace(bounds_list[0][0], bounds_list[0][1], 100)
            x2_lin = np.linspace(bounds_list[1][0], bounds_list[1][1], 100)
            X1, X2 = np.meshgrid(x1_lin, x2_lin)
            Z = np.zeros_like(X1)

            # Evaluate objective on grid
            grid_points = torch.tensor(np.stack([X1.ravel(), X2.ravel()], axis=-1), dtype=torch.float32)
            with torch.no_grad(): # Ensure no gradient tracking for plotting
                 Z = objective_function(grid_points).reshape(X1.shape).numpy()


            plt.contourf(X1, X2, Z, 50, cmap='viridis')
            plt.colorbar(label='Objective Value')

            # Plot sampled points for this trial
            all_x_trial = result['all_x']
            plt.scatter(
                all_x_trial[:, 0],
                all_x_trial[:, 1],
                c='red',
                marker='x',
                s=50,
                label='Sampled Points'
            )

            # Highlight best point for this trial
            best_x_trial = result['best_x']
            plt.scatter(
                best_x_trial[0],
                best_x_trial[1],
                c='yellow',
                marker='*',
                s=200,
                label='Best Point Found',
                edgecolors='black'
            )

            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.xlim(bounds_list[0])
            plt.ylim(bounds_list[1])
            plt.title(f'Objective Contour with Sampled Points (Trial {trial_idx + 1})')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(trial_save_dir, f"contour_plot_trial_{trial_idx}.png"))
            plt.close() # Close plot

        # Save individual trial data
        np.savez(
            os.path.join(trial_save_dir, f"results_trial_{trial_idx}.npz"),
            best_x=result['best_x'],
            best_y=result['best_y'],
            all_x=result['all_x'],
            all_y=result['all_y']
        )

    # --- Aggregate and Plot Results ---
    print("\n--- Aggregating Results Across Trials ---")

    max_len = max(len(y) for y in all_trials_all_y)
    num_iterations = cfg.method.bo.initial_points + cfg.method.bo.max_iterations
    if any(len(y) != num_iterations for y in all_trials_all_y):
         print(f"Warning: Not all trials completed {num_iterations} evaluations. Aggregation might be inaccurate.")
         # Simple approach: use the minimum length across trials for aggregation
         min_len = min(len(y) for y in all_trials_all_y)
         all_trials_all_y_padded = np.array([y[:min_len] for y in all_trials_all_y])
    else:
         all_trials_all_y_padded = np.array(all_trials_all_y)


    # Calculate mean and standard deviation (or standard error)
    mean_all_y = np.mean(all_trials_all_y_padded, axis=0)
    std_all_y = np.std(all_trials_all_y_padded, axis=0)
    stderr_all_y = std_all_y / np.sqrt(n_trials)

    iterations = np.arange(len(mean_all_y)) # X-axis for the plot

    # Plot Aggregate Results (Mean +/- Std Dev or Std Error)
    plt.figure(figsize=(12, 7))
    plt.plot(iterations, mean_all_y, label='Mean Objective Value', color='blue')
    plt.fill_between(
        iterations,
        mean_all_y - std_all_y, # Or use stderr_all_y
        mean_all_y + std_all_y, # Or use stderr_all_y
        color='blue',
        alpha=0.2,
        label='Mean +/- 1 Std Dev' # Adjust label if using stderr
    )
    plt.xlabel('Iteration (incl. initial points)')
    plt.ylabel('Objective Value')
    plt.title(f'Aggregate Optimization Progress ({n_trials} Trials)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    aggregate_plot_path = os.path.join(trial_save_dir, f"{save_prefix}_aggregate_optimization_progress.png")
    plt.savefig(aggregate_plot_path)
    print(f"Saved aggregate plot to: {aggregate_plot_path}")
    plt.close()

    # --- Save Aggregate Results ---
    aggregate_save_path = os.path.join(trial_save_dir, f"{save_prefix}_aggregate_results.npz")
    np.savez(
        aggregate_save_path,
        mean_all_y=mean_all_y,
        std_all_y=std_all_y,
        stderr_all_y = stderr_all_y,
        all_trials_best_y=np.array(all_trials_best_y), # Store best y from each trial
        iterations = iterations
        # Optionally save all_trials_all_y_padded if needed, but it can be large
    )
    print(f"Saved aggregate results to: {aggregate_save_path}")

    # Find the overall best result across all trials
    overall_best_trial_idx = np.argmax(all_trials_best_y)
    overall_best_y = all_trials_best_y[overall_best_trial_idx]
    overall_best_x = all_trials_results[overall_best_trial_idx]['best_x']

    print("\n--- Overall Best Result Across All Trials ---")
    print(f"Found in Trial: {overall_best_trial_idx + 1}")
    print(f"Best solution (X): {overall_best_x}")
    print(f"Best objective value (Y): {overall_best_y}")
    print("---------------------------------------------")


    return all_trials_results # Return list of dictionaries containing results for each trial

if __name__ == "__main__":

    main()