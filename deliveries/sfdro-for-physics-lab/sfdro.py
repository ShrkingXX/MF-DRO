"""
sfdro.py -- ask/tell wrapper around SF-DRO (single-fidelity Direct Regret
Optimization) for experiments that are run by hand.

The optimizer in src/policy/dro.py is written as a *closed* loop: you hand it a
callable `objective_function` and it calls that function itself, once per
iteration, inside `run_optimization()`. That is the right shape for a benchmark
and the wrong shape for a lab, where the "objective function" is a cell that
takes a day to cycle.

This file exposes the same algorithm as an *open* loop instead:

    opt = SFDRO(bounds=...)      # define the search space
    opt.initialize(X_init, y_init)   # (1) seed with data you already have
    x = opt.ask()                    #     what to measure next
    ...run the experiment...
    opt.tell(x, y)                   # (2) report the measured value

Nothing here re-implements the method. `ask` and `tell` are thin forwards to
the very same three internal calls that `BaseBayesianOptimizer.run_optimization`
makes each iteration (see src/policy/base.py, step 3a/3b/3d) --
`_update_models()`, `_propose_next_candidate()`, `_update_data_and_best()` --
with the objective evaluation left out so you can supply it yourself. The
optimizer is constructed with `objective_function=None`; on this path it is
never called.

Sign convention: SF-DRO's tested path always maximizes (every benchmark in the
paper is run with negate=True). This wrapper keeps that path and negates for
you when `minimize=True`, so you always pass in and read back values in your
own units and sign.
"""
import numpy as np
import torch
from omegaconf import OmegaConf

from src.policy.dro import DirectRegretOptimization

DEFAULT_DTYPE = torch.float64


def build_config(input_dim, domain_min, domain_max, seed=42, n_initial=5,
                 max_iterations=50,
                 # --- surrogate: the GP ensemble ---
                 gp_num_models=10, gp_kernel="rbf", gp_ard=True,
                 lengthscale_min=0.1, lengthscale_max=10.0, noise_constraint=1e-2,
                 # --- inner loop: simulated rollouts ---
                 rollouts_per_iter=200, rollout_length=8, rollout_acq_function="ei",
                 # --- the Decision Transformer ---
                 dt_hidden=64, dt_layers=2, dt_heads=4, dt_lr=1e-3, dt_epochs=100,
                 verbose=False):
    """
    The SF-DRO hyperparameters used for the paper's single-fidelity runs
    (experiments/h59-sfdro-baseline). Defaults here match that configuration;
    see the README for which knobs are worth turning and which are not.
    """
    return OmegaConf.create({
        "seed": int(seed),
        "save_dir": None,
        "verbose": verbose,
        "device": "cpu",
        "name": "dro",
        # Left unset on purpose: with exp_name/benchmark_name/variant_name all
        # absent, dro.py skips its checkpoint/log-file writing entirely, so
        # nothing needs checkpoint.setup_dirs() and nothing is written to disk.
        "use_mes_reward": False,
        "rtg_schema": "fixed",
        "alpha_floor": 0.5,
        "alpha_inference": None,
        "lambda_rtg": 1.0,
        "rtg_warmup": 3,
        # Only used to print a "regret" diagnostic. For a real experiment the
        # true optimum is unknown, so this number is meaningless -- ignore it
        # and read `opt.best` instead.
        "known_optimal_value": 0.0,
        "gp": {
            "kernel": gp_kernel, "noise_constraint": noise_constraint,
            "lengthscale_min": lengthscale_min, "lengthscale_max": lengthscale_max,
            "num_models": gp_num_models, "verbose": False, "retrain": False,
            "ard": gp_ard,
        },
        "acquisition": {
            "function": rollout_acq_function, "kappa": 2.0, "ucb_lcb_kappa": 6.0,
            "xi": 0.01, "early_stop_threshold": 1e-4, "constrain_ucb_lcb": True,
        },
        "transformer": {
            "hidden_size": dt_hidden, "num_layers": dt_layers, "num_heads": dt_heads,
            "dropout": 0.1, "lr": dt_lr, "weight_decay": 1e-5, "batch_size": 32,
            "num_epochs": dt_epochs, "max_seq_length": 20,
            "use_awr": False, "awr_temperature": None,
        },
        "simulation": {
            "num_rollouts": rollouts_per_iter, "max_rollout_length": rollout_length,
            "early_stop": False, "early_stop_threshold": 1e-4, "verbose": False,
        },
        "bo": {
            "max_iterations": max_iterations,
            "input_dim": input_dim,
            "domain_min": [float(v) for v in domain_min],
            "domain_max": [float(v) for v in domain_max],
            "initial_points": n_initial,
            "objective": "maximize",
            "initial_sampling_method": "lhs",
        },
    })


class SFDRO:
    """
    Ask/tell interface to DirectRegretOptimization.

    bounds:   (lo, hi) as two sequences of length d, or a (2, d) array --
              the physical limits of every knob you can set.
    minimize: True if lower measured values are better (e.g. internal
              resistance); False if higher is better (e.g. capacity).
    """

    def __init__(self, bounds, minimize=False, seed=42, **hyper):
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape[0] != 2:
            raise ValueError(f"bounds must be shape (2, d); got {bounds.shape}")
        self.lo, self.hi = bounds[0], bounds[1]
        self.dim = int(bounds.shape[1])
        self.minimize = bool(minimize)
        self._sign = -1.0 if minimize else 1.0

        self.config = build_config(self.dim, self.lo, self.hi, seed=seed, **hyper)
        # objective_function=None: SF-DRO only ever calls it from
        # sample_initial_points / _make_synthetic_expert_trajectory, and this
        # wrapper uses neither.
        self._opt = DirectRegretOptimization(self.config, None)
        self._initialized = False
        self._stale = True   # True when data has changed since the last GP fit

    # ---------------------------------------------------------------- (1) init
    def initialize(self, X, y):
        """
        Seed the optimizer with the measurements you already have and fit the
        GP ensemble to them. X is (n, d) in your own physical units; y is (n,)
        of measured values in your own sign.

        This replaces SF-DRO's built-in Latin-hypercube initial design, which
        would insist on calling the objective itself. Everything downstream --
        the GP ensemble, the simulated rollouts, the Decision Transformer --
        sees your points and nothing else.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]} values")
        if X.shape[1] != self.dim:
            raise ValueError(f"X has {X.shape[1]} columns but bounds define {self.dim}")
        if X.shape[0] < 2:
            raise ValueError("need at least 2 initial points to fit a GP")
        self._check_in_bounds(X)

        # Same call the base class uses for its own initial design (base.py,
        # sample_initial_points): it appends to data_x/data_y and updates the
        # running best. Per-iteration logging is gated off here because
        # _pending_log is only set by _propose_next_candidate.
        for xi, yi in zip(X, y):
            self._opt._update_data_and_best(
                torch.tensor(xi, dtype=DEFAULT_DTYPE).unsqueeze(0), self._sign * float(yi)
            )
        # Builds the ensemble slots (one GP per starting lengthscale on a
        # linspace grid) and fits them all to the data just added.
        self._opt._initialize_models()
        self._initialized = True
        self._stale = False
        return self

    # ----------------------------------------------------------------- (2) ask
    def ask(self):
        """
        Return the next point to measure, as a (d,) array in your units.

        This is one full SF-DRO iteration of *thinking*: refit the GP ensemble
        if new data arrived, simulate `rollouts_per_iter` optimization
        trajectories inside those GPs, train the Decision Transformer on them
        to predict the action that minimizes final regret, and read off its
        proposal. It is the expensive step -- minutes, not seconds.
        """
        self._require_init()
        if self._stale:
            self._opt._update_models()      # base.py step 3a
            self._stale = False
        x = self._opt._propose_next_candidate()   # base.py step 3b
        return x.squeeze(0).detach().cpu().numpy()

    # ---------------------------------------------------------------- (2) tell
    def tell(self, x, y):
        """
        Record a measured value and mark the surrogate as needing a refit.
        `x` is (d,) -- normally what `ask()` just returned, but any point is
        accepted, so you can also feed in results from experiments you ran for
        other reasons.
        """
        self._require_init()
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != self.dim:
            raise ValueError(f"x has length {x.shape[0]} but bounds define {self.dim}")
        self._check_in_bounds(x[None, :])
        # base.py step 3d.
        self._opt._update_data_and_best(
            torch.tensor(x, dtype=DEFAULT_DTYPE).unsqueeze(0), self._sign * float(y)
        )
        self._stale = True
        return self

    # ------------------------------------------------------------- inspection
    @property
    def best(self):
        """(x, y) of the best measurement so far, in your units and sign."""
        self._require_init()
        y = self._opt.best_observed_value
        y = y.item() if torch.is_tensor(y) else float(y)
        return np.asarray(self._opt.best_x, dtype=float), self._sign * y

    @property
    def history(self):
        """(X, y) of everything told to the optimizer, in your units and sign."""
        X = self._opt.data_x.detach().cpu().numpy()
        y = self._sign * self._opt.data_y.detach().cpu().numpy()
        return X, y

    def __len__(self):
        return int(self._opt.data_x.shape[0])

    # ------------------------------------------------------------------ guards
    def _require_init(self):
        if not self._initialized:
            raise RuntimeError("call initialize(X, y) before ask()/tell()")

    def _check_in_bounds(self, X):
        below = X < self.lo - 1e-9
        above = X > self.hi + 1e-9
        if below.any() or above.any():
            bad = np.unique(np.where(below | above)[1])
            raise ValueError(
                f"points fall outside bounds in dimension(s) {bad.tolist()}. "
                "SF-DRO only ever proposes inside the box you declared; widen "
                "`bounds` if these measurements are legitimate."
            )
