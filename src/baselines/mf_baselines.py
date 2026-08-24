"""
Multi-fidelity baselines for comparison against MF-DRO: MF-GP-UCB
(Kandasamy et al. 2016) and MF-MI-Greedy (Song, Chen & Yue 2019).

SOURCE ACCESS NOTE: neither paper's exact pseudocode text was directly
readable in this session (both PDFs render as compressed image streams
under the available tooling, and no poppler/pdftoppm is installed to
extract them). MF-GP-UCB's beta_t is the standard Srinivas et al. (2010)
GP-UCB confidence bound, which is independently well-established and does
not depend on reading Kandasamy 2016 itself. MF-MI-Greedy's information-gain
formula was re-derived here from first principles (mutual information
between a noisy observation and a Gaussian latent under the stated additive
model) and matches the closed form described in the task -- see
_information_gain's docstring. The two-phase alternation and three stopping
conditions for Explore-LF are implemented per the task's own description;
where that description leaves an implementation detail unspecified (the
additive-model GP fitting procedure, and how repeated candidate selection
within one Explore-LF call is avoided), a concrete, clearly-documented
choice was made rather than left ambiguous.

MultiFidelityBenchmark below does not correspond to any existing class in
this codebase (confirmed: no benchmarks/mf_benchmarks.py, no
benchmarks/ directory, no MultiFidelityBenchmark class anywhere) -- it is a
small adapter built here specifically because both baseline classes need a
single object exposing both fidelities together, whereas the existing
registry (benchmarks.py's BENCHMARKS dict / get_benchmark()) stores each
fidelity as a separate entry (e.g. "Currin_2D_HF", "Currin_2D_LF").
"""
import math
import warnings

import torch
import numpy as np
from scipy.stats.qmc import LatinHypercube

from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.exceptions import InputDataWarning
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.analytic import LogExpectedImprovement

from benchmarks import get_benchmark
from gumbel_thompson import thompson_sample_y_star
from src.models.ko_gp import KennedyOHaganGP
from src.baselines.greedy_mf import GreedyMFBase, cost_normalized_argmax
from src.utils.init_design import make_initial_design
from src.policy.mf_dro import (
    _build_hf_proxy_model, _compute_mes_hf_vectorized, _compute_mes_lf_vectorized,
)

DEFAULT_DTYPE = torch.float64
_NOISE_LB = 1e-4
_CANDIDATE_POOL = 1000


class MultiFidelityBenchmark:
    """
    Adapter bundling a benchmark's HF and LF registry entries (from
    benchmarks.py's BENCHMARKS dict) into one object. name: e.g. "Currin_2D",
    "Hartmann_6D" -- looks up "{name}_HF" and "{name}_LF".
    """

    def __init__(self, name):
        hf_spec = get_benchmark(f"{name}_HF")
        lf_spec = get_benchmark(f"{name}_LF")
        self.name = name
        self.dim = hf_spec["dim"]
        # [2, d] BoTorch convention (row 0 = lower, row 1 = upper) -- matches
        # every other bounds consumer in this codebase (ko.fit()'s Normalize,
        # dro.py's self.bounds[0]/self.bounds[1]).
        self.bounds = torch.tensor(
            [hf_spec["domain_min"], hf_spec["domain_max"]], dtype=DEFAULT_DTYPE
        )
        self.f_hf = hf_spec["make_objective"]()
        self.f_lf = lf_spec["make_objective"]()
        self.c_H = hf_spec["cost"]
        self.c_L = lf_spec["cost"]
        self.known_optimal_value_hf = hf_spec["known_optimal_value"]

    def evaluate(self, x, fidelity):
        """x: [d] or [1,d] tensor. fidelity: 'L' or 'H'. Returns scalar float."""
        x_in = x.unsqueeze(0) if x.dim() == 1 else x
        f = self.f_hf if fidelity == 'H' else self.f_lf
        return f(x_in.to(DEFAULT_DTYPE)).reshape(-1)[0].item()


def _sample_candidates(bounds, d, n):
    return bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n, d, dtype=DEFAULT_DTYPE)


def _lhs_init_points(bounds, d, n, seed, seed_offset, use_sequential_init=False):
    """
    Thin wrapper over src/utils/init_design.py's dispatcher, kept because
    every optimizer class in this module already calls it by this name. The
    (seed, seed_offset) convention (offset=0 for HF, offset=1 for LF) is
    shared with DirectMFRegretOptimization._sample_initial_points and
    GreedyMFBase, so every method draws the identical initial design for a
    given (benchmark, seed) -- the "same initialization across methods"
    requirement -- under either design.
    """
    return make_initial_design(bounds, d, n, seed, seed_offset,
                               use_sequential_init=use_sequential_init)


def _build_gp(X, Y, bounds, d, train_iter=50, lr=0.1, noise_lb=_NOISE_LB):
    """
    Fresh SingleTaskGP on (X, Y), MLL-fit via Adam. Mirrors dro.py's
    _construct_gp_model / src/models/ko_gp.py's _build_gp: RBF+ARD kernel,
    ScaleKernel wrapper, GaussianLikelihood(noise>=noise_lb), Normalize+
    Standardize transforms so the kernel always sees a common [0,1]^d /
    N(0,1) scale regardless of the benchmark's raw domain/value range.
    """
    likelihood = GaussianLikelihood(noise_constraint=GreaterThan(noise_lb))
    covar_module = ScaleKernel(RBFKernel(ard_num_dims=d))

    train_x = X.to(dtype=DEFAULT_DTYPE)
    train_y = Y.to(dtype=DEFAULT_DTYPE).reshape(-1, 1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=InputDataWarning)
        model = SingleTaskGP(
            train_X=train_x, train_Y=train_y,
            likelihood=likelihood, covar_module=covar_module,
            input_transform=Normalize(d=d, bounds=bounds),
            outcome_transform=Standardize(m=1),
        )

    model.train()
    model.likelihood.train()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(train_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, model.train_targets)
        loss.backward()
        optimizer.step()

    model.eval()
    model.likelihood.eval()
    return model


def _rebuild_frozen(old_model, X, Y, bounds, d, noise_lb=_NOISE_LB):
    """
    Rebuild on (X, Y) reusing old_model's frozen kernel/likelihood
    hyperparameters (no MLL refit) -- same pattern as dro.py's
    _make_fantasy_model / ko_gp.py's _rebuild_frozen_gp, used here for
    MF-GP-UCB's "refit hyperparameters every 25 iterations" schedule: every
    OTHER iteration's GP still needs to include the newest observation, just
    without re-running MLL optimization.
    """
    new_likelihood = GaussianLikelihood(noise_constraint=GreaterThan(noise_lb))
    new_likelihood.noise = old_model.likelihood.noise.detach().clone()
    new_covar_module = ScaleKernel(RBFKernel(ard_num_dims=d))
    new_covar_module.load_state_dict(old_model.covar_module.state_dict())

    train_x = X.to(dtype=DEFAULT_DTYPE)
    train_y = Y.to(dtype=DEFAULT_DTYPE).reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=InputDataWarning)
        new_model = SingleTaskGP(
            train_X=train_x, train_Y=train_y,
            likelihood=new_likelihood, covar_module=new_covar_module,
            input_transform=Normalize(d=d, bounds=bounds),
            outcome_transform=Standardize(m=1),
        )
    new_model.eval()
    return new_model


def _gp_posterior_or_prior(gp, x):
    """
    (mean, std) at x. gp=None (no observations yet at this fidelity) falls
    back to the GP prior: zero mean, std=1 (ScaleKernel's default
    outputscale=1, matching the untrained-prior magnitude BoTorch/GPyTorch
    kernels default to).
    """
    x_in = x.unsqueeze(0) if x.dim() == 1 else x
    if gp is None:
        n = x_in.shape[0]
        return torch.zeros(n, dtype=DEFAULT_DTYPE), torch.ones(n, dtype=DEFAULT_DTYPE)
    with torch.no_grad():
        posterior = gp.posterior(x_in)
        mean = posterior.mean.reshape(-1)
        var = posterior.variance.clamp_min(1e-12).reshape(-1)
    return mean, var.sqrt()


# ════════════════════════════════════
# BASELINE 1: MF-GP-UCB (Kandasamy et al. 2016)
# ════════════════════════════════════

class MFGPUCBOptimizer:
    """
    Independent GPs per fidelity -- NOT the KO model. gp_L is fitted only on
    LF observations, gp_H only on HF observations; no information sharing
    between fidelities through the GP itself, only through the
    cost-normalized UCB acquisition (a cheap, uncertain LF query can still
    win the argmax over an expensive, more-certain HF query).

    NOTE: On Currin 2D (c_H=3) and Hartmann 6D (c_H=8), MF-GP-UCB degenerates
    to all-LF selection when the cost ratio exceeds the prior variance ratio
    of the two fidelity GPs. This is a known limitation of the
    cost-normalized UCB approach (Kandasamy et al. 2016) and is expected
    behavior on these benchmarks. The regret curve will plateau since HF is
    never queried. This serves as a useful baseline comparison showing that
    naive cost-normalized UCB is insufficient and motivates the MF-DRO
    learned policy.
    """

    def __init__(self, benchmark, n_initial=5, delta=0.1,
                 n_initial_hf=None, n_initial_lf=None, seed=0,
                 cost_budget=None, use_sequential_init=False):
        self.use_sequential_init = use_sequential_init
        self.benchmark = benchmark
        self.d = benchmark.dim
        self.bounds = benchmark.bounds
        self.n_initial = n_initial
        self.n_initial_hf = n_initial_hf if n_initial_hf is not None else n_initial
        self.n_initial_lf = n_initial_lf if n_initial_lf is not None else n_initial
        self.seed = seed
        self.cost_budget = cost_budget
        self.delta = delta
        self.refit_every = 25  # per the Matlab reference implementation

        self.gp_L = None
        self.gp_H = None
        self.data_L_x, self.data_L_y = [], []
        self.data_H_x, self.data_H_y = [], []
        self._iters_since_refit = {'L': 0, 'H': 0}

    def _compute_beta_t(self, t, cardinality=_CANDIDATE_POOL):
        """
        beta_t = 2 * log(|X| * t^2 * pi^2 / (6 * delta))  (Srinivas et al.
        2010 GP-UCB confidence bound; |X|=cardinality as a discrete
        approximation of the continuous domain).
        """
        return 2.0 * math.log(
            cardinality * (t ** 2) * (math.pi ** 2) / (6.0 * self.delta)
        )

    def _ucb(self, x, gp, beta_t):
        """UCB at x using given GP (None -> prior fallback)."""
        mu, sigma = _gp_posterior_or_prior(gp, x)
        return (mu + beta_t ** 0.5 * sigma).item()

    def _add_obs(self, x, y, fidelity):
        if fidelity == 'L':
            self.data_L_x.append(x)
            self.data_L_y.append(y)
        else:
            self.data_H_x.append(x)
            self.data_H_y.append(y)

    def _maybe_refit(self, fidelity):
        """
        Rebuild the given fidelity's GP on its current accumulated data.
        Every self.refit_every calls, re-run full MLL optimization
        (_build_gp); otherwise rebuild with frozen hyperparameters copied
        from the current model (_rebuild_frozen) -- the data always stays
        current, only kernel hyperparameter re-optimization is throttled.
        """
        xs, ys = (self.data_L_x, self.data_L_y) if fidelity == 'L' else (self.data_H_x, self.data_H_y)
        if not xs:
            return
        X = torch.stack(xs)
        Y = torch.tensor(ys, dtype=DEFAULT_DTYPE)
        current = self.gp_L if fidelity == 'L' else self.gp_H

        self._iters_since_refit[fidelity] += 1
        do_full_refit = current is None or self._iters_since_refit[fidelity] >= self.refit_every
        if do_full_refit:
            new_gp = _build_gp(X, Y, self.bounds, self.d)
            self._iters_since_refit[fidelity] = 0
        else:
            new_gp = _rebuild_frozen(current, X, Y, self.bounds, self.d)

        if fidelity == 'L':
            self.gp_L = new_gp
        else:
            self.gp_H = new_gp

    def select_next(self, t):
        """
        Select (x_t, ell_t) by:
            score(x, ell) = UCB_ell(x) / c_ell
            (x_t, ell_t) = argmax score over candidates x {L, H}
        If gp_L or gp_H is None (no observations yet), that fidelity's UCB
        falls back to the GP prior (zero mean) via _gp_posterior_or_prior.
        """
        beta_t = self._compute_beta_t(t)
        candidates = _sample_candidates(self.bounds, self.d, _CANDIDATE_POOL)

        ucb_L = torch.stack([
            torch.tensor(self._ucb(candidates[i], self.gp_L, beta_t))
            for i in range(candidates.shape[0])
        ])
        ucb_H = torch.stack([
            torch.tensor(self._ucb(candidates[i], self.gp_H, beta_t))
            for i in range(candidates.shape[0])
        ])
        score_L = ucb_L / self.benchmark.c_L
        score_H = ucb_H / self.benchmark.c_H

        scores = torch.stack([score_L, score_H], dim=1)  # [N, 2]
        flat_best = scores.reshape(-1).argmax()
        cand_idx = (flat_best // 2).item()
        fid_idx = (flat_best % 2).item()  # 0=L, 1=H

        return candidates[cand_idx], ('L' if fid_idx == 0 else 'H')

    def _sample_initial(self):
        X_lf = _lhs_init_points(self.bounds, self.d, self.n_initial_lf,
                                 self.seed, seed_offset=1,
                                 use_sequential_init=self.use_sequential_init)
        for x in X_lf:
            y = self.benchmark.evaluate(x, 'L')
            self._add_obs(x, y, 'L')
        X_hf = _lhs_init_points(self.bounds, self.d, self.n_initial_hf,
                                 self.seed, seed_offset=0,
                                 use_sequential_init=self.use_sequential_init)
        for x in X_hf:
            y = self.benchmark.evaluate(x, 'H')
            self._add_obs(x, y, 'H')
        self.initial_hf_values = list(self.data_H_y)
        self._maybe_refit('L')
        self._maybe_refit('H')

    def run(self, bo_iterations):
        """
        Standard BO loop. Returns regret curve and cost curve (POST-INIT --
        starts near 0, excludes initialization spending, so methods with
        different initialization sizes are comparable on the same x-axis),
        plus the fidelity selected at each iteration. Terminates early once
        post_init_cost reaches self.cost_budget (bo_iterations remains a
        safety cap only when cost_budget is set).
        """
        self._sample_initial()

        regret_curve, cost_curve, fidelities = [], [], []
        cumulative_cost = sum(self.benchmark.c_L for _ in self.data_L_y) \
            + sum(self.benchmark.c_H for _ in self.data_H_y)
        post_init_cost = 0.0
        budget = self.cost_budget if self.cost_budget is not None else float('inf')

        for t in range(1, bo_iterations + 1):
            if post_init_cost >= budget:
                break
            x, ell = self.select_next(t)
            y = self.benchmark.evaluate(x, ell)
            cost = self.benchmark.c_L if ell == 'L' else self.benchmark.c_H
            cumulative_cost += cost
            post_init_cost += cost
            self._add_obs(x, y, ell)
            self._maybe_refit(ell)

            best_hf = max(self.data_H_y) if self.data_H_y else float('-inf')
            # known_optimal_value_hf is on the RAW (pre-negation) scale
            # (benchmarks.py's established convention); best_hf comes from
            # data_H_y, populated by benchmark.evaluate() which calls the
            # NEGATED (maximization-ready) f_hf -- so best_hf must be
            # negated back before comparing, matching dro.py's own
            # maximize-mode regret convention (-best_observed - known_opt).
            regret = -best_hf - self.benchmark.known_optimal_value_hf
            regret_curve.append(regret)
            cost_curve.append(post_init_cost)
            fidelities.append(ell)

        return {
            'regret_curve': regret_curve,
            'cost_curve': cost_curve,
            'fidelities': fidelities,
            'fidelity_trace': [0 if f == 'L' else 1 for f in fidelities],
            'lf_fraction': sum(1 for f in fidelities if f == 'L')
                / max(len(fidelities), 1),
            'initial_hf_values': self.initial_hf_values,
        }


# ════════════════════════════════════
# BASELINE 2: MF-MI-Greedy (Song, Chen & Yue 2019)
# ════════════════════════════════════

class MFMIGreedyOptimizer:
    """
    Additive GP model (different from the KO model used by MF-DRO):
        f_ell(x) = f_H(x) + epsilon_ell(x)
        f_H ~ GP(0, k_H)          -- target function GP (gp_H)
        epsilon_ell ~ GP(0, k_e)  -- error GP for the LF discrepancy from
                                     f_H (gp_error); HF observations have
                                     epsilon_H(x) == 0 identically (an HF
                                     query directly observes f_H, no
                                     discrepancy term), so only one error GP
                                     (for LF) is needed, matching the given
                                     class skeleton's single self.gp_error.

    FITTING PROCEDURE (engineering choice, not read from the source paper --
    see module docstring): gp_H is fit directly on (X_hf, Y_hf). gp_error is
    fit on (X_lf, Y_lf - mu_H(X_lf)), i.e. the LF residual against gp_H's
    OWN current posterior mean at the LF locations -- the same
    alternating-fit pattern already used by src/models/ko_gp.py's
    KennedyOHaganGP (Kennedy-O'Hagan uses this to fit delta against
    rho*mu_L; here it plays the same role with the roles of "primary" and
    "residual" GP reversed, weight fixed at 1 instead of a fitted rho, and
    the residual attached to LF instead of HF).
    """

    def __init__(self, benchmark, n_initial=5,
                 n_initial_hf=None, n_initial_lf=None, seed=0,
                 cost_budget=None, use_sequential_init=False):
        self.use_sequential_init = use_sequential_init
        self.benchmark = benchmark
        self.d = benchmark.dim
        self.bounds = benchmark.bounds
        self.n_initial = n_initial
        self.n_initial_hf = n_initial_hf if n_initial_hf is not None else n_initial
        self.n_initial_lf = n_initial_lf if n_initial_lf is not None else n_initial
        self.seed = seed
        self.cost_budget = cost_budget

        self.gp_H = None
        self.gp_error = None
        self.data_H_x, self.data_H_y = [], []
        self.data_L_x, self.data_L_y = [], []

    def _add_obs(self, x, y, fidelity):
        if fidelity == 'H':
            self.data_H_x.append(x)
            self.data_H_y.append(y)
        else:
            self.data_L_x.append(x)
            self.data_L_y.append(y)

    def _refit_hf_gp(self):
        if self.data_H_x:
            X = torch.stack(self.data_H_x)
            Y = torch.tensor(self.data_H_y, dtype=DEFAULT_DTYPE)
            self.gp_H = _build_gp(X, Y, self.bounds, self.d)

    def _refit_error_gp(self):
        if not self.data_L_x:
            return
        X = torch.stack(self.data_L_x)
        Y = torch.tensor(self.data_L_y, dtype=DEFAULT_DTYPE)
        if self.gp_H is not None:
            with torch.no_grad():
                mu_H_at_L = self.gp_H.posterior(X).mean.reshape(-1)
        else:
            mu_H_at_L = torch.zeros_like(Y)
        residual = Y - mu_H_at_L
        self.gp_error = _build_gp(X, residual, self.bounds, self.d)

    def _information_gain(self, x, ell):
        """
        I(y_{x,ell}; f_H | S) in closed form. Under the additive model,
        y_{x,ell} = f_H(x) + epsilon_ell(x) + noise, with f_H(x) and
        epsilon_ell(x) independent given S. For Y = Z + N (Z, N independent
        Gaussians given S), I(Y;Z) = H(Y) - H(Y|Z) = H(Y) - H(N) =
        0.5*log(Var(Y)/Var(N)) = 0.5*log(1 + Var(Z)/Var(N)). Here Z=f_H(x)
        (Var(Z)=sigma_H^2(x)) and N = epsilon_ell(x) + noise (Var(N) =
        sigma_eps^2(x) + sigma_n^2, with sigma_eps==0 identically for HF).
        """
        mu_H, sigma_H = _gp_posterior_or_prior(self.gp_H, x)
        var_H = (sigma_H ** 2).item()

        if ell == 'H':
            var_eps = 0.0
            noise = self.gp_H.likelihood.noise.item() if self.gp_H is not None else _NOISE_LB
        else:
            _, sigma_eps = _gp_posterior_or_prior(self.gp_error, x)
            var_eps = (sigma_eps ** 2).item()
            noise = self.gp_error.likelihood.noise.item() if self.gp_error is not None else _NOISE_LB

        ig = 0.5 * math.log(1.0 + var_H / (var_eps + noise + 1e-12))
        return max(ig, 0.0)

    def _explore_lf(self, remaining_budget_proxy, S=None):
        """
        Algorithm 2: greedy LF exploration. remaining_budget_proxy: use c_H
        (one HF query's cost) as the phase budget B, per the task's
        instruction (Song 2019 uses a fixed overall budget Lambda; this
        codebase runs per-round instead, so B is re-set to c_H each round
        rather than being a single shrinking global budget).

        Stops when ANY of:
            (a) cumulative LF cost would exceed B
            (b) best LF info-gain/cost < best HF info-gain/cost (LF no
                longer competitive with just querying HF directly)
            (c) cumulative benefit/cost ratio for this phase < beta
                (beta = 1/sqrt(B), O(1/sqrt(B)) per the paper); only checked
                after at least one LF point has been selected, since an
                empty phase has no cumulative ratio to compare.

        De-duplication: selected candidates are removed from the pool after
        being picked (a proxy for "diminishing returns" without implementing
        full fantasy-GP conditioning inside this loop -- not specified by
        the task, and full conditioning would require a third fantasy-model
        machinery beyond what's asked here; removal at least prevents this
        greedy loop from repeatedly re-selecting the exact same point).

        Returns: list of selected LF x locations (tensors, [d] each).
        """
        B = remaining_budget_proxy
        beta = 1.0 / math.sqrt(max(B, 1e-6))
        candidates = list(_sample_candidates(self.bounds, self.d, 200))

        selected = []
        cumulative_cost = 0.0
        cumulative_benefit = 0.0

        best_hf_ratio = max(
            (self._information_gain(x, 'H') / self.benchmark.c_H for x in candidates),
            default=0.0,
        )

        while candidates:
            ratios = [self._information_gain(x, 'L') / self.benchmark.c_L for x in candidates]
            best_idx = max(range(len(candidates)), key=lambda i: ratios[i])
            best_ratio = ratios[best_idx]

            if best_ratio < best_hf_ratio:
                break  # (b)
            if cumulative_cost + self.benchmark.c_L > B:
                break  # (a)
            if selected and (cumulative_benefit / max(cumulative_cost, 1e-12)) < beta:
                break  # (c)

            x = candidates.pop(best_idx)
            selected.append(x)
            cumulative_cost += self.benchmark.c_L
            cumulative_benefit += best_ratio * self.benchmark.c_L

        return selected

    def _select_hf_by_ei(self):
        """
        Phase 2 (SF-GP-OPT): one step of EI on gp_H using all accumulated
        HF observations to select the next HF query location.
        """
        candidates = _sample_candidates(self.bounds, self.d, _CANDIDATE_POOL)
        if self.gp_H is None or not self.data_H_y:
            return candidates[0]
        best_f = max(self.data_H_y)
        acq = LogExpectedImprovement(model=self.gp_H, best_f=best_f, maximize=True)
        X = candidates.unsqueeze(1)  # [N, 1, d] for a q=1 analytic acquisition
        with torch.no_grad():
            values = acq(X)
        best_idx = torch.argmax(values).item()
        return candidates[best_idx]

    def _sample_initial(self):
        X_hf = _lhs_init_points(self.bounds, self.d, self.n_initial_hf,
                                 self.seed, seed_offset=0,
                                 use_sequential_init=self.use_sequential_init)
        for x in X_hf:
            y = self.benchmark.evaluate(x, 'H')
            self._add_obs(x, y, 'H')
        self.initial_hf_values = list(self.data_H_y)
        X_lf = _lhs_init_points(self.bounds, self.d, self.n_initial_lf,
                                 self.seed, seed_offset=1,
                                 use_sequential_init=self.use_sequential_init)
        for x in X_lf:
            y = self.benchmark.evaluate(x, 'L')
            self._add_obs(x, y, 'L')
        self._refit_hf_gp()
        self._refit_error_gp()

    def run(self, bo_iterations):
        """
        Full run. Each round: Explore-LF (Phase 1) until its stopping
        conditions fire, then one HF query via SF-GP-OPT (Phase 2). Returns
        regret and cost curves (POST-INIT -- starts near 0, excludes
        initialization spending). Terminates early once post_init_cost
        reaches self.cost_budget; bo_iterations is a safety cap only.
        Each round costs at most 2*c_H (Explore-LF is itself capped at
        c_H by _explore_lf's own budget-proxy), so checking the budget
        only at the top of each round bounds worst-case overshoot to
        <2*c_H -- small relative to the cost_budget scales used here.
        """
        self._sample_initial()
        cumulative_cost = sum(self.benchmark.c_L for _ in self.data_L_y) \
            + sum(self.benchmark.c_H for _ in self.data_H_y)
        post_init_cost = 0.0
        budget = self.cost_budget if self.cost_budget is not None else float('inf')

        regret_curve, cost_curve = [], []
        # Per-QUERY fidelity trace (not per-round): a round issues a variable
        # number of LF queries followed by exactly one HF query, so counting
        # rounds would misreport lf_fraction.
        fidelity_trace = []
        for _ in range(bo_iterations):
            if post_init_cost >= budget:
                break
            lf_actions = self._explore_lf(self.benchmark.c_H)
            fidelity_trace.extend([0] * len(lf_actions))
            fidelity_trace.append(1)
            for x in lf_actions:
                y = self.benchmark.evaluate(x, 'L')
                self._add_obs(x, y, 'L')
                cumulative_cost += self.benchmark.c_L
                post_init_cost += self.benchmark.c_L
            self._refit_error_gp()

            x_hf = self._select_hf_by_ei()
            y_hf = self.benchmark.evaluate(x_hf, 'H')
            self._add_obs(x_hf, y_hf, 'H')
            cumulative_cost += self.benchmark.c_H
            post_init_cost += self.benchmark.c_H
            self._refit_hf_gp()
            self._refit_error_gp()

            best_hf = max(self.data_H_y)
            # known_optimal_value_hf is on the RAW (pre-negation) scale
            # (benchmarks.py's established convention); best_hf comes from
            # data_H_y, populated by benchmark.evaluate() which calls the
            # NEGATED (maximization-ready) f_hf -- so best_hf must be
            # negated back before comparing, matching dro.py's own
            # maximize-mode regret convention (-best_observed - known_opt).
            regret = -best_hf - self.benchmark.known_optimal_value_hf
            regret_curve.append(regret)
            cost_curve.append(post_init_cost)

        return {
            'regret_curve': regret_curve,
            'cost_curve': cost_curve,
            'fidelity_trace': fidelity_trace,
            'lf_fraction': sum(1 for e in fidelity_trace if e == 0)
                / max(len(fidelity_trace), 1),
            'initial_hf_values': self.initial_hf_values,
        }


# ════════════════════════════════════
# BASELINE 3: Greedy MF-MES (DT-free ablation of MF-DRO's own acquisition)
# ════════════════════════════════════

class GreedyMFMESOptimizer(GreedyMFBase):
    """
    DT-free greedy ablation of MF-DRO's own joint MF-MES acquisition
    (compute_joint_mf_mes in src/policy/mf_dro.py), and the "KO-MES" method
    of the KO-vs-additive experiment. No DT, no rollouts, no training -- at
    each real iteration, refit a single KO GP on all accumulated data and
    greedily pick (x, ell) = argmax InfoGain(x,ell)/c(ell) over candidates
    sampled uniformly across the FULL domain (not ROI-filtered, unlike
    MF-DRO's real inference path -- guarantees full-domain coverage
    regardless of whatever basin contains the optimum, sidestepping the
    ROI-candidate-density question raised by the coverage-failure
    diagnostic).

    Isolates whether MF-DRO's underperformance traces to the DT policy
    learner, or to something upstream shared by both (the KO GP itself, the
    MES formula, or initialization/coverage).

    Reuses _build_hf_proxy_model / _compute_mes_hf_vectorized /
    _compute_mes_lf_vectorized from src.policy.mf_dro directly -- same
    acquisition math MF-DRO's rollout teacher uses, just applied greedily
    with no DT in the loop. The initial design, cost accounting and regret
    convention come from GreedyMFBase, shared with Additive-MES/SF-MES.
    """

    def __init__(self, benchmark, n_initial_hf=5, n_initial_lf=5, seed=0,
                 cost_budget=None, use_sequential_init=False,
                 n_candidates=200, rho_fixed=None, use_dkl=False):
        super().__init__(benchmark, n_initial_hf=n_initial_hf,
                         n_initial_lf=n_initial_lf, seed=seed,
                         cost_budget=cost_budget,
                         use_sequential_init=use_sequential_init,
                         n_candidates=n_candidates)
        # use_dkl=False pushes KennedyOHaganGP's dkl_threshold out of reach so
        # the run stays pure-RBF end to end. This has to be explicit: the KO
        # GP's own default threshold is 30 HF observations, which EVERY
        # benchmark here crosses partway through a run (Currin_2D starts at 6
        # HF and reaches 30; Hartmann_6D starts at 18; Borehole_8D at 24), so
        # leaving the default in place would silently switch the surrogate to
        # a deep kernel mid-run. That would both contaminate the KO-MES
        # reference and destroy the KO-MES vs KO-MES+DKL contrast, since the
        # "no-DKL" arm would already be running DKL for its second half.
        self.use_dkl = use_dkl
        self.ko_ensemble = [KennedyOHaganGP(
            d=self.d, rho_fixed=rho_fixed,
            dkl_threshold=(30 if use_dkl else float('inf')),
        )]

    def _update_model(self):
        X_hf = torch.stack(self.data_hf_x)
        Y_hf = torch.tensor(self.data_hf_y, dtype=DEFAULT_DTYPE)
        if self.data_lf_x:
            X_lf = torch.stack(self.data_lf_x)
            Y_lf = torch.tensor(self.data_lf_y, dtype=DEFAULT_DTYPE)
        else:
            X_lf = X_hf[:0]
            Y_lf = Y_hf[:0]
        self.ko_ensemble[0].fit(X_lf, Y_lf, X_hf, Y_hf, self.bounds)

    def _propose_greedy(self, X_cand):
        ko = self.ko_ensemble[0]
        hf_proxy = _build_hf_proxy_model(ko)
        y_star = thompson_sample_y_star(hf_proxy, X_cand, K=10)
        mes_hf = _compute_mes_hf_vectorized(X_cand, hf_proxy, y_star)
        mes_lf = _compute_mes_lf_vectorized(X_cand, ko, y_star, n_quad=32)
        return cost_normalized_argmax(mes_lf, mes_hf, self.c_L, self.c_H, X_cand)

    def final_rho(self):
        return float(self.ko_ensemble[0].rho.item())
