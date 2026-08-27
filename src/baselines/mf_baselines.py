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


from src.baselines.additive_mfgp import AdditiveMFGP


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

class MFGPUCBCostNormalized:
    """
    Independent GPs per fidelity -- NOT the KO model. gp_L is fitted only on
    LF observations, gp_H only on HF observations; no information sharing
    between fidelities through the GP itself, only through the
    cost-normalized UCB acquisition (a cheap, uncertain LF query can still
    win the argmax over an expensive, more-certain HF query).

    SUPERSEDED 2026-08-26 -- THIS IS NOT MF-GP-UCB.

    Kandasamy et al. (2016) Algorithm 1 never divides a UCB by cost. It picks
    x_t = argmax_x min_m [mu^(m) + beta^0.5 sigma^(m) + zeta^(m)] and then picks
    the fidelity by an UNCERTAINTY THRESHOLD, m_t = min{m : beta^0.5 sigma^(m)(x_t)
    >= gamma^(m)}, with a gamma-doubling rule that forces escalation when the
    algorithm stalls at a low fidelity. Cost enters only through that schedule.

    This class implements score(x,l) = UCB_l(x)/c_l instead. Dividing by cost is
    why it degenerates to all-LF whenever c_H exceeds the GPs' variance ratio --
    an artifact of the wrong rule, not a property of MF-GP-UCB. The earlier
    docstring attributed that degeneracy to Kandasamy et al.; that attribution
    was wrong.

    Retained ONLY so h57/h72's recorded MF-GP-UCB numbers remain explainable.
    Do not use it as a baseline. See MFGPUCBOptimizer below.
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

class MFGPUCBOptimizer:
    """
    MF-GP-UCB, Kandasamy et al. (2016) -- arXiv:1603.06288 Algorithm 1.

    Faithful re-implementation (2026-08-26). The previous class under this name
    scored candidates by UCB_l(x)/c_l, which is NOT in the paper and which made
    the method degenerate to all-LF whenever c_H exceeded the GPs' variance
    ratio. That class is retained above as MFGPUCBCostNormalized purely so
    h57/h72's recorded numbers stay explainable.

    ALGORITHM 1, for M=2 (m=1 low fidelity, m=2=M high fidelity):

      per-fidelity GPs, each conditioned ONLY on its own fidelity's data:
          mu^(m), sigma^(m)

      1. phi^(m)(x) = mu^(m)(x) + beta_t^0.5 sigma^(m)(x) + zeta^(m)
         phi(x)     = min_m phi^(m)(x)                       <- MIN over fidelities
         x_t        = argmax_x phi(x)
      2. m_t = min{ m : beta_t^0.5 sigma^(m)(x_t) >= gamma^(m) }, else M
      3. query f^(m_t) at x_t
      4. update D^(m_t)

    zeta^(M) = 0; for M=2 there is a single zeta = zeta^(1). Cost NEVER enters
    the selection rule -- it appears only in the gamma schedule below.

    THE TWO ADAPTIVE HEURISTICS (paper Section 5), both implemented:

      gamma doubling -- "If the algorithm does not query above the m-th fidelity
        for more than lambda^(m+1)/lambda^(m) iterations, we double gamma^(m)."
        Raising gamma^(1) makes the LF condition harder to satisfy and forces
        escalation to HF. This is what prevents the stall the old class could
        not escape.

      zeta adaptation -- "Whenever we query at any fidelity m > 1 we also check
        the posterior mean of the (m-1)-th fidelity. If |f^(m)(x_t) -
        mu^(m-1)(x_t)| > zeta, we query again at x_t at the (m-1)-th fidelity.
        If |f^(m)(x_t) - f^(m-1)(x_t)| > zeta, we update zeta to twice the
        violation."

    INITIALISATION (paper: "start with small values", no formula given). Both are
    auto-calibrated per benchmark from the initial design so no hand-picked
    constant has to be scale-matched to Borehole's hundreds vs Hartmann's units:
      zeta_0  = 10th percentile of |y_H - mu_L(x_H)| over the HF initial design
      gamma_0 = 0.1 * mean_x beta^0.5 sigma^(1)(x) over a Sobol pool
    Both then evolve by the paper's own update rules.
    """

    def __init__(self, benchmark, n_initial=5, delta=0.1,
                 n_initial_hf=None, n_initial_lf=None, seed=0,
                 cost_budget=None, use_sequential_init=False):
        self.benchmark = benchmark
        self.d = benchmark.dim
        self.bounds = benchmark.bounds
        self.n_initial = n_initial
        self.n_initial_hf = n_initial_hf if n_initial_hf is not None else n_initial
        self.n_initial_lf = n_initial_lf if n_initial_lf is not None else n_initial
        self.seed = seed
        self.cost_budget = cost_budget
        self.delta = delta
        self.use_sequential_init = use_sequential_init
        self.refit_every = 25
        self.gp_L = None
        self.gp_H = None
        self.data_L_x, self.data_L_y = [], []
        self.data_H_x, self.data_H_y = [], []
        self._iters_since_refit = {'L': 0, 'H': 0}
        # adaptive parameters (calibrated in _sample_initial)
        self.zeta = None
        self.gamma = None
        self._iters_since_hf = 0
        # paper: lambda^(m+1)/lambda^(m) iterations before doubling gamma^(m)
        self._gamma_patience = max(1, int(round(benchmark.c_H / benchmark.c_L)))
        self.n_gamma_doublings = 0
        self.n_zeta_updates = 0
        self.n_zeta_triggered_lf = 0

    def _compute_beta_t(self, t, cardinality=_CANDIDATE_POOL):
        return 2.0 * math.log(
            cardinality * (t ** 2) * (math.pi ** 2) / (6.0 * self.delta))

    def _post(self, gp, X):
        """Batched (mu, sigma) for a [N,d] tensor; prior fallback when gp is None."""
        if gp is None:
            n = X.shape[0]
            return (torch.zeros(n, dtype=DEFAULT_DTYPE),
                    torch.ones(n, dtype=DEFAULT_DTYPE))
        with torch.no_grad():
            p = gp.posterior(X)
            return (p.mean.reshape(-1),
                    p.variance.clamp_min(1e-12).reshape(-1).sqrt())

    def _add_obs(self, x, y, fidelity):
        if fidelity == 'L':
            self.data_L_x.append(x); self.data_L_y.append(y)
        else:
            self.data_H_x.append(x); self.data_H_y.append(y)

    def _maybe_refit(self, fidelity):
        xs, ys = ((self.data_L_x, self.data_L_y) if fidelity == 'L'
                  else (self.data_H_x, self.data_H_y))
        if not xs:
            return
        X = torch.stack(xs); Y = torch.tensor(ys, dtype=DEFAULT_DTYPE)
        current = self.gp_L if fidelity == 'L' else self.gp_H
        self._iters_since_refit[fidelity] += 1
        full = current is None or self._iters_since_refit[fidelity] >= self.refit_every
        if full:
            new_gp = _build_gp(X, Y, self.bounds, self.d)
            self._iters_since_refit[fidelity] = 0
        else:
            new_gp = _rebuild_frozen(current, X, Y, self.bounds, self.d)
        if fidelity == 'L':
            self.gp_L = new_gp
        else:
            self.gp_H = new_gp

    def select_next(self, t):
        """Algorithm 1 steps 1-2. Returns (x_t, fidelity)."""
        beta_t = self._compute_beta_t(t)
        rb = beta_t ** 0.5
        cand = _sample_candidates(self.bounds, self.d, _CANDIDATE_POOL)
        muL, sdL = self._post(self.gp_L, cand)
        muH, sdH = self._post(self.gp_H, cand)
        phi_L = muL + rb * sdL + self.zeta      # zeta^(1) = zeta
        phi_H = muH + rb * sdH                  # zeta^(M) = 0
        phi = torch.minimum(phi_L, phi_H)       # step 1: MIN over fidelities
        i = int(torch.argmax(phi).item())
        x_t = cand[i]
        # step 2: lowest fidelity whose confidence band is still wide at x_t
        fid = 'L' if (rb * sdL[i]).item() >= self.gamma else 'H'
        return x_t, fid

    def _post_query_updates(self, x_t, y_t, fid):
        """Paper Section 5: zeta adaptation and gamma doubling.
        Returns extra cost incurred by a zeta-triggered LF re-query (0 or c_L)."""
        extra = 0.0
        if fid == 'H':
            self._iters_since_hf = 0
            muL_at, _ = self._post(self.gp_L, x_t.reshape(1, -1))
            viol = abs(float(y_t) - float(muL_at.item()))
            if viol > self.zeta:   # ZETA_INC_COEFF = 2 below, per mfBO.m
                # "we query again at x_t, but at the (m-1)-th fidelity"
                y_lf = self.benchmark.evaluate(x_t, 'L')
                self._add_obs(x_t, y_lf, 'L')
                self._maybe_refit('L')
                extra = self.benchmark.c_L
                self.n_zeta_triggered_lf += 1
                true_viol = abs(float(y_t) - float(y_lf))
                if true_viol > self.zeta:
                    self.zeta = 2.0 * true_viol      # "twice the violation"
                    self.n_zeta_updates += 1
        else:
            self._iters_since_hf += 1
            # Reference mfBO.m: counter fires at costs(m+1)/costs(m) and the
            # increase coefficient is GAMMA_INC_COEFF = 5, not a doubling.
            if self._iters_since_hf >= self._gamma_patience:
                self.gamma *= 5.0
                self._iters_since_hf = 0
                self.n_gamma_doublings += 1
        return extra

    def _sample_initial(self):
        X_lf = _lhs_init_points(self.bounds, self.d, self.n_initial_lf,
                                self.seed, seed_offset=1,
                                use_sequential_init=self.use_sequential_init)
        for x in X_lf:
            self._add_obs(x, self.benchmark.evaluate(x, 'L'), 'L')
        X_hf = _lhs_init_points(self.bounds, self.d, self.n_initial_hf,
                                self.seed, seed_offset=0,
                                use_sequential_init=self.use_sequential_init)
        for x in X_hf:
            self._add_obs(x, self.benchmark.evaluate(x, 'H'), 'H')
        self.initial_hf_values = list(self.data_H_y)
        self._maybe_refit('L'); self._maybe_refit('H')
        # --- zeta / gamma initialisation, taken from the AUTHORS' MATLAB
        # reference (mfBO/mfboPreProcessParams.m), not invented here:
        #     params.zeta  = ceil(1e4 * 2*maxF1F2Diff)/1e4
        #     params.gamma = ceil(1e4 * 0.01 * rangeY)/1e4
        # where maxF1F2Diff is max(0, max(f_H - f_L)) over PAIRED initial points
        # and rangeY is the observed range of all initial values. An earlier
        # version of this method used a 10th-percentile zeta and a
        # 0.1*mean(beta^0.5 sigma) gamma -- both invented, neither anchored to
        # the value scale. ---
        allY = torch.tensor(list(self.data_H_y) + list(self.data_L_y),
                            dtype=DEFAULT_DTYPE)
        rangeY = float((allY.max() - allY.min()).item())
        Xh = torch.stack(self.data_H_x)
        muL_at, _ = self._post(self.gp_L, Xh)
        # no paired HF/LF design here, so mu_L at the HF points stands in for f_L
        diffs = torch.tensor(self.data_H_y, dtype=DEFAULT_DTYPE) - muL_at
        maxDiff = float(torch.clamp(diffs.max(), min=0.0).item())
        self.zeta = math.ceil(1e4 * 2.0 * maxDiff) / 1e4
        self.gamma = math.ceil(1e4 * 0.01 * rangeY) / 1e4 or 1e-6

    def run(self, bo_iterations):
        self._sample_initial()
        regret_curve, cost_curve, fidelities = [], [], []
        post_init_cost = 0.0
        for t in range(1, bo_iterations + 1):
            x_t, fid = self.select_next(t)
            y_t = self.benchmark.evaluate(x_t, fid)
            self._add_obs(x_t, y_t, fid)
            self._maybe_refit(fid)
            post_init_cost += (self.benchmark.c_L if fid == 'L'
                               else self.benchmark.c_H)
            post_init_cost += self._post_query_updates(x_t, y_t, fid)
            best_hf = max(self.data_H_y) if self.data_H_y else float('-inf')
            regret_curve.append(-best_hf - self.benchmark.known_optimal_value_hf)
            cost_curve.append(post_init_cost)
            fidelities.append(fid)
            if self.cost_budget is not None and post_init_cost >= self.cost_budget:
                break
        return {
            'regret_curve': regret_curve,
            'cost_curve': cost_curve,
            'fidelities': fidelities,
            'fidelity_trace': [0 if f == 'L' else 1 for f in fidelities],
            'lf_fraction': sum(1 for f in fidelities if f == 'L') / max(len(fidelities), 1),
            'initial_hf_values': self.initial_hf_values,
            'final_zeta': self.zeta, 'final_gamma': self.gamma,
            'n_gamma_doublings': self.n_gamma_doublings,
            'n_zeta_updates': self.n_zeta_updates,
            'n_zeta_triggered_lf': self.n_zeta_triggered_lf,
        }


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
        # reference episode state (mfBO.m lines 101-110)
        self._lambda = 1.0            # params.lambda = 1
        self._mean_acq = 0.0
        self._cost_low = 0.0
        self._num_low_fidel = 0
        self._episode_best_bcr = -1.0
        self._budget_total = float(cost_budget) if cost_budget else 1.0
        self._spent = 0.0
        self._amfgp = AdditiveMFGP(self.d, 2)
        self._lf_noise_var = 1.0
        self._target_noise_var = 1e-4
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

    def _acq_mi(self, x, ell):
        """Reference acquisition, from the authors' acqMFMIGreedy.m:

            [~,~,nextStd]   = funcs{i}(x);          % fidelity i's OWN posterior std
            [nextNoiseStd]  = noiseFuncHs{i}(x,x);  % fidelity i's OWN noise
            acq = 0.5*log(1 + nextStd^2/nextNoiseStd^2) / costs(i);

        This replaces 0.5*log(1 + var_H/(var_eps+noise)), which used the TARGET
        fidelity's variance for both arms. That form appears in acqMFMIGreedy.m
        as commented-out code -- the author tried it and did not ship it.

        DELIBERATE DEVIATION, one squaring. `funcs{i}` is AdditiveGPRegression's
        handle and GPComputeOutputs sets `yStd = sqrt(real(diag(yK)))`, so
        `nextStd^2` is a genuine variance. But `noiseFuncHs{i}(x,x)` is a kernel
        DIAGONAL -- sqExpKernel is `scale*exp(-D/2bw^2)`, hence k(x,x) = scale --
        which is already a variance, and the reference squares it again. Taken
        literally the acquisition is variance / variance^2, and at the target
        fidelity the denominator becomes (1e-4*stdY^2)^2 = 1e-8*stdY^4, driving
        the target-fidelity ratio so high that the episode terminates on its
        first step every time. That is precisely the 100%-target-fidelity
        collapse we measured. We therefore use the paper's Gaussian mutual
        information, 0.5*log(1 + sigma^2/sigma_noise^2), treating
        noiseFuncs{i}(x,x) as the variance it is. Every other element of the
        acquisition -- per-fidelity posterior, per-fidelity noise, cost
        division -- follows the reference exactly.
        """
        fi = 0 if ell == 'L' else 1
        _, var = self._amfgp.posterior(x.reshape(1, -1).cpu().numpy(), fi)
        noise_var = self._amfgp.noise_var[fi]
        cost = self.benchmark.c_H if ell == 'H' else self.benchmark.c_L
        return 0.5 * math.log(1.0 + float(var[0]) / max(noise_var, 1e-12)) / cost

    def _acq_mi_batch(self, X, ell):
        fi = 0 if ell == 'L' else 1
        _, var = self._amfgp.posterior(X.cpu().numpy(), fi)
        noise_var = self._amfgp.noise_var[fi]
        cost = self.benchmark.c_H if ell == 'H' else self.benchmark.c_L
        return 0.5 * np.log(1.0 + var / max(noise_var, 1e-12)) / cost

    def _select_mi_greedy(self, t):
        """One query, per the authors' strategyMFMIGreedy.m.

        The reference does NOT batch an exploration phase: each call scores every
        fidelity's own argmax and picks the best benefit-cost ratio, then applies
        the episode-termination test.

        TWO STRUCTURES THE PAPER'S PSEUDOCODE DOES NOT CONTAIN, both load-bearing:

        isFirstEpisode. While the first episode is live, a target-fidelity
        argmax win is OVERRIDDEN to fidelity 1 (`nextFidel = 1`), so the run
        opens with a forced low-fidelity exploration phase. The flag clears at
        the first target-fidelity query and never returns. Without it every
        episode terminates on its first step and the method is single-fidelity.

        remainBudget is FROZEN within an episode -- mfBO.m updates it only in
        the `nextFidel == numFidels` branch. The threshold's
        sqrt(totalBudget/remainBudget) factor is therefore constant across an
        episode rather than growing every query.

        The threshold is knife-edge by construction: on an episode's first step
        meanAcq and costLowFidel are 0, so the left side equals episodeBestBCR
        exactly and the test reduces to 1 < sqrt(totalBudget/remainBudget). Any
        budget already spent makes that true, which is why lambda is a
        per-problem constant in the reference (mfBO.m carries 0.2, 1, 45 and 150
        for different applications).
        """
        cands = _sample_candidates(self.bounds, self.d, _CANDIDATE_POOL)
        best = None
        for ell in ('L', 'H'):
            accs = self._acq_mi_batch(cands, ell)
            j = int(np.argmax(accs))
            if best is None or accs[j] > best[2]:
                best = (cands[j], ell, float(accs[j]))
        x_t, fid, acq = best
        episode_best = acq if self._episode_best_bcr < 0 else self._episode_best_bcr
        self._episode_best_bcr = episode_best

        remain = max(self._remain_budget, 1e-9)
        thresh = (self._lambda * episode_best
                  * math.sqrt(self._budget_total / remain))

        def weighted_with(c):
            return ((self._mean_acq * self._cost_low + acq * c)
                    / max(self._cost_low + c, 1e-12))

        if self._is_first_episode:
            if fid == 'H':                      # forced down to fidelity 1
                fid = 'L'
            terminate = weighted_with(self.benchmark.c_L) < thresh
        else:
            terminate = (fid == 'H'
                         or weighted_with(self.benchmark.c_H if fid == 'H'
                                          else self.benchmark.c_L) < thresh
                         or self._num_low_fidel > 20)

        if terminate:
            fid = 'H'
            # target-fidelity point via GP-UCB (acqMFGPUCB with the single
            # highest-fidelity model), NOT expected improvement. The paper says
            # SF-GP-OPT may be any off-the-shelf BO routine; the reference
            # instantiates it as GP-UCB.
            beta_t = self._compute_beta_t_mi(t)
            # strategyMFMIGreedy.m passes the SAME funcs to the target-point
            # routine, so the surrogate is the joint additive GP, not gp_H.
            mu, var = self._amfgp.posterior(cands.cpu().numpy(), 1)
            x_t = cands[int(np.argmax(mu + (beta_t ** 0.5) * np.sqrt(var)))]
            self._episode_best_bcr = -1.0
            acq = 0.0
        return x_t, fid, acq

    def _compute_beta_t_mi(self, t):
        return 2.0 * math.log(_CANDIDATE_POOL * (t ** 2) * (math.pi ** 2)
                              / (6.0 * 0.1))

    def _batch_post(self, gp, X):
        if gp is None:
            n = X.shape[0]
            return (torch.zeros(n, dtype=DEFAULT_DTYPE),
                    torch.ones(n, dtype=DEFAULT_DTYPE))
        with torch.no_grad():
            p = gp.posterior(X)
            return (p.mean.reshape(-1),
                    p.variance.clamp_min(1e-12).reshape(-1).sqrt())

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
        self._refit_mi_extras()

    def _refit_mi_extras(self):
        """Fit the authors' JOINT additive multi-fidelity GP over both fidelities.

        AdditiveGPRegression.m models f_i(x) = f_M(x) + eps_i(x) with a SHARED
        target kernel over the pooled data plus a per-fidelity discrepancy
        kernel, inverted with a single Cholesky, so low-fidelity observations
        inform the target posterior directly. `noiseFuncs{i}(x,x)` is then that
        discrepancy kernel's diagonal, i.e. the fitted scale_i.

        This replaces two independently fitted GPs (gp_H on target data,
        gp_error on low-fidelity residuals) whose outputscale merely stood in for
        the noise GP. Under that approximation low-fidelity data could not inform
        the target posterior at all, and the noise magnitude -- which sets the
        entire low-vs-target balance in _acq_mi -- was never fit to anything.
        """
        X_list, Y_list = [], []
        for xs, ys in ((self.data_L_x, self.data_L_y), (self.data_H_x, self.data_H_y)):
            if xs:
                X_list.append(torch.stack(list(xs)).cpu().numpy())
                Y_list.append(np.asarray(list(ys), dtype=float))
            else:
                X_list.append(np.zeros((0, self.d)))
                Y_list.append(np.zeros(0))
        lo = self.bounds[0].cpu().numpy() if torch.is_tensor(self.bounds[0]) else np.asarray(self.bounds[0])
        hi = self.bounds[1].cpu().numpy() if torch.is_tensor(self.bounds[1]) else np.asarray(self.bounds[1])
        self._amfgp.fit(X_list, Y_list, (lo, hi))
        self._lf_noise_var = self._amfgp.noise_var[0]
        self._target_noise_var = self._amfgp.noise_var[1]

    def run(self, bo_iterations):
        """One query per iteration, per the authors' mfBO.m loop.

        The reference does not batch an exploration phase: getNextQuery returns a
        single (point, fidelity) each step and the episode state (meanAcq,
        numLowFidel, costLowFidel) is carried in params and reset whenever a
        target-fidelity query is issued. Returns POST-INIT regret and cost curves.
        """
        self._sample_initial()
        budget = self.cost_budget if self.cost_budget is not None else float('inf')
        self._budget_total = budget if budget != float('inf') else 1.0
        self._remain_budget = self._budget_total
        self._is_first_episode = True
        post_init_cost = 0.0
        regret_curve, cost_curve, fidelity_trace = [], [], []

        for t in range(1, bo_iterations + 1):
            if post_init_cost >= budget:
                break
            self._spent = post_init_cost
            x_t, fid, acq = self._select_mi_greedy(t)
            y_t = self.benchmark.evaluate(x_t, fid)
            self._add_obs(x_t, y_t, fid)
            c_sel = self.benchmark.c_H if fid == 'H' else self.benchmark.c_L
            post_init_cost += c_sel

            # Episode state, mfBO.m lines 229-246. meanAcq is a COST-weighted
            # running mean, not a plain one -- lower fidelities can differ in
            # cost, which the commented-out unweighted version did not handle.
            if fid == 'H':
                self._mean_acq = 0.0
                self._num_low_fidel = 0
                self._cost_low = 0.0
                # mfBO.m:232 -- remainBudget is refreshed here and ONLY here,
                # so it stays frozen for the whole of the next episode.
                self._remain_budget = max(budget - post_init_cost, 1e-9)
                self._is_first_episode = False
            else:
                self._mean_acq = ((self._mean_acq * self._cost_low + acq * c_sel)
                                  / max(self._cost_low + c_sel, 1e-12))
                self._num_low_fidel += 1
                self._cost_low += c_sel

            if fid == 'H':
                self._refit_hf_gp()
            self._refit_error_gp()
            self._refit_mi_extras()

            best_hf = max(self.data_H_y) if self.data_H_y else float('-inf')
            regret_curve.append(-best_hf - self.benchmark.known_optimal_value_hf)
            cost_curve.append(post_init_cost)
            fidelity_trace.append(1 if fid == 'H' else 0)

        return {
            'regret_curve': regret_curve,
            'cost_curve': cost_curve,
            'fidelity_trace': fidelity_trace,
            'fidelities': ['H' if f else 'L' for f in fidelity_trace],
            'lf_fraction': 1.0 - (sum(fidelity_trace) / max(len(fidelity_trace), 1)),
            'initial_hf_values': self.initial_hf_values,
        }

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
