"""
NaiveBO: a standard sequential BoTorch qEI Bayesian optimization loop with no
Decision Transformer / rollout-simulation component at all. Used as the
"NaiveBO" variant in Experiment 1 (MES reward ablation) to give a
non-DRO baseline on the same benchmarks.
"""
import time
import warnings

import numpy as np
import torch
from botorch.acquisition import qExpectedImprovement, qMaxValueEntropy
from botorch.exceptions import InputDataWarning
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.transforms import standardize
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

DEFAULT_DTYPE = torch.float64

# Placeholder keys carried in NaiveBO's result dict purely so downstream
# checkpoint/analysis code (which iterates a common schema across all variants)
# doesn't need special-casing -- NaiveBO has no rollouts, so these are always NaN.
_ROLLOUT_ONLY_KEYS = ("mean_reward", "zero_frac", "rtg_target", "batch_max_rtg", "running_max_rtg")


def _sample_initial_points_lhs(n_initial, dim, domain_min, domain_max, device, dtype):
    """Latin Hypercube Sampling, matching DirectRegretOptimization's own approach."""
    points = torch.zeros((n_initial, dim), device=device, dtype=dtype)
    for i in range(dim):
        grid_coords = torch.linspace(0.0, 1.0, n_initial + 1)[:-1] + 0.5 / n_initial
        perm = torch.randperm(n_initial, device=device)
        points[:, i] = grid_coords[perm]
    return domain_min + (domain_max - domain_min) * points


def run_naive_bo(objective_function, domain_min, domain_max, dim, seed,
                  max_iterations=50, initial_points=5, known_optimal_value=0.0,
                  iter_callback=None, naivebo_acq_function="ei", mes_num_candidates=500,
                  mes_num_mv_samples=10):
    """
    Runs a standard BO loop (ARD-RBF SingleTaskGP, fit via Adam, q=1
    sequential proposals via optimize_acqf) for max_iterations real evaluations
    after an LHS initial design.

    objective_function: BoTorch SyntheticTestFunction with negate=True (i.e.
    higher is better, optimum at known_optimal_value negated).
    iter_callback(t, regret, best_observed, iter_time): optional, called once
    per real iteration (for progress logging).
    naivebo_acq_function: "ei" (default, unchanged behavior) or "mes". When
    "mes", uses qMaxValueEntropy with a fresh domain-wide uniform candidate_set
    each iteration (the same non-local-ROI candidate source as DRO's
    gumbel_candidates -- NaiveBO has no local-ROI concept at all, so this is
    just "sample uniformly over the full domain", matching the same principle).

    Returns a result dict with the same schema as DirectRegretOptimization's
    run_single_seed results, so checkpoint/analysis code can treat all variants
    uniformly.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    dtype = DEFAULT_DTYPE

    bounds = torch.tensor([domain_min, domain_max], device=device, dtype=dtype)
    domain_min_t, domain_max_t = bounds[0], bounds[1]

    train_x = _sample_initial_points_lhs(initial_points, dim, domain_min_t, domain_max_t, device, dtype)
    train_y = objective_function(train_x).reshape(-1, 1).to(device=device, dtype=dtype)

    result = {
        "regret_curve": [], "best_observed": [], "iter_times": [],
        **{k: [] for k in _ROLLOUT_ONLY_KEYS},
    }
    best_so_far = train_y.max().item()

    for t in range(1, max_iterations + 1):
        iter_start = time.perf_counter()

        y_standardized = standardize(train_y.squeeze(-1)).unsqueeze(-1)
        likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-4))
        covar_module = ScaleKernel(RBFKernel(ard_num_dims=dim)) # ARD-RBF, per experiment hyperparameters
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=InputDataWarning)
            model = SingleTaskGP(
                train_X=train_x, train_Y=y_standardized,
                likelihood=likelihood, covar_module=covar_module,
            ).to(device=device, dtype=dtype)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        model.train()
        model.likelihood.train()
        gp_optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        for _ in range(50):
            gp_optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, y_standardized.squeeze(-1))
            loss.backward()
            gp_optimizer.step()
        model.eval()
        model.likelihood.eval()

        if naivebo_acq_function == "mes":
            gumbel_candidates = domain_min_t + (domain_max_t - domain_min_t) * torch.rand(
                mes_num_candidates, dim, device=device, dtype=dtype
            )
            acq_func = qMaxValueEntropy(
                model=model, candidate_set=gumbel_candidates,
                num_mv_samples=mes_num_mv_samples, maximize=True,
            )
        else:
            acq_func = qExpectedImprovement(model=model, best_f=y_standardized.max())
        candidate, _ = optimize_acqf(
            acq_function=acq_func, bounds=bounds, q=1,
            num_restarts=10, raw_samples=512,
        )

        y_new = objective_function(candidate).reshape(-1, 1).to(device=device, dtype=dtype)
        train_x = torch.cat([train_x, candidate], dim=0)
        train_y = torch.cat([train_y, y_new], dim=0)
        best_so_far = max(best_so_far, y_new.item())

        regret = -best_so_far - known_optimal_value
        iter_time = time.perf_counter() - iter_start

        result["regret_curve"].append(regret)
        result["best_observed"].append(best_so_far)
        result["iter_times"].append(iter_time)
        for k in _ROLLOUT_ONLY_KEYS:
            result[k].append(float('nan'))

        if iter_callback is not None:
            iter_callback(t, regret, best_so_far, iter_time)

    return result
