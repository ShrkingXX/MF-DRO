"""
Max-value Entropy Search (MES) reward for DRO rollouts.

Dense alternative to the sparse improvement reward used in
src/policy/dro.py's `_simulate_trajectory`. Reward at a query point x_tau is
the mutual information between the (unknown) global maximum y* and the
noisy observation at x_tau, following Wang & Jegelka (2017).
"""
import math

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.stats import norm as scipy_norm

DEFAULT_DTYPE = torch.float64

_LOG_SQRT_2PI_E = 0.5 * math.log(2.0 * math.pi * math.e)
_EPS_STD = 1e-9
_EPS_PHI = 1e-12


def _cdf_of_max(z, mu_np, sigma_np):
    """F(z) = prod_i Phi((z - mu_i) / sigma_i), evaluated in log-space for stability."""
    log_terms = scipy_norm.logcdf((z - mu_np) / sigma_np)
    return float(np.exp(np.sum(log_terms)))


def _bracket_for_quantiles(mu_np, sigma_np, max_expand=200):
    """Find [lo, hi] such that _cdf_of_max(lo) ~ 0 and _cdf_of_max(hi) ~ 1."""
    scale = float(sigma_np.max()) + 1e-6
    lo = float(mu_np.max())
    hi = float(mu_np.max()) + scale

    step = scale
    n = 0
    while _cdf_of_max(hi, mu_np, sigma_np) < 1.0 - 1e-6 and n < max_expand:
        hi += step
        step *= 1.5
        n += 1

    step = scale
    n = 0
    while _cdf_of_max(lo, mu_np, sigma_np) > 1e-6 and n < max_expand:
        lo -= step
        step *= 1.5
        n += 1

    return lo, hi


def gumbel_sample_y_star(mu_candidates: torch.Tensor, sigma_candidates: torch.Tensor, K: int = 10) -> torch.Tensor:
    """
    Sample K draws of the global maximum y* via the Gumbel approximation
    (Wang & Jegelka 2017, Section 3.1).

    mu_candidates:    [N] GP posterior means at ROI candidates
    sigma_candidates: [N] GP posterior std devs at ROI candidates
    Returns: y_star_samples [K], all > max(mu_candidates)
    """
    device, dtype = mu_candidates.device, mu_candidates.dtype

    mu_np = mu_candidates.detach().cpu().double().numpy()
    sigma_np = np.clip(sigma_candidates.detach().cpu().double().numpy(), _EPS_STD, None)

    lo, hi = _bracket_for_quantiles(mu_np, sigma_np)

    z_25 = brentq(lambda z: _cdf_of_max(z, mu_np, sigma_np) - 0.25, lo, hi)
    z_75 = brentq(lambda z: _cdf_of_max(z, mu_np, sigma_np) - 0.75, lo, hi)

    log_term_25 = math.log(-math.log(0.25))
    log_term_75 = math.log(-math.log(0.75))

    b = (z_75 - z_25) / (log_term_25 - log_term_75)
    a = z_25 + b * log_term_25

    u = torch.rand(K, dtype=torch.float64).clamp(1e-6, 1.0 - 1e-6)
    y_star_samples = a - b * torch.log(-torch.log(u))

    return y_star_samples.to(device=device, dtype=dtype)


_DEFAULT_ROI_SUBSAMPLE = 150


def compute_mes_reward(x_tau: torch.Tensor, gp_model, roi_candidates: torch.Tensor, K: int = 10,
                        roi_subsample: int = _DEFAULT_ROI_SUBSAMPLE) -> torch.Tensor:
    """
    x_tau:          [d] query location
    gp_model:       current (fantasy) GP at this rollout step (must expose .posterior(X))
    roi_candidates: [N_roi, d] candidate points in ROI
    roi_subsample:  cap on how many roi_candidates are used for the joint
                    rsample() below (see rationale in the y* sampling
                    paragraph). None disables subsampling.
    Returns: scalar >= 0, in nats

    y* sampling: Thompson sampling (exact joint rsample() from the GP
    posterior at roi_candidates, then max) instead of gumbel_sample_y_star's
    product-CDF approximation. validation/phase1_gumbel_quality.py found the
    product-CDF method (independence across roi_candidates) is a poor fit
    under RBF-kernel posterior correlation (KS 0.16-0.86 across 15
    benchmark/stage combinations); validation/phase1_thompson_gumbel.py
    confirmed Thompson sampling removes that assumption entirely (KS
    0.03-0.10).

    COST CAVEAT (found empirically, see mes_cost_benchmark.py): the
    product-CDF approximation only ever touches marginal means/variances --
    O(N_roi) per CDF evaluation, no covariance matrix -- which is exactly
    the independence assumption that makes it inaccurate, but also what
    makes it cheap. rsample() needs the FULL joint covariance over
    roi_candidates (a Cholesky factorization that scales worse than
    linearly), so Thompson sampling is only cheaper than the product-CDF
    approach up to roughly N_roi~500-700; past that it becomes slower
    (measured: 1.36x slower at N_roi=1000, vs. 0.52x-0.82x faster at
    N_roi=50-500). Since roi_candidates commonly reaches ~1000 in this
    codebase, roi_subsample keeps the rsample() call in the cheaper regime
    by capping how many candidates it's computed over -- a random subsample
    of a domain-wide candidate pool is still representative of that pool for
    estimating the max-value distribution, just with a coarser resolution.

    Also batches what were two separate .posterior() calls (roi_candidates,
    then x_tau) into a single call over their concatenation, since a GP's
    marginal mean/variance at any point is unaffected by what other points
    are queried alongside it -- this is a pure efficiency change, not an
    approximation.
    """
    device, dtype = roi_candidates.device, roi_candidates.dtype
    x_tau_batched = x_tau if x_tau.ndim > 1 else x_tau.unsqueeze(0)

    if roi_subsample is not None and roi_candidates.shape[0] > roi_subsample:
        idx = torch.randperm(roi_candidates.shape[0], device=roi_candidates.device)[:roi_subsample]
        roi_for_sampling = roi_candidates[idx]
    else:
        roi_for_sampling = roi_candidates
    n_roi = roi_for_sampling.shape[0]

    with torch.no_grad():
        combined_posterior = gp_model.posterior(torch.cat([roi_for_sampling, x_tau_batched], dim=0))

        combined_samples = combined_posterior.rsample(torch.Size([K])).reshape(K, -1) # [K, N_roi + 1]
        y_star_samples = combined_samples[:, :n_roi].max(dim=-1).values # Thompson samples of y*, [K]

        combined_mean = combined_posterior.mean.reshape(-1)
        combined_var = combined_posterior.variance.clamp_min(_EPS_STD ** 2).reshape(-1)
        mu_x = combined_mean[n_roi]
        sigma_x = combined_var[n_roi].sqrt()

    log_sqrt_2pi_e = torch.tensor(_LOG_SQRT_2PI_E, device=device, dtype=dtype)
    term1 = torch.log(sigma_x) + log_sqrt_2pi_e

    gamma = (y_star_samples - mu_x) / sigma_x
    standard_normal = torch.distributions.Normal(
        torch.zeros((), device=device, dtype=dtype),
        torch.ones((), device=device, dtype=dtype),
    )
    Phi = standard_normal.cdf(gamma).clamp_min(_EPS_PHI)
    phi = standard_normal.log_prob(gamma).exp()

    H = torch.log(sigma_x) + log_sqrt_2pi_e + torch.log(Phi) - (gamma * phi) / (2 * Phi)
    term2 = H.mean()

    reward = term1 - term2
    return reward.clamp_min(0.0)


if __name__ == '__main__':
    from botorch.models import SingleTaskGP
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.constraints import GreaterThan

    torch.manual_seed(0)
    np.random.seed(0)

    def make_random_gp(dim=2, n_train=8, bounds=(-5.0, 5.0)):
        train_x = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n_train, dim, dtype=DEFAULT_DTYPE)
        train_y = torch.sin(train_x.sum(dim=-1, keepdim=True)) + 0.05 * torch.randn(n_train, 1, dtype=DEFAULT_DTYPE)
        likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-6))
        model = SingleTaskGP(train_X=train_x, train_Y=train_y, likelihood=likelihood)
        model = model.to(dtype=DEFAULT_DTYPE)
        model.eval()
        return model, train_x, train_y

    # --- Test 1: MES reward >= 0 at 50 random locations ---
    model, train_x, train_y = make_random_gp()
    roi_candidates = -5.0 + 10.0 * torch.rand(30, 2, dtype=DEFAULT_DTYPE)
    query_points = -5.0 + 10.0 * torch.rand(50, 2, dtype=DEFAULT_DTYPE)

    violations = 0
    for i in range(query_points.shape[0]):
        reward = compute_mes_reward(query_points[i], model, roi_candidates, K=10)
        if reward.item() < 0.0:
            violations += 1
    print(f"Test 1 - MES reward >= 0 at 50 random locations: violations={violations} (EXPECT: 0)")

    # --- Test 2: MES reward at an already-observed location ---
    observed_x = train_x[0]
    reward_observed = compute_mes_reward(observed_x, model, roi_candidates, K=10)
    print(f"Test 2 - MES reward at an already-observed location: {reward_observed.item():.6e} (EXPECT: small but > 0)")

    # --- Test 3: All y* samples > max(mu_candidates) ---
    roi_posterior = model.posterior(roi_candidates)
    mu_roi = roi_posterior.mean.reshape(-1)
    sigma_roi = roi_posterior.variance.clamp_min(1e-18).sqrt().reshape(-1)
    y_star_samples = gumbel_sample_y_star(mu_roi, sigma_roi, K=10)
    print(f"Test 3 - min(y_star_samples)={y_star_samples.min().item():.4f}, "
          f"max(mu_candidates)={mu_roi.max().item():.4f} "
          f"(EXPECT: min sample > max mean)")

    # --- Test 4: 200 random GP states, count NaN/Inf ---
    nan_inf_count = 0
    for trial in range(200):
        dim = np.random.choice([1, 2, 3, 5])
        n_train = np.random.randint(3, 15)
        model_t, train_x_t, _ = make_random_gp(dim=dim, n_train=n_train)
        roi_t = -5.0 + 10.0 * torch.rand(np.random.randint(5, 40), dim, dtype=DEFAULT_DTYPE)
        x_query = -5.0 + 10.0 * torch.rand(dim, dtype=DEFAULT_DTYPE)
        reward_t = compute_mes_reward(x_query, model_t, roi_t, K=10)
        if torch.isnan(reward_t).any() or torch.isinf(reward_t).any():
            nan_inf_count += 1
    print(f"Test 4 - 200 random GP states, NaN/Inf count: {nan_inf_count} (EXPECT: 0)")

    # --- Test 5: Thompson-sampled y* (now used internally by compute_mes_reward)
    # is USUALLY, not deterministically, above max(mu_roi) -- unlike
    # gumbel_sample_y_star's formula-guaranteed samples (Test 3), a raw
    # Thompson draw's max can occasionally fall below the ROI mean's max,
    # since each sampled point is a genuine random draw around its own mean.
    # This is expected and handled by compute_mes_reward's downstream clamps.
    with torch.no_grad():
        roi_posterior_t5 = model.posterior(roi_candidates)
        thompson_samples = roi_posterior_t5.rsample(torch.Size([500])).reshape(500, -1)
        thompson_y_star = thompson_samples.max(dim=-1).values
    frac_above = (thompson_y_star > mu_roi.max()).float().mean().item()
    print(f"Test 5 - Thompson y* samples above max(mu_candidates): {frac_above:.1%} of 500 "
          f"(EXPECT: high, e.g. >80%, but not necessarily 100% -- unlike Test 3's formula-based samples)")
