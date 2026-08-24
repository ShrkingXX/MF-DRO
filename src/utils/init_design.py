"""
Shared initial-design generators.

Every optimizer in this codebase (MF-DRO, the MF baselines, the KO-MES /
Additive-MES / SF-MES methods) needs to be able to draw the SAME initial
design for a given (benchmark, seed) -- otherwise a difference in final
regret cannot be attributed to the method rather than to which starting
points it happened to get. This module is the single definition of those
designs so there is exactly one implementation to keep in sync.

`sequential_max_variance_design` is lifted verbatim (same lengthscale
convention, same RNG seeding, same candidate-pool size) out of
DirectMFRegretOptimization._sample_initial_points's nested
`_seq_max_var_points`, which remains its only prior caller; that method now
delegates here so MF-DRO's existing behaviour is bit-for-bit unchanged.
"""
import math

import torch
from scipy.stats.qmc import LatinHypercube

DEFAULT_DTYPE = torch.float64
_UCB_CANDIDATE_POOL = 500
_UCB_DELTA = 0.1


def lhs_design(bounds, d, n, seed, seed_offset=0):
    """
    Latin hypercube design, rescaled from the sampler's [0,1]^d unit
    hypercube to the actual domain (a no-op for Currin_2D/Hartmann_6D, both
    already [0,1]^d, but required for Borehole_8D's physical per-dimension
    ranges).

    bounds: [2, d] BoTorch convention (row 0 = lower, row 1 = upper).
    seed_offset: 0 for HF, 1 for LF, so the two fidelities never receive
    identical designs -- the convention every caller in this codebase uses.
    """
    sampler = LatinHypercube(d=d, seed=seed + seed_offset)
    unit_X = torch.tensor(sampler.random(n=n), dtype=DEFAULT_DTYPE)
    return bounds[0] + (bounds[1] - bounds[0]) * unit_X


def sequential_max_variance_design(bounds, d, n, seed, seed_offset=0,
                                    n_candidates=500):
    """
    Sequential max-variance design, replacing LHS. Greedily picks each point
    to maximize GP posterior variance conditioned ONLY on the locations
    already selected -- no y-values needed, since for a GP with a FIXED
    (untrained) kernel, posterior variance depends solely on which X have
    been conditioned on, not their observed values (Var(x) = k(x,x) -
    k(x,X_sel)^T [K(X_sel,X_sel)+jitter*I]^-1 k(X_sel,x), no y anywhere in
    that formula). This lets the whole design be built BEFORE any real
    function evaluation, unlike an actual active-learning loop.

    Uses a fixed RBF lengthscale at the same geometric-mean convention
    KennedyOHaganGP._lengthscale_bounds already establishes (0.05*sqrt(d) to
    2*sqrt(d)), on inputs normalized to [0,1]^d (matching the Normalize
    transform every GP in this codebase uses) so the kernel's distance
    notion is domain-scale-consistent.

    Intended to give better coverage of narrow-basin benchmarks than LHS's
    fixed stratified-random design (motivation: GP miscalibration at the
    true optimum traced to sparse/unlucky LHS draws never landing near it in
    high-d).

    bounds: [2, d] BoTorch convention. Returns [n, d] on the raw domain.
    """
    torch.manual_seed(seed + seed_offset + 1000)
    low = 0.05 * math.sqrt(d)
    high = 2.0 * math.sqrt(d)
    ls = math.sqrt(low * high)

    def _sample_unit(m):
        return torch.rand(m, d, dtype=DEFAULT_DTYPE)

    def _rbf(A, B):
        # A: [p,d], B: [q,d], both already normalized -> [p,q]
        sq = ((A.unsqueeze(1) - B.unsqueeze(0)) ** 2).sum(-1)
        return torch.exp(-0.5 * sq / (ls ** 2))

    selected_unit = [_sample_unit(1)[0]]  # first point: uniform random
    for _ in range(1, n):
        cand_unit = _sample_unit(n_candidates)
        X_sel = torch.stack(selected_unit)
        K_ss = _rbf(X_sel, X_sel) + 1e-6 * torch.eye(
            len(selected_unit), dtype=DEFAULT_DTYPE
        )
        K_sc = _rbf(X_sel, cand_unit)  # [i, n_candidates]
        L = torch.linalg.cholesky(K_ss)
        v = torch.linalg.solve_triangular(L, K_sc, upper=False)  # [i, n_cand]
        var = 1.0 - (v ** 2).sum(dim=0)  # k(x,x)=1 for RBF
        selected_unit.append(cand_unit[torch.argmax(var)])

    unit_X = torch.stack(selected_unit)
    return bounds[0] + (bounds[1] - bounds[0]) * unit_X


def lf_screened_hf_init(bounds, d, n_hf, n_lf, seed, f_lf,
                         n_candidates=_UCB_CANDIDATE_POOL, delta=_UCB_DELTA):
    """
    LF-screened HF initialization: place the n_hf expensive HF init points
    where a cheap LF surrogate says they matter, instead of independent
    random LHS at both fidelities.

    Phase 1: LHS-screen n_lf LF points (seed_offset=1, same convention as
    lhs_design's HF=0/LF=1 split), evaluate f_lf, fit a standalone LF-only
    GP via KennedyOHaganGP._build_gp -- the identical lognormal-prior /
    noise_lb=1e-2 fitting procedure this codebase already uses for gp_lf,
    so the screening GP is fit exactly the way the real one will be.

    Phase 2: sequential greedy UCB selection of n_hf HF locations from that
    GP. At each step t (1-indexed), draw n_candidates fresh uniform
    candidates, score mu + sqrt(beta_t)*sigma with beta_t the Srinivas et
    al. (2010) GP-UCB confidence bound (same formula/defaults as
    MFGPUCBOptimizer._compute_beta_t in src/baselines/mf_baselines.py),
    take the argmax, then fantasize it into the GP as a pseudo-observation
    (posterior mean, kriging-believer heuristic) via
    KennedyOHaganGP._rebuild_frozen_gp's frozen-hyperparameter fantasy
    conditioning -- the same cheap-posterior-update mechanism this
    codebase already uses for KO ensemble fantasy conditioning elsewhere.
    Without this fantasy step every pick would land on the same peak.

    Returns (X_lf, Y_lf, X_hf): X_lf/Y_lf are the n_lf screening
    observations (already evaluated at f_lf). X_hf are the n_hf SELECTED
    raw-scale locations, NOT yet evaluated -- the caller evaluates f_hf
    there, so total evaluation cost (n_lf LF + n_hf HF) is unchanged from
    the plain-LHS init this replaces.
    """
    from src.models.ko_gp import KennedyOHaganGP

    X_lf = lhs_design(bounds, d, n_lf, seed, seed_offset=1)
    Y_lf = torch.tensor(
        [f_lf(x.unsqueeze(0)).reshape(-1)[0].item() for x in X_lf],
        dtype=DEFAULT_DTYPE,
    )

    screen_gp = KennedyOHaganGP(d=d)
    screen_gp.bounds = bounds.to(dtype=DEFAULT_DTYPE)
    model = screen_gp._build_gp(X_lf, Y_lf)

    # Distinct RNG stream from HF(seed_offset=0)/LF(seed_offset=1)'s LHS
    # draws and sequential_max_variance_design's seed+1000 stream.
    torch.manual_seed(seed + 3000)
    X_aug, Y_aug = X_lf.clone(), Y_lf.clone()
    selected = []
    for t in range(1, n_hf + 1):
        cand = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(
            n_candidates, d, dtype=DEFAULT_DTYPE
        )
        beta_t = 2.0 * math.log(
            n_candidates * (t ** 2) * (math.pi ** 2) / (6.0 * delta)
        )
        with torch.no_grad():
            post = model.posterior(cand)
            mu = post.mean.reshape(-1)
            sigma = post.variance.clamp_min(0).sqrt().reshape(-1)
            ucb = mu + (beta_t ** 0.5) * sigma
        best_idx = torch.argmax(ucb)
        x_new = cand[best_idx]
        y_fantasy = mu[best_idx].item()
        selected.append(x_new)
        X_aug = torch.cat([X_aug, x_new.unsqueeze(0)], dim=0)
        Y_aug = torch.cat(
            [Y_aug, torch.tensor([y_fantasy], dtype=DEFAULT_DTYPE)], dim=0
        )
        model = screen_gp._rebuild_frozen_gp(model, X_aug, Y_aug)

    X_hf = torch.stack(selected)
    return X_lf, Y_lf, X_hf


def make_initial_design(bounds, d, n, seed, seed_offset=0,
                        use_sequential_init=False):
    """
    Dispatcher every optimizer calls, so switching the whole experiment
    between LHS and sequential max-variance is one flag rather than an edit
    in each optimizer class.
    """
    if use_sequential_init:
        return sequential_max_variance_design(bounds, d, n, seed, seed_offset)
    return lhs_design(bounds, d, n, seed, seed_offset)
