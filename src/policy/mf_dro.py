"""
Multi-fidelity DRO state extraction and joint (location, fidelity) MES
acquisition, built on top of the Kennedy-O'Hagan two-fidelity GP
(src/models/ko_gp.py).

Deliberately independent of src/policy/dro.py's `_extract_state` -- the MF
state vector has a different structure (per-ensemble-member RAW KO
hyperparameters -- lengthscale/outputscale/rho for gp_lf and gp_delta, since
there are two GPs per member instead of SF-DRO's one -- plus fidelity-cost
features with no SF-DRO analogue). `_extract_state` was read for style
reference only and is never called from this module.
"""
import math
from collections import deque
from types import SimpleNamespace

import torch
import gpytorch
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.stats import norm as scipy_norm

from gumbel_thompson import thompson_sample_y_star, fit_gumbel_to_samples
from mes_reward import compute_mes_reward
from src.models.ko_gp import (
    KennedyOHaganGP, DeepKernel, LENGTHSCALE_PRIOR_LOC, LENGTHSCALE_PRIOR_SCALE
)
from src.model.decisionTransformer import DecisionTransformer

EULER_GAMMA = 0.5772156649015329

# Number of fixed reference points for the state's reference-grid block
# (Fix for Bug A -- see _reference_grid_features/_extract_mf_state's
# docstrings). One module-level constant so _get_mf_state_dim's default and
# DirectMFRegretOptimization.__init__'s actual grid construction can never
# drift out of sync with each other.
STATE_REF_GRID_R = 10
# Reference-set size for rollout_reward='kg_incumbent' (Sobol prefix of
# y_star_pool). Fixed across steps/trajectories so V differences are comparable.
KG_REF_M = 64


def _kg_V(ko, ref, topk):
    """Incumbent-belief statistic for rollout_reward='kg_incumbent'.

    topk=1 is the hard max used by H12, whose gate showed it is insensitive to
    every observation that does not land near the current argmax (only 27.2% of
    LF steps moved it). topk>1 averages the top-k HF posterior means instead, so
    an observation that reshapes the believed-good REGION registers even when it
    does not displace the single best point.
    """
    with torch.no_grad():
        mu = ko.hf_posterior(ref)[0].flatten()
    if topk <= 1:
        return float(mu.max())
    k = min(int(topk), mu.numel())
    return float(mu.topk(k).values.mean())

#: Size of the fixed pool y* is Thompson-drawn from (CRITICAL-1 fix, see
#: _y_star_for_model). Matches compute_joint_mf_mes's own 200-candidate
#: convention; kept separate from STATE_REF_GRID_R because the state block
#: wants a SMALL grid (it costs 4 state dims per point) while y* wants a
#: LARGE one (max over few points underestimates the true max badly in 6D).
Y_STAR_POOL_P = 200


# ════════════════════════════════════
# PART A: State Dimension Helper
# ════════════════════════════════════

def _get_mf_state_dim(d, M=10, R=STATE_REF_GRID_R):
    """
    MF-DRO state dimension.
    Breakdown:
        5*M slots: raw KO hyperparameters per ensemble member (Fix 1 --
                  see _ko_hp_features/_extract_mf_state's docstrings):
                  [gp_lf lengthscale, gp_lf outputscale, gp_delta
                  lengthscale, gp_delta outputscale, rho] x M. Replaces the
                  old 2*M candidate-dependent HF/LF sigma summary, which
                  required a roi_candidates set at state-extraction time --
                  a REAL train/inference mismatch, since training
                  (simulate_mf_trajectory) used 200 unfiltered
                  domain-spanning candidates while inference
                  (_propose_next_query) used a completely different
                  ROI-filtered <=50-candidate pool for the exact same
                  feature slots. Raw hyperparameters need no candidate set
                  at all, eliminating that mismatch entirely.
        1 slot:   best_value_HF
        1 slot:   step_norm
        d slots:  best_position_HF
        1 slot:   recent_hf_frac (rolling window, last 5 fidelity choices)
        1 slot:   c_L / c_H
        1 slot:   use_dkl_flag (1.0 if ko_ensemble[0].use_dkl else 0.0) --
                  tells the DT whether the GP is in RBF or DKL mode, since
                  that changes what its uncertainty estimates mean.
        4*R slots: reference-grid posterior features (Fix for Bug A -- see
                  _reference_grid_features/_extract_mf_state's docstrings):
                  [mu_H, sigma_H, mu_L, sigma_L] at each of R fixed
                  reference points, from the CALLER-SPECIFIED single model
                  (the rollout's own progressively-conditioned current_ko
                  during training, ko_ensemble[0] at real inference) -- NOT
                  the shared, unconditioned ko_ensemble the 5*M block above
                  reads. This is the block that actually varies within a
                  batch/rollout; the 5*M block above is constant across an
                  entire real BO iteration's whole rollout batch (see Bug A).
    (The old separate "mean_rho across ensemble" slot is dropped -- rho is
    now already present per-member in the 5*M block above, so a separate
    ensemble-mean would be redundant, reconstructable by the DT itself.)
    Total: 5*M + d + 5 + 4*R
    """
    return 5 * M + d + 5 + 4 * R


def _reference_grid_features(ko, ref_grid, c_H, c_L, y_star_arr):
    """
    Fix for Bug A (state-collapse-at-tau=0): [mu_H, sigma_H, MES_H/c_H,
    MES_L/c_L] (Change 2a -- was [mu_H, sigma_H, mu_L, sigma_L]; MES/cost
    replaces the raw LF posterior so this block encodes the SAME
    cost-normalized-information-gain quantity the policy is actually
    trained to chase, at each of R fixed reference points, from ONE
    specific model. Unlike the 5*M ko_ensemble block in _extract_mf_state
    (always read from ko_ensemble_full/self.ko_ensemble, the SAME shared,
    unconditioned ensemble object for every trajectory in an entire rollout
    batch), this block is computed from whatever model the CALLER passes
    (current_ko during a training rollout -- reconditioned every step via
    make_fantasy_ko, so genuinely differs across ensemble members AND across
    steps within one rollout; ko_ensemble[0] at real inference, matching the
    single-representative-model convention _refine_proposal already uses).

    NOTE (verified directly, see conversation): switching ref_model from
    ko_ensemble_full to current_ko/ko_ensemble[0] alone only reduces the
    tau=0 collapse from 1 group of ~70 identical states to M=10 groups of
    rollouts_per_model=7 identical states each (current_ko == ko_model, the
    SAME object, for every one of the rollouts_per_model trajectories drawn
    from one ensemble member, until step 5's first make_fantasy_ko call --
    which happens AFTER tau=0's state is already recorded). This block
    alone does not fully solve Bug A; Change 1 (candidate-conditioned
    scoring) is the fix for the remaining within-group collapse.

    ref_grid: [R, d] RAW domain-scale tensor (ko.hf_posterior/lf_posterior
    expect raw-domain X and normalize internally via their own
    input_transform, exactly like every other consumer of a KO model in
    this module) -- a FIXED set of points, generated once per real BO run
    and reused identically by every rollout/iteration and by real inference,
    so train and inference share the exact same reference distribution (the
    mismatch that motivated removing the old, candidate-dependent 2*M block
    in the first place -- see _get_mf_state_dim's docstring).

    mu_H is returned RAW (unstandardized) -- _extract_mf_state applies
    Change 2b's z-scoring (against data_hf_y's running mean/std) uniformly
    to this and to best_value_HF, since both are only meaningful relative
    to that same y-scale and _extract_mf_state is where data_hf_y is
    available.

    Returns a flat list of 4*R floats: [mu_H_1, sigma_H_1, mes_h_over_cH_1,
    mes_l_over_cL_1, mu_H_2, ...].
    """
    mu_H, sigma_H, _mu_L, _sigma_L, mes_h_over_cH, mes_l_over_cL = \
        _gp_candidate_features(ko, ref_grid, c_H, c_L, y_star_arr)
    feats = torch.stack([mu_H, sigma_H, mes_h_over_cH, mes_l_over_cL], dim=-1)  # [R, 4]
    return feats.reshape(-1).tolist()


def _ko_hp_features(ko, log_transform=True):
    """
    Raw hyperparameter features for one KO ensemble member (Fix 1):
    [gp_lf lengthscale (mean over ARD dims), gp_lf outputscale,
     gp_delta lengthscale, gp_delta outputscale, rho].

    Handles both covar_module shapes in this codebase: plain RBF
    (ScaleKernel(RBFKernel(...))) and DKL (DeepKernel, whose OWN
    .base_kernel attribute is itself a ScaleKernel(RBFKernel(...)) one
    level deeper -- see ko_gp.py's DeepKernel.__init__). Naively reading
    covar_module.base_kernel.lengthscale / covar_module.outputscale (correct
    for plain RBF) raises AttributeError under DKL, since DeepKernel has no
    top-level .outputscale and its .base_kernel is a ScaleKernel, not an
    RBFKernel.

    log_transform (Change 2b, default True -- gated by config's state-fix
    flag, see _extract_mf_state/DirectMFRegretOptimization.__init__):
    lengthscale/outputscale are strictly positive, often spanning multiple
    orders of magnitude (the LogNormalPrior itself is fit in log-space --
    see ko_gp.py), so feeding them to the DT on their raw, right-skewed
    scale gives the state a badly-conditioned dynamic range on 4 of its
    5*M ko-hp slots. rho is left alone -- already in (0,1) via sigmoid, not
    skewed the same way.
    """
    def _ls_os(covar_module):
        scale_kernel = covar_module.base_kernel if isinstance(covar_module, DeepKernel) else covar_module
        ls = scale_kernel.base_kernel.lengthscale.mean().item()
        os_ = scale_kernel.outputscale.item()
        if log_transform:
            ls = math.log(max(ls, 1e-8))
            os_ = math.log(max(os_, 1e-8))
        return ls, os_

    ls_lf, os_lf = _ls_os(ko.gp_lf.covar_module)
    ls_delta, os_delta = _ls_os(ko.gp_delta.covar_module)
    return [ls_lf, os_lf, ls_delta, os_delta, ko.rho.item()]


# ════════════════════════════════════
# PART B: MF State Extraction
# ════════════════════════════════════

def _extract_mf_state(data_hf_x, data_hf_y,
                       data_lf_x, data_lf_y,
                       ko_ensemble,
                       n_real_iter, T_real,
                       c_L, c_H, device, dtype,
                       recent_hf_frac, bounds,
                       ref_grid, ref_model, y_star_arr, standardize=True):
    """
    Build MF state vector. Does NOT call _extract_state from dro.py -- this
    is an entirely independent function.

    standardize (Change 2b, default True -- gated by
    config.use_state_standardization via the callers): z-scores the two
    y-scale quantities (best_value_HF, and the reference block's mu_H
    entries) against data_hf_y's own running mean/std, and log-transforms
    lengthscale/outputscale in _ko_hp_features. Applied identically on the
    training path (simulate_mf_trajectory) and the inference path
    (_propose_next_query) -- both call this one function, so there is no
    second implementation to drift.

    data_hf_x: list of [d] tensors (may be empty)
    data_hf_y: list of scalar floats (may be empty)
    data_lf_x: list of [d] tensors (may be empty)
    data_lf_y: list of scalar floats (may be empty)
    ko_ensemble: list of KennedyOHaganGP (length M), already fit
    recent_hf_frac: float, fraction of HF choices among the last 5
    fidelity decisions (real queries at inference, or this rollout's own
    prior simulated steps during training-data generation). Replaces the
    previous n_hf/(n_hf+n_lf) feature, which was self-reinforcing: once an
    LF run started, that cumulative ratio only decayed monotonically toward
    0 and could never recover even if recent decisions turned HF-heavy, and
    could not be reset by e.g. cold-start HF seeding overriding just the
    first few queries -- computed and passed in by the caller since chosen
    fidelities are chronologically ordered (real queries in
    DirectMFRegretOptimization.recent_ell_history, rollout steps in
    simulate_mf_trajectory's own local actions_ell) in a way data_hf_x/
    data_lf_x (split by fidelity, cumulative-only) cannot reconstruct.
    ref_grid: [R, d] RAW domain-scale tensor, FIXED for an entire real BO
    run (see DirectMFRegretOptimization.__init__'s self.state_ref_grid) --
    passed straight through to _reference_grid_features.
    ref_model: ONE KennedyOHaganGP -- the rollout's own current_ko during
    training (reconditioned every step, differs per ensemble member and
    diverges further per rollout as fantasy data accumulates), or
    ko_ensemble[0] at real inference. See _reference_grid_features's
    docstring for why this needs to be a SINGLE, caller-chosen model rather
    than reading ko_ensemble (the fix for Bug A specifically).

    No roi_candidates param for the ko_ensemble block (Fix 1): the old 2*M
    sigma-over-candidates block needed one, and training/inference passed
    DIFFERENT candidate distributions for it (simulate_mf_trajectory's 200
    unfiltered domain-spanning points vs. _propose_next_query's ROI-filtered
    <=50-point pool) -- a genuine train/inference covariate shift for those
    slots. Raw KO hyperparameters (_ko_hp_features) need no candidates at
    all. The reference-grid block below deliberately reintroduces a
    candidate-like (posterior-at-fixed-points) feature, but with ONE FIXED
    ref_grid shared identically by training and inference, avoiding that
    exact mismatch this time (see _reference_grid_features's docstring).

    Returns: Tensor [5*M + d + 5 + 4*R]
    """
    M = len(ko_ensemble)
    d = ko_ensemble[0].d
    state_list = []

    # Change 2b: y-scale standardization statistics, from the SAME
    # data_hf_y this state describes. Used for best_value_HF below and for
    # the reference block's mu_H entries -- both live on the objective's
    # own value scale, which varies by orders of magnitude across
    # benchmarks (Hartmann ~3, Borehole ~300). std floored so a
    # near-constant early data_hf_y can't blow the z-scores up.
    if standardize and data_hf_y:
        _y_mean = float(np.mean(data_hf_y))
        _y_std = max(float(np.std(data_hf_y)), 1e-6)
    else:
        _y_mean, _y_std = 0.0, 1.0

    # Slots 0..5M-1: raw KO hyperparameters per ensemble member (Fix 1) --
    # see _ko_hp_features's docstring for the exact 5 values and the
    # RBF-vs-DKL covar_module handling.
    for ko in ko_ensemble:
        state_list.extend(_ko_hp_features(ko, log_transform=standardize))

    # Slot 5M: best_value_HF (z-scored against data_hf_y when standardize)
    best_val = max(data_hf_y) if data_hf_y else 0.0
    state_list.append((float(best_val) - _y_mean) / _y_std)

    # Slot 5M+1: step_norm
    state_list.append(float(n_real_iter) / max(float(T_real), 1.0))

    # Slots 5M+2 .. 5M+1+d: best_position_HF, normalized to [0,1]^d by
    # bounds -- a no-op for [0,1]^d domains (Currin_2D/Hartmann_6D) but
    # required for any other domain (e.g. Borehole_8D, per-dimension
    # ranges up to [100,50000]): feeding raw domain-scale coordinates as
    # a DT input feature caused training divergence (L_loc exploding into
    # the billions), confirmed directly on Borehole_8D. See REVISION_LOG.md.
    if data_hf_y:
        best_idx = int(np.argmax(data_hf_y))
        best_pos_raw = data_hf_x[best_idx]
        best_pos = ((best_pos_raw - bounds[0]) / (bounds[1] - bounds[0])).tolist()
    else:
        best_pos = [0.0] * d
    state_list.extend(best_pos)

    # Slot 5M+2+d: recent_hf_frac (rolling window, last 5 fidelity choices)
    state_list.append(float(recent_hf_frac))

    # Slot 5M+3+d: c_L / c_H
    state_list.append(float(c_L) / float(c_H))

    # Slot 5M+4+d: use_dkl_flag -- whether the ensemble's GPs are in RBF
    # or DKL mode (see KennedyOHaganGP.fit's dkl_threshold switch). All
    # ensemble members switch together (same n_hf triggers fit() for all),
    # so ko_ensemble[0] is representative of the whole ensemble.
    use_dkl_flag = 1.0 if (ko_ensemble and getattr(ko_ensemble[0], 'use_dkl', False)) else 0.0
    state_list.append(use_dkl_flag)

    # Slots 5M+5 .. 5M+4+4R: reference-grid features (Fix for Bug A,
    # Change 2a: [mu_H, sigma_H, MES_H/c_H, MES_L/c_L] per point) -- see
    # _reference_grid_features's docstring. mu_H comes back raw; z-score it
    # here (Change 2b) against the same data_hf_y statistics best_value_HF
    # used, since both are on the objective's value scale. Layout is
    # [mu_H, sigma_H, mes_h, mes_l] repeating, so mu_H is every 4th entry
    # starting at 0.
    ref_feats = _reference_grid_features(ref_model, ref_grid, c_H, c_L, y_star_arr)
    if standardize:
        # Layout per point: [mu_H, sigma_H, MES_H/c_H, MES_L/c_L].
        # ISSUE-3 FIX: sigma_H (i%4==1) is ALSO on the objective's value
        # scale and was previously left raw -- only mu_H was scaled. mu_H is
        # centered+scaled; sigma_H is a spread, so scale-only. The two MES
        # columns are cost-normalized nats and stay untouched.
        ref_feats = [
            ((v - _y_mean) / _y_std) if (i % 4 == 0)
            else (v / _y_std) if (i % 4 == 1)
            else v
            for i, v in enumerate(ref_feats)
        ]
    state_list.extend(ref_feats)

    return torch.tensor(state_list, device=device, dtype=dtype)


# ════════════════════════════════════
# PART C: HF Proxy Model for Thompson Sampling
# ════════════════════════════════════

def _build_hf_proxy_model(ko_model):
    """
    Correct joint HF posterior via composition of gp_lf and gp_delta.
    f_H = rho * f_L + delta, with f_L independent of delta.

    SUPERSEDES an earlier diagonal-marginal version of this function, which
    stored only per-candidate mean/variance (ko.hf_posterior's marginal
    output) and sampled each candidate independently in rsample() -- that
    breaks compute_mes_reward/thompson_sample_y_star's Thompson-sampled
    y*=max(f_H) estimate, since the max of independent per-candidate draws
    has a wider, less-informative distribution than the max of a properly
    correlated joint GP draw (the same independence-vs-joint distinction
    mes_reward.py's own docstring documents as the reason compute_mes_reward
    uses Thompson sampling over the product-CDF/Gumbel approximation in the
    first place -- this proxy had silently reintroduced that exact flaw for
    the HF term specifically). An inflated y* spread systematically shrinks
    the MES reward (more Thompson draws land in the weakly-informative
    large-gamma tail of the truncated-Gaussian entropy term), so HF MES came
    out underestimated relative to LF MES (whose "gp_lf" branch already used
    a real joint posterior, unaffected by this bug) -- biasing the
    cost-normalized joint acquisition toward LF regardless of cost ratio.

    Since gp_lf and gp_delta are both real SingleTaskGP objects, their own
    .posterior(X).rsample(sample_shape) calls already draw true joint
    samples (full Cholesky-factored covariance, not diagonal). Because
    f_L and delta are independent by construction, the sum of independent
    joint samples IS itself an exact joint sample of f_H = rho*f_L + delta
    -- no additional approximation beyond what hf_posterior already assumes
    (f_L independent of delta).
    """
    class HFJointPosterior:
        def __init__(self, post_lf, post_delta, rho_val):
            self._post_lf = post_lf
            self._post_delta = post_delta
            self._rho = rho_val

        @property
        def mean(self):
            return (self._rho * self._post_lf.mean.reshape(-1)
                    + self._post_delta.mean.reshape(-1))

        @property
        def variance(self):
            return (self._rho ** 2 * self._post_lf.variance.reshape(-1)
                    + self._post_delta.variance.reshape(-1))

        def rsample(self, sample_shape=torch.Size()):
            z_lf = self._post_lf.rsample(sample_shape)       # joint, [..., N, 1]
            z_delta = self._post_delta.rsample(sample_shape)  # joint, [..., N, 1]
            return self._rho * z_lf + z_delta                 # joint, [..., N, 1]

    class HFProxyModel:
        def __init__(self, ko):
            self._ko = ko

        def posterior(self, X):
            with torch.no_grad():
                post_lf = self._ko.gp_lf.posterior(X)
                post_delta = self._ko.gp_delta.posterior(X)
                rho_val = self._ko.rho.item()
            return HFJointPosterior(post_lf, post_delta, rho_val)

    return HFProxyModel(ko_model)


# ════════════════════════════════════
# PART D: Joint MES Acquisition
# ════════════════════════════════════

def _lf_mes_info_gain(x, ko_model, y_star_hf_arr, n_quad=32):
    """
    LF MES info gain I(f_L(x); f* | D), following Takeno et al. 2020
    (arXiv:1901.08275) Lemma 3.1's decomposition H0 - H1 under the KO
    model's joint Gaussianity (f_H = rho*f_L + delta, f_L independent of
    delta, so Cov(f_L(x), f_H(x)) = rho*Var(f_L(x)) and
    f_H(x) | f_L(x)=v ~ N(rho*(v-mu_L(x))+mu_H(x), sigma_delta^2(x))).

    NUMERICAL CORRECTION vs. the originally given pseudocode: the given
    Z = 1/(sigma_L*Phi_H) and final scaling `sigma_L*sqrt(pi)` do not form a
    self-consistent quadrature for numpy.polynomial.hermite_e.hermegauss
    (probabilist's Hermite, weight e^{-t^2/2}) -- verified two ways before
    fixing: (1) empirically, applying that exact node/scaling combination to
    a simple E[v^2] test integral under a known Gaussian gives 41.65 instead
    of the true 6.25; (2) analytically, the correct conditional density is
    q(v) = phi_L(v)*Phi_cond(v)/Phi_H (Bayes' rule on the truncation event
    f_H(x)<=f*), which needs Z=1/Phi_H, not 1/(sigma_L*Phi_H) -- the given Z
    has a spurious extra 1/sigma_L that leaves q(v) not integrating to 1.
    Fixed here by keeping hermegauss's own probabilist's convention (nodes
    v_i = mu_L + sigma_L*x_i, matching the given v_nodes) but computing the
    expectation E_{v~N(mu_L,sigma_L^2)}[g(v)] as sum(w_i*g(v_i))/sum(w_i)
    (normalizing by the total quadrature weight directly, rather than an
    assumed closed-form scaling constant) -- verified against the same test
    integral (recovers 6.25 exactly) and against the identity that should
    hold when Phi_cond(v)==1 everywhere (truncation event certain -> zero
    info gain, i.e. H1 must equal H0 exactly; confirmed to machine
    precision before use here).

    y_star_hf_arr: np.ndarray [K], Thompson samples of f* from the HF proxy.
    Returns: float >= 0
    """
    x_in = x.unsqueeze(0) if x.dim() == 1 else x

    with torch.no_grad():
        mu_L, var_L = ko_model.lf_posterior(x_in)
        mu_H, var_H = ko_model.hf_posterior(x_in)
        var_delta = ko_model.gp_delta.posterior(x_in).variance
    mu_L = mu_L.item()
    sigma_L = max(var_L.sqrt().item(), 1e-12)
    mu_H = mu_H.item()
    sigma_H = max(var_H.sqrt().item(), 1e-12)
    sigma_delta = max(var_delta.sqrt().item(), 1e-12)
    rho = ko_model.rho.item()

    LOG_SQRT_2PIE = 0.5 * math.log(2.0 * math.pi * math.e)
    H0 = math.log(sigma_L) + LOG_SQRT_2PIE  # marginal entropy of f_L(x)

    xi, wi = hermegauss(n_quad)  # probabilist's Hermite (weight e^{-t^2/2})
    v_nodes = sigma_L * xi + mu_L  # maps to N(mu_L, sigma_L^2)
    phi_L_vals = scipy_norm.pdf(v_nodes, loc=mu_L, scale=sigma_L)
    wi_sum = wi.sum()

    h1_per_fstar = []
    for f_star in y_star_hf_arr:
        gamma_H = (f_star - mu_H) / sigma_H
        Phi_H = scipy_norm.cdf(gamma_H)
        if Phi_H < 1e-300:
            h1_per_fstar.append(H0)  # degenerate: treat as zero info gain
            continue

        u_v = rho * (v_nodes - mu_L) + mu_H  # E[f_H | f_L=v], KO model
        Phi_cond = scipy_norm.cdf((f_star - u_v) / sigma_delta)
        Psi_v = Phi_cond * phi_L_vals
        q_v = Psi_v / Phi_H  # properly normalized conditional density q(v)

        # g(v) = (Phi_cond(v)/Phi_H) * log(q(v)); x*log(x)->0 as x->0, so
        # skip (not clamp-and-multiply-by-zero, which risks 0*(-inf)=nan)
        # wherever q_v is negligible.
        g_vals = np.zeros_like(v_nodes)
        safe = q_v > 1e-300
        g_vals[safe] = (Phi_cond[safe] / Phi_H) * np.log(q_v[safe])
        H1_fstar = -float(np.sum(wi * g_vals) / wi_sum)
        h1_per_fstar.append(H1_fstar)

    H1 = float(np.mean(h1_per_fstar))
    return max(H0 - H1, 0.0)


def _compute_mes_hf_vectorized(roi_candidates, hf_proxy, y_star_arr):
    """
    Vectorized HF MES for all candidates simultaneously, replacing the
    per-candidate compute_mes_reward() loop in compute_joint_mf_mes (each
    call of which drew its own fresh Thompson sample independently -- N_roi
    redundant resamplings of the same y* distribution per acquisition call).

    hf_proxy: the composed joint HF posterior from _build_hf_proxy_model,
    NOT ko_model.hf_posterior directly -- ko_model.hf_posterior's marginal
    mean/variance is exactly the "diagonal-marginal" approach
    _build_hf_proxy_model's own docstring documents as previously fixed
    (inflated y* spread from ignoring the f_L/delta joint correlation).
    Using hf_proxy.posterior() here matches what compute_mes_reward already
    did internally.

    y_star_arr: np.ndarray [K] -- precomputed Thompson samples (shared with
    the LF branch, drawn once by the caller).
    Returns: np.ndarray [N_roi] of MES values, all >= 0.

    CORRECTED vs. the originally given pseudocode: that version defined
    H1 = mean_k[ log(sigma_H*Phi) - gamma*phi/(2*Phi) ], omitting the
    "+ LOG_SQRT_2PIE" term paper Eq. (4) requires in the truncated entropy
    (H0 has it, H1 didn't) -- H0 and H1 do NOT cancel unless that term is
    present in both (they always are within compute_mes_reward's original
    per-candidate scalar version, where term1/term2 share the identical
    log(sigma_x)+log_sqrt_2pi_e prefix). The omission added a constant
    +log(sqrt(2*pi*e)) (~1.42 nats) to every HF MES value -- verified by
    direct algebraic comparison against compute_mes_reward's term1-term2
    cancellation, which contains no such standalone additive constant.
    """
    with torch.no_grad():
        posterior = hf_proxy.posterior(roi_candidates)
        mu_H = posterior.mean.reshape(-1).numpy()
        var_H = posterior.variance.clamp_min(1e-12).reshape(-1).numpy()
    sigma_H = np.sqrt(var_H)

    LOG_SQRT_2PIE = 0.5 * np.log(2.0 * np.pi * np.e)

    # [N, K]
    gamma = (y_star_arr[None, :] - mu_H[:, None]) / sigma_H[:, None]
    phi = scipy_norm.pdf(gamma)
    Phi = np.maximum(scipy_norm.cdf(gamma), 1e-300)

    H0 = np.log(sigma_H) + LOG_SQRT_2PIE  # [N]
    H1 = np.mean(
        np.log(sigma_H[:, None]) + LOG_SQRT_2PIE + np.log(Phi)
        - gamma * phi / (2.0 * Phi),
        axis=1,
    )  # [N]
    return np.maximum(H0 - H1, 0.0)


def _compute_mes_lf_vectorized(roi_candidates, ko_model, y_star_arr, n_quad=32):
    """
    Vectorized LF MES (Takeno 2020 Lemma 3.1) for ALL candidates
    simultaneously, replacing the per-candidate _lf_mes_info_gain loop.
    Verified term-for-term equivalent to _lf_mes_info_gain's scalar
    computation (same H0, same per-f_star quadrature contribution, only the
    candidate dimension is vectorized).

    y_star_arr: np.ndarray [K] -- shared HF Thompson samples.
    Returns: np.ndarray [N_roi], all >= 0.
    """
    LOG_SQRT_2PIE = 0.5 * np.log(2.0 * np.pi * np.e)

    with torch.no_grad():
        mu_L, var_L = ko_model.lf_posterior(roi_candidates)
        mu_H, var_H = ko_model.hf_posterior(roi_candidates)
        var_delta = ko_model.gp_delta.posterior(roi_candidates).variance.reshape(-1)
    mu_L = mu_L.numpy()
    sigma_L = np.sqrt(np.maximum(var_L.numpy(), 1e-12))
    mu_H = mu_H.numpy()
    sigma_H = np.sqrt(np.maximum(var_H.numpy(), 1e-12))
    sigma_d = np.sqrt(np.maximum(var_delta.numpy(), 1e-12))
    rho = ko_model.rho.item()

    N = roi_candidates.shape[0]
    H0 = np.log(sigma_L) + LOG_SQRT_2PIE  # [N]

    xi, wi = hermegauss(n_quad)
    wi_sum = wi.sum()

    v_nodes = mu_L[:, None] + sigma_L[:, None] * xi[None, :]  # [N, n_quad]
    phi_L = scipy_norm.pdf(v_nodes, loc=mu_L[:, None], scale=sigma_L[:, None])

    K = len(y_star_arr)
    H1 = np.zeros(N)

    for f_star in y_star_arr:
        gamma_H = (f_star - mu_H) / sigma_H  # [N]
        Phi_H = np.maximum(scipy_norm.cdf(gamma_H), 1e-300)  # [N]

        u_v = rho * (v_nodes - mu_L[:, None]) + mu_H[:, None]  # [N, n_quad]
        Phi_cond = scipy_norm.cdf((f_star - u_v) / sigma_d[:, None])  # [N, n_quad]

        Psi_v = Phi_cond * phi_L  # [N, n_quad]
        q_v = Psi_v / Phi_H[:, None]  # [N, n_quad]

        safe = q_v > 1e-300
        g_v = np.zeros_like(q_v)
        g_v[safe] = (Phi_cond[safe] / Phi_H[np.where(safe)[0]]) * np.log(q_v[safe])

        H1 += -np.sum(wi[None, :] * g_v, axis=1) / wi_sum

    H1 /= K
    return np.maximum(H0 - H1, 0.0)


def _y_star_for_model(ko, y_star_pool, K=64, seed=0):
    """
    CRITICAL-1 FIX: one y* draw per (model, BO-iteration), from a FIXED
    reference pool, used by every feature computation that iteration.

    Why this exists: _gp_candidate_features used to call
    thompson_sample_y_star(hf_proxy, X, K) on whatever X it was featurizing.
    y* = max over a joint posterior draw at |X| points, so BOTH its location
    and its scale depend systematically on |X| -- and the three call sites
    pass 10 (ref_grid), K_cands=20 (training candidates) and
    n_infer_candidates=200 (inference candidates) points respectively. Since
    MES enters through gamma = (y* - mu)/sigma, the MES_H/c_H and MES_L/c_L
    feature columns were on different effective scales at training vs
    inference, and the 10-point ref_grid case badly underestimated the true
    max in 6D. Sharing one implementation did NOT prevent this, because the
    implementation's output depended on the size of its input.

    y_star_pool: [P, d] RAW domain-scale, FIXED for the whole run (see
    DirectMFRegretOptimization.__init__'s self.y_star_pool -- seeded Sobol,
    P~200, deliberately NOT the 10-point state_ref_grid).

    Determinism: thompson_sample_y_star draws from the GP posterior, so it is
    stochastic. The RNG state is saved, seeded with `seed`, and restored
    around the draw, making the returned y* a pure function of (ko,
    y_star_pool, K, seed) -- required so the MES feature columns are
    deterministic given the model (previously they were not; two calls with
    an identical model differed by ~0.02, which also contaminated the
    uniq_tau0_states diagnostic). This seeds ONLY this draw and restores the
    prior state immediately, so it does not perturb any other RNG consumer.
    """
    hf_proxy = _build_hf_proxy_model(ko)
    _rng_state = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        y_star_arr = thompson_sample_y_star(hf_proxy, y_star_pool, K=K)
    finally:
        torch.set_rng_state(_rng_state)
    return y_star_arr


def _gp_candidate_features(ko, X, c_H, c_L, y_star_arr):
    """
    Shared GP/MES feature extractor (Change 1c + Change 2a): returns
    (mu_H, sigma_H, mu_L, sigma_L, MES_H/c_H, MES_L/c_L), each [N], for the
    N rows of X under a SINGLE model ko.

    Deliberately ONE function used by BOTH the training path
    (simulate_mf_trajectory's per-candidate block) and the inference path
    (_propose_next_query -> propose_mf's per-candidate block), and by
    _reference_grid_features for the state's reference block -- the spec's
    "these features must be computed identically on both paths" requirement
    is enforced structurally here rather than by two parallel
    implementations that could drift.

    Reuses _compute_mes_hf_vectorized/_compute_mes_lf_vectorized unchanged
    (same Takeno-2020 math, same shared-Thompson-samples convention
    compute_joint_mf_mes already uses -- both branches consume ONE y_star_arr
    draw, not two independent ones). Does NOT modify compute_joint_mf_mes or
    the MES math itself.

    X: [N, d] RAW domain-scale (ko.hf_posterior/lf_posterior and both MES
    helpers all expect raw-domain input and normalize internally).
    y_star_arr: np.ndarray [K], REQUIRED -- computed ONCE per (model,
    iteration) by _y_star_for_model from a fixed pool, then passed to every
    call site. Deliberately has no default: a missing y_star_arr must be a
    TypeError, never a silent per-call-site fallback (that fallback was
    CRITICAL-1 itself -- see _y_star_for_model's docstring).
    Returns 6 tensors of shape [N], dtype matching X.
    """
    hf_proxy = _build_hf_proxy_model(ko)

    with torch.no_grad():
        mu_H, var_H = ko.hf_posterior(X)
        mu_L, var_L = ko.lf_posterior(X)
    sigma_H = var_H.clamp_min(0.0).sqrt()
    sigma_L = var_L.clamp_min(0.0).sqrt()

    mes_h = torch.tensor(_compute_mes_hf_vectorized(X, hf_proxy, y_star_arr),
                          dtype=X.dtype, device=X.device)
    mes_l = torch.tensor(_compute_mes_lf_vectorized(X, ko, y_star_arr),
                          dtype=X.dtype, device=X.device)
    return mu_H, sigma_H, mu_L, sigma_L, mes_h / c_H, mes_l / c_L


def build_candidate_features(ko, X_raw, bounds, c_H, c_L, best_pos_norm,
                             y_star_arr, y_mean=0.0, y_std=1.0):
    """
    Per-candidate feature block (Change 1c): for each of the K rows of
    X_raw, [x_norm (d dims), mu_H, sigma_H, mu_L, sigma_L, MES_H/c_H,
    MES_L/c_L, ||x_norm - best_pos_norm||] -> [K, d+7].

    Used IDENTICALLY by simulate_mf_trajectory (training-data generation,
    with ko=current_ko) and by DirectMFRegretOptimization._propose_next_query
    (real inference, with ko=ko_ensemble[0]) -- one implementation, so the
    two paths cannot drift apart.

    X_raw: [K, d] RAW domain-scale candidates. The returned block's first d
    columns are the [0,1]^d-NORMALIZED coordinates (matching what
    score_head/propose_mf have always consumed and what actions_x uses --
    see simulate_mf_trajectory step 6's own normalization comment), while
    the GP/MES features are computed on the raw-domain X_raw the KO model
    expects. best_pos_norm: [d] normalized incumbent location.
    """
    X_norm = ((X_raw - bounds[0]) / (bounds[1] - bounds[0])).clamp(0, 1)
    mu_H, sigma_H, mu_L, sigma_L, _mes_h_c, _mes_l_c = _gp_candidate_features(
        ko, X_raw, c_H, c_L, y_star_arr
    )
    # PROBE-A REMOVAL: MES_H/c_H and MES_L/c_L dropped from the CANDIDATE
    # feature block (d+7 -> d+5). Probe A measured that softmax(standardized(
    # max(these two columns))) correlated 0.80 (mean, min -0.09 across 560
    # steps) with the actual softmax(standardized(teacher_scores)) target --
    # i.e. two of the seven columns are a noisy but substantial reconstruction
    # of the label the score head is being trained to match, computed from
    # the SAME KO model/y* pool the teacher itself is built from. Left in the
    # STATE's reference-grid block (_reference_grid_features) unchanged --
    # that block feeds an aggregate SUMMARY of belief across R fixed points
    # into the state description, not per-candidate features paired 1:1 with
    # the exact label being predicted, so the leakage concern does not
    # transfer there the same way.
    mu_H = (mu_H - y_mean) / y_std
    mu_L = (mu_L - y_mean) / y_std
    sigma_H = sigma_H / y_std
    sigma_L = sigma_L / y_std
    dist_inc = (X_norm - best_pos_norm.to(dtype=X_norm.dtype)).norm(dim=-1)
    return torch.cat([
        X_norm,
        torch.stack([mu_H, sigma_H, mu_L, sigma_L, dist_inc], dim=-1),
    ], dim=-1)


#: Number of extra per-candidate feature columns build_candidate_features
#: appends beyond the d raw coordinates (Change 1c) -- score_head's input
#: width is hidden_size + action_dim + this. One constant so the model and
#: the feature builder cannot disagree.
N_CAND_EXTRA_FEATURES = 5


def compute_joint_mf_mes(ko_model, roi_candidates, c_H, c_L, K=10):
    """
    Joint MF-MES acquisition (Takeno et al. 2020, arXiv:1901.08275,
    Algorithm 1). Selects (x_tau, ell_tau) = argmax_{x,ell}
    InfoGain(x,ell)/c(ell) using the EXACT MF-MES formula for LF (via
    _lf_mes_info_gain), not the variance-ratio approximation this function
    used previously.

    HF branch: _compute_mes_hf_vectorized (vectorized across candidates,
    corrected from Takeno Eq. 4's truncated entropy -- see its docstring).
    LF branch: _compute_mes_lf_vectorized (vectorized version of
    _lf_mes_info_gain's Lemma 3.1 1D Gauss-Hermite quadrature).
    Both branches use SHARED Thompson-sampled y*_H draws (same K samples
    feed both, avoiding a second independent Monte Carlo estimate of the
    same global-optimum distribution) -- previously true only for the LF
    branch, since compute_mes_reward's per-candidate loop drew its own
    fresh Thompson sample on every call.

    roi_candidates is expected to already be capped to a small N_roi (e.g.
    simulate_mf_trajectory's 200-point pool below) -- the vectorization
    removes the O(N_roi) Python-loop overhead, but posterior/quadrature
    tensors are still O(N_roi) in memory, so this isn't a substitute for
    capping N_roi.

    Returns:
        x_tau:   [d] tensor
        ell_tau: int 0=L, 1=H
        scores:  [N_roi, 2] tensor (col0=LF, col1=HF)
    """
    # Shared HF y* samples (used for both HF and LF branches)
    hf_proxy = _build_hf_proxy_model(ko_model)
    y_star_arr = thompson_sample_y_star(hf_proxy, roi_candidates, K=K)

    # HF MES at all candidates, vectorized, shared y_star_arr (no more
    # per-candidate independent Thompson resampling).
    mes_hf_arr = _compute_mes_hf_vectorized(roi_candidates, hf_proxy, y_star_arr)
    mes_hf = torch.tensor(mes_hf_arr, dtype=roi_candidates.dtype)

    # LF MES at all candidates, vectorized (Takeno 2020 Lemma 3.1).
    mes_lf_arr = _compute_mes_lf_vectorized(roi_candidates, ko_model, y_star_arr, n_quad=32)
    mes_lf = torch.tensor(mes_lf_arr, dtype=roi_candidates.dtype)

    # Cost-normalized scores [N, 2]
    scores = torch.stack([mes_lf / c_L, mes_hf / c_H], dim=1)

    flat_best = scores.reshape(-1).argmax()
    cand_idx = (flat_best // 2).item()
    fid_idx = (flat_best % 2).item()  # 0=L, 1=H

    return roi_candidates[cand_idx], fid_idx, scores


def simulate_mf_trajectory(ko_model, real_data_hf, real_data_lf,
                            rollout_length, c_H, c_L, bounds,
                            n_real_iter, T_real,
                            ko_ensemble_full,
                            ref_grid,
                            K_rtg=100,
                            device='cpu', dtype=torch.float64,
                            minimum_hf_fraction=0.25,
                            use_rtg_grounding=False,
                            bes_delta=0.05,
                            use_candidate_scoring=False,
                            K_cands=20,
                            rollout_policy="mes",
                            # ITEM 1: default switched to regret-based RTG
                            # (forward-sum of per-step improvement over the
                            # simulated incumbent, normalized -- see the
                            # rollout_reward=="improvement" branch below).
                            # mes_entropy (the old default) stays available
                            # by passing rollout_reward="mes_entropy"
                            # explicitly; nothing about that branch changed.
                            rollout_reward="improvement",
                            use_real_rollout_queries=False,
                            f_hf_real=None,
                            f_lf_real=None,
                            refit_hyperparams_in_rollout=False,
                            recent_ell_seed=None,
                            use_candidate_features=True,
                            use_state_standardization=True,
                            y_star_pool=None,
                            kg_signed=False,
                            kg_topk=1,
                            fantasy_mode='sample',
                            use_roi=False,
                            roi_beta_sqrt=2.0,
                            roi_raw_pool=2000,
                            roi_x_star=None,
                            roi_stats=None,
                            y_star_seed=0):
    """
    One MF rollout, up to rollout_length steps (Bayesian Early Stopping,
    Change 1, may terminate it sooner -- see bes_delta below).

    recent_ell_seed (Change 3b): the caller's REAL recent fidelity history
    (DirectMFRegretOptimization.recent_ell_history), used to seed this
    rollout's recent_hf_frac window at tau=0 instead of the previous
    hardcoded 0.5. Fixes a train/inference mismatch on one of the state's
    few genuinely-varying slots: training only ever showed 0.5 at tau=0,
    while real inference (always timestep=0) passes whatever the actual
    recent real mix is -- typically 0.0 given the observed LF-bias. None
    (default) keeps the old 0.5-at-tau=0 behavior.

    use_candidate_features (Change 1c, default True): append per-candidate
    GP/MES features (build_candidate_features) to the K_cands candidate
    block, so score_head conditions on each candidate's own predicted
    value/uncertainty/information-gain rather than its bare coordinates.
    use_candidate_scoring only.

    use_state_standardization (Change 2b, default True): forwarded to
    _extract_mf_state's `standardize`.

    real_data_hf: (list_of_x, list_of_y) -- actual real HF data
    real_data_lf: (list_of_x, list_of_y) -- actual real LF data
    ko_ensemble_full: M KO models for state extraction's raw-hyperparameter
    block only (_ko_hp_features) -- this is the SAME, unconditioned object
    for every trajectory in an entire batch, by design (it reflects the
    real ensemble's overall calibration, not this rollout's own progress).
    ref_grid: [R, d] raw domain-scale tensor, fixed for the whole real BO
    run -- passed to _extract_mf_state's reference-grid block (Fix for Bug
    A) together with current_ko (THIS rollout's own progressively-
    conditioned model, reassigned every step below), which is what actually
    gives the state per-trajectory, per-step variation. See
    _reference_grid_features's docstring for why ko_ensemble_full alone
    (constant across the whole batch) made every trajectory's tau=0 state
    bit-for-bit identical -- the DT could only ever learn the conditional
    mean of that timestep's targets, independent of anything real inference
    later provides.
    K_rtg: Thompson samples for RTG Gumbel estimation

    use_rtg_grounding (Change B): clips each Thompson-sampled y*_H draw
    from BELOW at the REAL best-observed HF value (max(real_hf_y_list))
    before fitting the Gumbel distribution used for RTG. Motivation
    (Failure 2/3): when the GP believes the global max is mediocre
    (miscalibration), Thompson samples of y* can fall BELOW a value
    already directly observed in reality -- a logical impossibility, since
    the true y* is by definition >= any real observation of f_H. Grounding
    corrects this specific inconsistency without touching the acquisition
    function itself (compute_joint_mf_mes's own Thompson sampling for
    location/fidelity selection is untouched -- this only affects the RTG
    training SIGNAL, matching Failure 3's neg_rtg_frac framing).

    minimum_hf_fraction: if, after joint MES selects (x_tau, ell_tau), the
    fraction of HF steps among this rollout's PRIOR steps (0..tau-1) falls
    below this threshold and MES chose LF, ell_tau is overridden to HF. This
    is a TRAINING-DATA-GENERATION constraint only -- it shapes what the DT
    is trained on, not a runtime policy override; propose_mf/inference can
    still select any fidelity freely. Motivation: Song 2019's MF-MI-Greedy
    guarantees at least one HF query per round via its SF-GP-OPT phase,
    since all-LF exploration cannot directly optimize the HF objective --
    MF-DRO needs the analogous guarantee at the rollout-training level, or
    the DT never sees HF-labeled training examples to learn from.

    bes_delta (Change 1, Bayesian Early Stopping -- DRO.pdf Section D.3,
    MF extension p.18): RELATIVE threshold, as a fraction of the rollout's
    OWN tau=0 cost-normalized MES score (bes_signal_0). After step 2's joint
    MES acquisition, if the best cost-normalized MES score across
    roi_candidates (scores.max(), already computed by compute_joint_mf_mes
    -- no duplicate acquisition call) falls below bes_delta * bes_signal_0,
    the rollout stops immediately: that step's (x_tau, ell_tau) is discarded
    (not sampled, not conditioned on, not appended), and the trajectory
    returned has T < rollout_length. An ABSOLUTE threshold was tried first
    and abandoned: cost-normalized MES is in nats-per-cost, a scale with no
    fixed relationship to any given benchmark/GP state, so a single absolute
    cutoff was either always active or always inactive depending on that
    scale rather than actually discriminating "this step stopped being
    informative." bes_delta=0.05 means "stop once a step provides <5% as
    much information per cost as the rollout's first step did" --
    scale-invariant across benchmarks by construction. Never checked at
    tau=0 (a rollout always takes at least one step, and bes_signal_0 isn't
    set until tau=0 runs). Set bes_delta<=0 to disable (check is skipped
    entirely). NOTE: this makes T variable across trajectories in a batch --
    _generate_rollout_batch's caller, DirectMFRegretOptimization._train_dt,
    pads each trajectory to rollout_length and passes a valid_mask marking
    the padded positions so they're excluded from the loss (Change 2, see
    _train_dt/DecisionTransformer.forward_mf's own docstrings).

    use_candidate_scoring (False by default -- when False, this function's
    behavior is bit-for-bit unchanged from before this flag existed):
    switches step 6's recording from storing the regression target
    (actions_x) to building a K_cands-candidate set (K_cands=20: the
    teacher's own x_tau plus K_cands-1 uniform-random [0,1]^d distractors,
    shuffled so the teacher's position isn't always index 0) and recording
    which shuffled index it landed at. See the return dict below for which
    keys appear in which mode.

    rollout_policy ("mes" default, bit-for-bit unchanged from before this
    flag existed; or "thompson"): selects step 2's (x_tau, ell_tau).
    "thompson" draws one HF posterior sample over roi_candidates (via
    ko_model.hf_posterior's diagonal-marginal mean/variance, NOT the
    hf_proxy joint model) and takes its argmax as x_tau, then picks
    ell_tau via cost-normalized MES evaluated at that single location only
    (cheap vs. the full "mes" policy's MES scan over all 200 candidates).
    Bayesian Early Stopping (2a) is a no-op under "thompson": there is no
    scores tensor over the full candidate pool to threshold against, so BES
    never fires in this mode regardless of bes_delta.

    rollout_reward ("mes_entropy" default, bit-for-bit unchanged; or
    "improvement"): selects how rtg is labeled. "mes_entropy" is the
    original Gumbel-scale relative-entropy RTG (Change 3, see below).
    "improvement" instead accumulates a per-step reward r_tau = max(0,
    y_tau - best_sim_hf_before_this_step) on HF steps (0 on LF steps, since
    only a real HF observation can beat the incumbent), then labels
    rtg_values[tau] = sum(r_i for i>=tau), normalized by rtg_values[0]
    (floored at 1e-8) so RTG stays in [0,1] -- always >=0 by construction,
    unlike the entropy RTG which can go negative. Skips the Gumbel
    Thompson-sampling/fitting (_rollout_gumbel_b) entirely in this mode,
    since it isn't needed for the reward signal.

    use_real_rollout_queries (False default, bit-for-bit unchanged when
    off): DIAGNOSTIC-ONLY oracle ablation ("MF-DRO-Real"), testing whether
    rollouts built from the GP's OWN posterior samples (ko_model.
    sample_fantasy) -- rather than real function evaluations -- is what
    lets the training data become "confidently wrong" and drive the
    incumbent freeze. When True, Step 4 below evaluates f_hf_real/f_lf_real
    (the TRUE benchmark objective, required in this mode) at x_tau instead
    of sampling y_tau from current_ko's posterior; everything else
    (acquisition via compute_joint_mf_mes, GP conditioning via
    make_fantasy_ko, state/RTG/reward computation) is unchanged, isolating
    exactly the sampled-vs-real y-value variable. Never a legitimate
    production setting: it requires unlimited free access to the true
    objective for every rollout step (M*rollouts_per_model*rollout_length
    real evaluations per BO iteration), which defeats the entire premise of
    "simulated" rollouts in real Bayesian optimization, where the true
    objective is the expensive thing being conserved.

    refit_hyperparams_in_rollout (False default, bit-for-bit unchanged when
    off; only meaningful combined with use_real_rollout_queries=True --
    refitting hyperparameters on FANTASY-sampled data would just re-fit the
    GP to its own noisy guess, not a coherent thing to test): the ordinary
    conditioning step (make_fantasy_ko) is a CHEAP, FROZEN-hyperparameter
    posterior update -- same kernel lengthscale/noise/rho as whatever the
    last real _update_ko_ensemble() produced, for every rollout step,
    regardless of use_real_rollout_queries. That means even MF-DRO-Real
    only tests "does an accurate observed VALUE fix the freeze," not "does
    the GP's BELIEF about the kernel shape get a chance to correct itself
    mid-rollout." When True, Step 5 replaces make_fantasy_ko with a full
    KennedyOHaganGP(...).fit(...) MLL re-optimization on the augmented real
    dataset at EVERY rollout step -- ~300 Adam steps (3 alternating rounds
    x 2 GPs x default train_iter=50) per step instead of a near-free
    posterior update, so this is orders of magnitude more expensive and
    only intended for small-scale diagnostic runs, never a real Stage 2
    config.

    Returns dict (T = rollout_length, or fewer if BES stopped it early):
        states:       [T, state_dim]
        actions_x:    [T, d] (use_candidate_scoring=False only)
        candidates:   [T, K_cands, d], chosen_idx: [T] int64
                      (use_candidate_scoring=True only)
        actions_ell:  [T] int64, 0=L 1=H
        rtg:          [T] float64  log(b_tau / b_T), per-step RELATIVE
                      information-remaining (Change 3) -- b_tau is the
                      Gumbel scale of y*_H|D_tau BEFORE conditioning on step
                      tau's own observation (matches SF-DRO's validated
                      _simulate_trajectory_joint in dro.py); b_T is a
                      separate post-loop draw from the fully-conditioned
                      final model. RTG[-1]~=0 (not exactly, since b_values
                      [-1] and b_T are independent estimates of two very
                      similar but not identical posteriors); RTG[0]=total
                      rollout information gain when the rollout was
                      net-informative. Replaces the old absolute
                      differential entropy log(b_tau)+EULER_GAMMA+1 (see the
                      assignment below).
        btg:          [T] float64  future cumulative cost (backward)
        costs:        [T] float64  c(ell_tau) per step
        total_cost:   float  == btg[0]
        lf_fraction:  float
        neg_rtg_frac: float
    """
    # Unpack real data
    real_hf_x_list = real_data_hf[0] if real_data_hf[0] else []
    real_hf_y_list = real_data_hf[1] if real_data_hf[1] else []
    real_lf_x_list = real_data_lf[0] if real_data_lf[0] else []
    real_lf_y_list = real_data_lf[1] if real_data_lf[1] else []

    # Change B: real incumbent for RTG grounding (None if no real HF data
    # exists yet, e.g. before the very first real iteration).
    real_best_hf = max(real_hf_y_list) if real_hf_y_list else None

    # roi_candidates ONCE before the loop. Unfiltered random candidates
    # spanning the full domain, NOT an ROI-filtered pool -- confirmed by
    # direct measurement that an earlier ROI-filtered version produced zero
    # rollout steps within L2=0.2 of the true optimum across 280 sampled
    # steps, so the DT was never shown a training example anywhere near it.
    # Used below for the joint MES acquisition (compute_joint_mf_mes), BES,
    # and RTG (_rollout_gumbel_b) -- NOT for state extraction, which no
    # longer needs any candidate set at all (Fix 1, see _extract_mf_state's
    # docstring); _propose_next_query correspondingly no longer builds an
    # ROI-filtered pool either, since it had no other use for one.
    _N_POOL = 200
    if not use_roi:
        roi_candidates = (
            bounds[0]
            + (bounds[1] - bounds[0])
            * torch.rand(_N_POOL, ko_model.d,
                         device=ko_model.device,
                         dtype=ko_model.dtype)
        )
    else:
        # DRO paper Sec 4.2: X_hat_{m,t} = {x | UCB_m(x) >= max_x' LCB_m(x')},
        # UCB/LCB = mu_m +/- sqrt(beta) sigma_m, computed per ensemble member
        # on ITS OWN posterior, constraining rollout simulations only (never
        # the real query). MF adaptation: the HF posterior, since the HF
        # function is what is being optimized.
        #
        # beta: the paper calls it "an exploration-exploitation trade-off
        # parameter" and does not give a value. The Srinivas et al. 2010
        # formula used elsewhere in this repo gives sqrt(beta) ~ 5.5 at these
        # pool sizes, which admits essentially the whole domain and makes the
        # ablation vacuous. roi_beta_sqrt is therefore an explicit knob
        # (default 2.0, a standard 2-sigma band) and roi_stats records the
        # acceptance rate so it is visible whether the ROI actually bound.
        _raw = (
            bounds[0]
            + (bounds[1] - bounds[0])
            * torch.rand(roi_raw_pool, ko_model.d,
                         device=ko_model.device,
                         dtype=ko_model.dtype)
        )
        with torch.no_grad():
            _mu, _var = ko_model.hf_posterior(_raw)
            _mu = _mu.reshape(-1)
            _sd = _var.reshape(-1).clamp_min(1e-12).sqrt()
        _b = float(roi_beta_sqrt)
        _keep = (_mu + _b * _sd) >= (_mu - _b * _sd).max()
        _surv = _raw[_keep]
        _acc = float(_keep.float().mean())
        if _surv.shape[0] >= _N_POOL:
            _idx = torch.randperm(_surv.shape[0], device=_surv.device)[:_N_POOL]
            roi_candidates = _surv[_idx]
        elif _surv.shape[0] > 0:
            # Sample WITH replacement rather than falling back to the global
            # pool: a tight ROI is the mechanism working, not a failure.
            _idx = torch.randint(_surv.shape[0], (_N_POOL,), device=_surv.device)
            roi_candidates = _surv[_idx]
        else:
            roi_candidates = _raw[:_N_POOL]
        if roi_stats is not None:
            _rec = {'accept_frac': _acc, 'n_surv': int(_surv.shape[0])}
            if roi_x_star is not None:
                # The failure that got ROI deleted before: an ROI that never
                # contains anything near the optimum starves the DT of
                # near-optimal training examples. Measured, not assumed.
                _span = (bounds[1] - bounds[0])
                _dn = ((roi_candidates - roi_x_star.to(roi_candidates)) / _span
                       ).norm(dim=1)
                _rec['min_dist_to_xstar'] = float(_dn.min())
                _rec['frac_within_0.2'] = float((_dn <= 0.2).float().mean())
            roi_stats.append(_rec)

    def _rollout_gumbel_b(ko_for_b):
        """
        Thompson-sample + Gumbel-fit the domain-max y*_H scale b from
        ko_for_b's HF posterior over the fixed roi_candidates above, honoring
        RTG grounding (Change B) and the degenerate-variance floor. Shared by
        the per-step (pre-conditioning) b_tau computation and the post-loop
        b_T computation below -- SF-DRO's _simulate_trajectory_joint
        (dro.py) draws b_T the same way, as a SEPARATE Thompson draw from
        the fully-conditioned model, not a reused per-step value (the last
        per-step b_tau only reflects D_{T-1}, one step short of the fully-
        conditioned D_T the rollout actually ends at).
        """
        hf_proxy = _build_hf_proxy_model(ko_for_b)
        y_star_arr = thompson_sample_y_star(hf_proxy, roi_candidates, K=K_rtg)
        if use_rtg_grounding and real_best_hf is not None:
            y_star_arr = np.maximum(y_star_arr, real_best_hf)
        # Degenerate case introduced by grounding: if the GP's Thompson
        # samples are ALL below real_best_hf (plausible, even expected, in
        # exactly the miscalibrated-GP scenarios grounding targets), every
        # sample clips to the identical constant -- a zero-variance array
        # scipy's gumbel_r.fit() MLE crashes on internally (verified:
        # TypeError inside _average_with_log_weights, not a clean exception
        # scipy itself guards against). Conceptually the b->0
        # (near-zero-entropy) limit the floor below already anticipates --
        # skip the crash-prone fit and go directly to that floor.
        if np.ptp(y_star_arr) < 1e-9:
            b = 1e-12
        else:
            _, b = fit_gumbel_to_samples(y_star_arr)
        return max(b, 1e-12)

    states = []
    actions_x = []               # used when use_candidate_scoring=False
    candidates_list = []         # used when use_candidate_scoring=True
    chosen_idx_list = []         # used when use_candidate_scoring=True
    teacher_scores_list = []     # used when use_candidate_scoring=True (Change 1d)
    has_soft_list = []           # per-step: is teacher_scores a real MES vector? (CRITICAL-2)
    actions_ell = []
    y_values = []                 # per-step sampled/real y_tau, always
    # recorded regardless of rollout_reward mode (rollout_reward=="improvement"
    # already had this via r_values/best_sim_hf, but "mes_entropy" -- the
    # Stage-2 default -- previously discarded y_tau once it was folded into
    # current_ko's fantasy conditioning. Needed for the action-reward
    # correlation diagnostic: corr(rtg_tau, y_tau - incumbent) requires the
    # raw y_tau even when the RTG label itself is Gumbel-entropy-based, not
    # improvement-based.
    b_values = []  # Gumbel scale per step; RTG is labeled AFTER the loop
    # (Change 3, per-step relative RTG) -- see docstring above.
    costs = []

    current_ko = ko_model

    # Local sim data (starts from real data, grows with fantasy obs)
    sim_hf_x = list(real_hf_x_list)
    sim_hf_y = list(real_hf_y_list)
    sim_lf_x = list(real_lf_x_list)
    sim_lf_y = list(real_lf_y_list)

    # rollout_reward=="improvement" only -- see docstring above.
    best_sim_hf = max(sim_hf_y) if sim_hf_y else -1e9
    r_values = []

    # Bayesian Early Stopping (Change 1) reference point: cost-normalized
    # MES at tau=0, set once step 2 first runs below. bes_delta is a
    # FRACTION of this value (relative threshold), not an absolute nats/cost
    # figure -- see bes_delta's docstring above for why an absolute
    # threshold doesn't work (MES's nats-per-cost scale bears no fixed
    # relationship to any particular benchmark's function-value units).
    bes_signal_0 = None

    # rollout_reward=="kg_incumbent": knowledge-gradient-style reward. The
    # ONLY reward mode that credits an LF query. V = max over a FIXED
    # reference set of the HF posterior mean -- i.e. the model's belief about
    # the best attainable HF value, which is exactly the quantity our frozen
    # evaluation (final simple regret on the HF incumbent) is defined on. An
    # LF observation moves mu_H only THROUGH rho in the KO model, so its
    # credit is automatically discounted by how much LF actually informs HF:
    # the "low certainty discount" is derived from the fitted model rather
    # than hand-tuned. Reference set is a fixed Sobol prefix of y_star_pool,
    # so V differences are comparable across steps and trajectories.
    _kg_ref = None
    _V_prev = None
    if rollout_reward == "kg_incumbent":
        if y_star_pool is None:
            raise ValueError("rollout_reward='kg_incumbent' requires y_star_pool "
                              "(the fixed reference set V is maximised over).")
        _kg_ref = y_star_pool[:KG_REF_M]
        _V_prev = _kg_V(current_ko, _kg_ref, kg_topk)

    for tau in range(rollout_length):

        # 1. State BEFORE conditioning. recent_hf_frac uses this rollout's
        # own prior simulated steps, PREPENDED with the caller's real recent
        # fidelity history (Change 3b, recent_ell_seed) so the tau=0 window
        # isn't the hardcoded 0.5 that real inference never presents -- the
        # combined window is chronological (real history first, then this
        # rollout's own simulated steps), then truncated to the last 5, the
        # same rolling-window convention DirectMFRegretOptimization.
        # recent_ell_history uses on the real side. recent_ell_seed=None
        # restores the old 0.5-at-tau=0 behavior.
        # CRITICAL-1: y* for THIS step, drawn once from the fixed pool under
        # this step's own current_ko, then reused by both the ref-grid state
        # block and the per-candidate feature block -- "within one state
        # extraction / one proposal, all features must use the SAME
        # y_star_arr draw".
        _y_star_step = _y_star_for_model(current_ko, y_star_pool, seed=y_star_seed)
        _seed_list = list(recent_ell_seed) if recent_ell_seed else []
        recent_window = (_seed_list + list(actions_ell))[-5:]
        recent_hf_frac = (sum(recent_window) / len(recent_window)
                           if recent_window else 0.5)
        s_tau = _extract_mf_state(
            sim_hf_x, sim_hf_y, sim_lf_x, sim_lf_y,
            ko_ensemble_full,
            n_real_iter, T_real,
            c_L, c_H, device, dtype,
            recent_hf_frac, bounds=bounds,
            # Fix for Bug A: current_ko (THIS rollout's own progressively-
            # conditioned model -- reassigned every step below via
            # make_fantasy_ko) gives the reference-grid block genuine
            # per-trajectory, per-step variation, unlike ko_ensemble_full
            # above (shared, unconditioned, constant across the whole batch).
            ref_grid=ref_grid, ref_model=current_ko,
            # CRITICAL-1: ONE y* draw per step, from the fixed pool, shared
            # by this step's ref-grid block AND its candidate block below.
            y_star_arr=_y_star_step,
            standardize=use_state_standardization,
        )

        # 2. (x_tau, ell_tau) -- "mes" (default): joint MES over all
        # roi_candidates. ITEM 2 (teacher pool): "cost_ei"/"ucb_beta1"/
        # "ucb_beta3"/"thompson" are additional teachers _generate_rollout_
        # batch samples uniformly per rollout from {mes, cost_ei, ucb_beta1,
        # ucb_beta3, thompson} when config.use_teacher_pool is set -- see
        # its own docstring for why. ALL FOUR non-mes teachers below build a
        # real [N_roi, 2] (col0=LF/c_L, col1=HF/c_H) scores matrix in
        # compute_joint_mf_mes's own return contract, so BES, the flat-
        # argmax candidate-index recovery, and teacher_scores extraction
        # downstream (step 6) all work unchanged regardless of which teacher
        # ran -- "teacher_scores comes from whichever teacher generated that
        # step" falls out of this structurally, with no separate code path.
        # "random": uniform-random candidate, independent of GP state --
        # diagnostic teacher for the teacher-determinism investigation, NOT
        # part of the ITEM-2 pool (scores=None, no real per-candidate score
        # exists to build one from).
        #
        # THOMPSON REDESIGN: previously picked x_tau from an HF-only sample
        # (mu_H+sigma_H*eps).argmax(), then chose fidelity via a separate
        # single-point MES ratio, with scores=None (no candidate-set score
        # field existed, so it always fell back to hard cross-entropy for
        # teacher_scores). Now draws an LF+HF Thompson sample over the WHOLE
        # roi_candidates pool and takes a joint cost-normalized flat argmax,
        # exactly like the other teachers -- required so Thompson-taught
        # steps get a real, has_soft=True teacher_scores vector too.
        if rollout_policy == "thompson":
            with torch.no_grad():
                mu_H, var_H = current_ko.hf_posterior(roi_candidates)
                mu_L, var_L = current_ko.lf_posterior(roi_candidates)
            sigma_H = var_H.clamp_min(0).sqrt()
            sigma_L = var_L.clamp_min(0).sqrt()
            f_sample_H = mu_H + sigma_H * torch.randn_like(mu_H)
            f_sample_L = mu_L + sigma_L * torch.randn_like(mu_L)
            scores = torch.stack([f_sample_L / c_L, f_sample_H / c_H], dim=1)
            flat_best = scores.reshape(-1).argmax()
            cand_idx = (flat_best // 2).item()
            ell_tau = (flat_best % 2).item()
            x_tau = roi_candidates[cand_idx]
        elif rollout_policy == "cost_ei":
            with torch.no_grad():
                mu_H, var_H = current_ko.hf_posterior(roi_candidates)
                mu_L, var_L = current_ko.lf_posterior(roi_candidates)
            sigma_H = var_H.clamp_min(1e-12).sqrt()
            sigma_L = var_L.clamp_min(1e-12).sqrt()
            best = max(sim_hf_y) if sim_hf_y else -1e9
            _normal = torch.distributions.Normal(0, 1)
            def _ei(mu, sigma):
                z = (mu - best) / sigma
                return (mu - best) * _normal.cdf(z) + sigma * torch.exp(_normal.log_prob(z))
            ei_H, ei_L = _ei(mu_H, sigma_H), _ei(mu_L, sigma_L)
            scores = torch.stack([ei_L / c_L, ei_H / c_H], dim=1)
            flat_best = scores.reshape(-1).argmax()
            cand_idx = (flat_best // 2).item()
            ell_tau = (flat_best % 2).item()
            x_tau = roi_candidates[cand_idx]
        elif rollout_policy in ("ucb_beta1", "ucb_beta3"):
            beta = 1.0 if rollout_policy == "ucb_beta1" else 3.0
            with torch.no_grad():
                mu_H, var_H = current_ko.hf_posterior(roi_candidates)
                mu_L, var_L = current_ko.lf_posterior(roi_candidates)
            sigma_H = var_H.clamp_min(0).sqrt()
            sigma_L = var_L.clamp_min(0).sqrt()
            ucb_H, ucb_L = mu_H + beta * sigma_H, mu_L + beta * sigma_L
            scores = torch.stack([ucb_L / c_L, ucb_H / c_H], dim=1)
            flat_best = scores.reshape(-1).argmax()
            cand_idx = (flat_best // 2).item()
            ell_tau = (flat_best % 2).item()
            x_tau = roi_candidates[cand_idx]
        elif rollout_policy == "random":
            N = roi_candidates.shape[0]
            cand_idx = torch.randint(0, N, (1,)).item()
            x_tau = roi_candidates[cand_idx]
            ell_tau = 1 if torch.rand(1).item() < 0.25 else 0
            scores = None
        else:
            x_tau, ell_tau, scores = compute_joint_mf_mes(
                current_ko, roi_candidates, c_H, c_L
            )
        if tau == 0 and scores is not None:
            bes_signal_0 = scores.max().item()

        # 2a. Bayesian Early Stopping (Change 1) -- reuses `scores` from
        # step 2 above, no duplicate MES computation. Never fires at tau=0
        # (a rollout always takes at least one step). Fires BEFORE the
        # minimum_hf_fraction override (2b): a step BES has decided isn't
        # worth taking shouldn't be resurrected by the HF-diversity floor.
        # RELATIVE threshold: fires once the cost-normalized MES score has
        # dropped below bes_delta * bes_signal_0 (a fraction of tau=0's own
        # score), not an absolute nats-per-cost figure -- MES's scale
        # varies by orders of magnitude across benchmarks/GP states, so an
        # absolute threshold (bes_delta=1e-4) was either always or never
        # active depending on that scale, never actually discriminating.
        # No-op under rollout_policy=="thompson" (scores is None there).
        if (tau > 0 and bes_delta > 0 and bes_signal_0 is not None
                and scores is not None):
            bes_signal = scores.max().item()
            if bes_signal < bes_delta * bes_signal_0:
                break

        # 2b. Minimum-HF-fraction override (training-data-generation only --
        # see minimum_hf_fraction's docstring above). Must happen before
        # Steps 3/4/6 below, which all consume ell_tau (fantasy sampling,
        # KO conditioning, and cost/action bookkeeping must all reflect
        # whichever fidelity is ACTUALLY simulated this step).
        # REVERTED (tau=0 override): the measured tau=0 HF label rate was
        # 0.371, not ~0, so it was not the cause of p_pred collapsing to
        # ~0.0002 -- and seeding the floor from recent_ell_seed drove that
        # rate to 1.0000 in the all-LF regime, i.e. a CONSTANT label at the
        # one position inference actually uses. Original guard restored.
        # recent_ell_seed is still used for recent_hf_frac above, which is
        # correct (a state feature, not a label).
        hf_steps_so_far = sum(1 for e in actions_ell if e == 1)
        if (tau > 0 and
                hf_steps_so_far / tau < minimum_hf_fraction and
                ell_tau == 0):
            ell_tau = 1  # forced HF to ensure training diversity

        # 3. b_tau: Thompson-sample + Gumbel-fit the domain-max y*_H scale,
        # BEFORE conditioning on this step's own (x_tau, y_tau). Reflects
        # D_tau (steps 0..tau-1 only) -- the information state the policy
        # actually had when it chose x_tau -- matching SF-DRO's validated
        # _simulate_trajectory_joint (dro.py, rtg_schema=="joint"/
        # "entropy_joint"), which this was meant to mirror but originally
        # didn't: the first version of this computed b_tau AFTER
        # conditioning, folding that single step's own fresh (possibly
        # noisy) fantasy draw directly into the quantity meant to measure
        # only PRIOR accumulated information. That made every b_tau jumpy
        # step-to-step in a way not tied to genuine information accumulation,
        # and was the direct cause of neg_rtg_frac_batch staying far above
        # the "rare" rate the log-ratio construction is supposed to produce
        # (observed 0.6-0.78 instead of well under 0.5).
        # No-op (skipped) under rollout_reward=="improvement" -- that mode's
        # RTG signal doesn't use the Gumbel scale, see docstring above.
        b_tau = _rollout_gumbel_b(current_ko) if rollout_reward == "mes_entropy" else None

        # 4. Sample fantasy observation -- or, under use_real_rollout_queries
        # (MF-DRO-Real diagnostic), evaluate the TRUE objective at x_tau
        # instead (see docstring above).
        if use_real_rollout_queries:
            f_real = f_hf_real if ell_tau == 1 else f_lf_real
            y_tau = f_real(x_tau.unsqueeze(0)).reshape(-1)[0].item()
        else:
            y_tau = current_ko.sample_fantasy(x_tau, 'LH'[ell_tau],
                                               mode=fantasy_mode)

        # 4a. rollout_reward=="improvement" only: per-step improvement
        # reward, using the incumbent BEFORE this step's own observation.
        if rollout_reward == "improvement":   # kg_incumbent scores at 5b
            if ell_tau == 1:
                y_tau_f = float(y_tau)
                r_tau = max(0.0, y_tau_f - best_sim_hf)
                best_sim_hf = max(best_sim_hf, y_tau_f)
            else:
                r_tau = 0.0
            r_values.append(r_tau)

        # 5. Condition (returns new object, does not mutate)
        if use_real_rollout_queries and refit_hyperparams_in_rollout:
            # Full MLL refit on the augmented REAL dataset, instead of a
            # frozen-hyperparameter posterior update -- see docstring above.
            aug_hf_x = list(sim_hf_x) + ([x_tau.detach()] if ell_tau == 1 else [])
            aug_hf_y = list(sim_hf_y) + ([float(y_tau)] if ell_tau == 1 else [])
            aug_lf_x = list(sim_lf_x) + ([x_tau.detach()] if ell_tau == 0 else [])
            aug_lf_y = list(sim_lf_y) + ([float(y_tau)] if ell_tau == 0 else [])
            new_ko = KennedyOHaganGP(
                d=current_ko.d, rho_init=current_ko.rho_init, lr=current_ko.lr,
                train_iter=current_ko.train_iter, noise_lb=current_ko.noise_lb,
                dkl_threshold=current_ko.dkl_threshold,
                dkl_train_iter=current_ko.dkl_train_iter,
                d_feature=current_ko.d_feature, rho_fixed=current_ko.rho_fixed,
                initial_lengthscale=current_ko.initial_lengthscale,
            )
            new_ko.fit(
                torch.stack(aug_lf_x), torch.tensor(aug_lf_y, dtype=dtype),
                torch.stack(aug_hf_x), torch.tensor(aug_hf_y, dtype=dtype),
                bounds,
            )
            current_ko = new_ko
        else:
            current_ko = current_ko.make_fantasy_ko(
                x_tau.unsqueeze(0),
                torch.tensor([y_tau], device=device, dtype=dtype),
                'LH'[ell_tau]
            )

        # 5b. kg_incumbent reward -- must be computed AFTER conditioning,
        # since it is the CHANGE in the believed best HF value caused by this
        # step's observation. Unlike "improvement" (which hard-codes 0.0 for
        # every LF step), both fidelities are scored by the same quantity and
        # LF's smaller effect on mu_H is the discount, not a special case.
        if rollout_reward == "kg_incumbent":
            _V_new = _kg_V(current_ko, _kg_ref, kg_topk)
            _dV = _V_new - _V_prev
            # kg_signed=True keeps DOWNWARD revisions. Learning that a region is
            # worse than believed is real progress toward the optimum; clipping
            # it to zero (H12) threw that signal away.
            r_values.append(_dV if kg_signed else max(0.0, _dV))
            _V_prev = _V_new

        # 6. Record
        states.append(s_tau)
        y_values.append(float(y_tau))
        # propose_mf's location head is architecturally normalized to
        # [0,1]^d (its own output is x_pred.clamp(0.0, 1.0)) -- actions_x
        # must match that convention for L_loc=MSE(x_pred, actions_x) to
        # be meaningfully scaled. x_tau itself stays RAW domain-scale for
        # the KO-GP-facing calls above/below (sample_fantasy,
        # make_fantasy_ko, sim_hf_x/sim_lf_x) -- only this stored copy is
        # normalized. No-op for [0,1]^d domains. See REVISION_LOG.md.
        if use_candidate_scoring:
            # K_cands-candidate set: the teacher's own choice (x_tau,
            # normalized the same way actions_x always was) plus K_cands-1
            # distractors, shuffled so the teacher's position isn't always
            # index 0 (score_head must learn to find it wherever it lands,
            # not just always pick slot 0).
            #
            # FIX (Bug E, label leak): distractors now sampled from the SAME
            # roi_candidates pool the MES argmax (x_tau) was actually chosen
            # from, instead of independent torch.rand draws. Previously the
            # positive (an acquisition-selected argmax) and negatives (raw
            # uniform noise) came from visibly different distributions --
            # verified directly against compute_joint_mf_mes's own return
            # contract (scores is [N_roi,2], indexed identically to
            # roi_candidates; cand_idx = argmax_flat//2 recovers exactly
            # which roi_candidates row won) -- letting a score head win by
            # detecting "looks acquisition-selected" rather than "is good."
            # roi_candidates is itself uniform-random over the domain
            # (see its construction above), matching propose_mf's own
            # inference-time candidate distribution (200 fresh uniform
            # draws) -- so this isn't introducing a new train/inference
            # mismatch, only removing the distributional shortcut.
            # Only applies when scores is available (rollout_policy=="mes",
            # the only mode any experiment has combined with
            # use_candidate_scoring=True this session); "thompson"/"random"
            # modes (scores=None) keep the original torch.rand distractors.
            x_tau_norm = ((x_tau - bounds[0]) / (bounds[1] - bounds[0])).clamp(0, 1)
            if scores is not None:
                cand_idx = (scores.reshape(-1).argmax() // 2).item()
                # VERIFICATION (spec): the row compute_joint_mf_mes's own
                # flat-argmax//2 identifies must be exactly the x_tau it
                # returned -- guards the index arithmetic this whole
                # candidate/teacher-score block depends on.
                assert torch.equal(roi_candidates[cand_idx], x_tau), (
                    "compute_joint_mf_mes index arithmetic mismatch: "
                    f"roi_candidates[{cand_idx}] != returned x_tau"
                )
                other_idx = torch.randperm(roi_candidates.shape[0])
                other_idx = other_idx[other_idx != cand_idx][:K_cands - 1]
                # Selected rows INTO roi_candidates (winner first, then
                # K_cands-1 distractors), permuted together below so the
                # teacher's own choice isn't always at slot 0.
                sel_idx = torch.cat([torch.tensor([cand_idx]), other_idx])
                perm = torch.randperm(sel_idx.numel())
                sel_idx = sel_idx[perm]
                cidx = (perm == 0).nonzero(as_tuple=True)[0].item()
                all_cands_raw = roi_candidates[sel_idx]     # [K,d] raw scale
                # Change 1d: soft distillation target. teacher_scores is the
                # cost-normalized MES value of each selected candidate at its
                # OWN best fidelity (scores is [N_roi,2], col0=LF/c_L,
                # col1=HF/c_H -- max over dim=1 is exactly the quantity
                # compute_joint_mf_mes's flat argmax maximizes). The DT then
                # learns the teacher's whole preference DISTRIBUTION over the
                # candidate set, not just its argmax -- a multimodal
                # acquisition surface can be represented, which a hard
                # one-hot label discards.
                teacher_scores = scores.max(dim=1).values[sel_idx]
                has_soft = True
            else:
                # rollout_policy "thompson"/"random": no scores tensor over a
                # candidate pool exists, so distractors stay uniform and the
                # soft target degenerates to the hard one-hot (all-zeros
                # teacher_scores -> uniform softmax is wrong, so mark the
                # winner explicitly instead).
                distractors_raw = (bounds[0] + (bounds[1] - bounds[0])
                                    * torch.rand(K_cands - 1, ko_model.d,
                                                  device=x_tau.device, dtype=x_tau.dtype))
                all_cands_raw = torch.cat([x_tau.unsqueeze(0), distractors_raw], dim=0)
                perm = torch.randperm(K_cands)
                all_cands_raw = all_cands_raw[perm]
                cidx = (perm == 0).nonzero(as_tuple=True)[0].item()
                # CRITICAL-2: no -1e9 sentinel. Standardizing a sentinel
                # one-hot yields a SOFT distribution (silently wrong), so
                # this step is marked with an explicit boolean instead and
                # forward_mf falls back to hard cross-entropy on chosen_idx
                # for it, bypassing standardization entirely.
                teacher_scores = torch.zeros(K_cands, dtype=x_tau.dtype)
                has_soft = False

            if use_candidate_features:
                # Change 1c: [x_norm(d), mu_H, sigma_H, mu_L, sigma_L,
                # MES_H/c_H, MES_L/c_L, ||x-x_incumbent||] -> [K, d+7], from
                # current_ko (this rollout's own model) -- the SAME
                # build_candidate_features the inference path calls.
                if sim_hf_y:
                    _bi = int(np.argmax(sim_hf_y))
                    _best_pos_norm = ((sim_hf_x[_bi] - bounds[0])
                                      / (bounds[1] - bounds[0])).clamp(0, 1)
                else:
                    _best_pos_norm = torch.zeros(ko_model.d, dtype=x_tau.dtype)
                _ym = float(np.mean(sim_hf_y)) if (use_state_standardization and sim_hf_y) else 0.0
                _ys = max(float(np.std(sim_hf_y)), 1e-6) if (use_state_standardization and sim_hf_y) else 1.0
                all_cands = build_candidate_features(
                    current_ko, all_cands_raw, bounds, c_H, c_L, _best_pos_norm,
                    y_star_arr=_y_star_step, y_mean=_ym, y_std=_ys,
                )
            else:
                all_cands = ((all_cands_raw - bounds[0])
                             / (bounds[1] - bounds[0])).clamp(0, 1)

            candidates_list.append(all_cands)
            chosen_idx_list.append(cidx)
            teacher_scores_list.append(teacher_scores)
            has_soft_list.append(has_soft)
        else:
            actions_x.append(((x_tau - bounds[0]) / (bounds[1] - bounds[0])).detach())
        actions_ell.append(ell_tau)
        if rollout_reward == "mes_entropy":
            b_values.append(b_tau)
        step_cost = c_L if ell_tau == 0 else c_H
        costs.append(step_cost)

        # Update sim data for next step's state
        if ell_tau == 0:
            sim_lf_x.append(x_tau.detach())
            sim_lf_y.append(float(y_tau))
        else:
            sim_hf_x.append(x_tau.detach())
            sim_hf_y.append(float(y_tau))

    # BTG: backward cumulative sum of costs
    cost_t = torch.tensor(costs, dtype=dtype)
    btg = cost_t.flip(0).cumsum(0).flip(0)

    # Change 3: per-step relative RTG, labeled backward AFTER the rollout
    # completes -- RTG_tau = log(b_tau) - log(b_T) = log(b_tau / b_T), the
    # information REMAINING to be gained between step tau and the rollout's
    # end, relative to however informative the end state turned out to be.
    # Replaces the old absolute Gumbel differential entropy
    # (log(b_tau) + EULER_GAMMA + 1): under a plain entropy DIFFERENCE the
    # EULER_GAMMA+1 term cancels exactly, which is why it no longer appears
    # here.
    #
    # b_T is a SEPARATE Thompson draw from the fully-conditioned final
    # current_ko (matching SF-DRO's _simulate_trajectory_joint), NOT
    # b_values[-1] -- the last per-step b_tau reflects D_{T-1} (computed
    # BEFORE that step's own conditioning, per step 3 above), one step
    # short of the fully-conditioned D_T the rollout actually ends at.
    # Consequently RTG[-1] is only approximately 0 (b_values[-1] and b_T are
    # two independent K=100 Thompson-sample estimates of two very similar,
    # but not identical, posteriors), not exactly 0 -- unlike the earlier,
    # incorrect version of this function, which reused b_values[-1] as b_T
    # and so forced RTG[-1]==0 by construction.
    #
    # RTG[0] >= 0 whenever the rollout net-gained information (b shrank from
    # tau=0 to the end), but is NOT guaranteed >= 0 at every tau (b_tau is
    # not necessarily monotonic step-to-step -- see the BES ratio
    # diagnostic's finding that cost-normalized MES, a related uncertainty
    # signal, is noisy and non-monotonic within a short rollout).
    # rollout_reward=="improvement": forward-sum of per-step improvement
    # rewards (always >=0 by construction), normalized into [0,1] by its
    # own tau=0 value -- see rollout_reward's docstring above. Replaces the
    # Gumbel-entropy RTG below entirely; b_T/b_values aren't needed.
    zero_reward_frac = None
    if rollout_reward in ("improvement", "kg_incumbent"):
        if r_values:
            r_t = torch.tensor(r_values, dtype=dtype)
            # RTG-CAP FIX: previously divided by this trajectory's OWN
            # rtg_raw[0] here, which forces rtg[0]==1.0 for EVERY trajectory
            # with any positive total improvement (confirmed directly: 200
            # trajectories, exactly 2 distinct rtg[0] values in the whole
            # batch -- 0.0 or 1.0 -- a binary "improved at all" signal with
            # the actual MAGNITUDE of improvement discarded, which is
            # exactly what the item-4 gate needs to correlate against true
            # quality). Returns UNNORMALIZED rtg_raw now; normalization by a
            # single BATCH-LEVEL scale happens in _generate_rollout_batch,
            # once all trajectories exist, so relative ordering ACROSS
            # trajectories at tau=0 is preserved instead of erased.
            rtg_t = r_t.flip(0).cumsum(0).flip(0)
            zero_reward_frac = float((r_t == 0).float().mean())
        else:
            rtg_t = torch.zeros(0, dtype=dtype)
    else:
        b_T = _rollout_gumbel_b(current_ko)
        log_b_T = math.log(b_T)
        rtg_values = [math.log(b) - log_b_T for b in b_values]
        rtg_t = torch.tensor(rtg_values, dtype=dtype)

    traj = {
        'states': torch.stack(states),
        'actions_ell': torch.tensor(actions_ell, dtype=torch.long),
        'rtg': rtg_t,
        'btg': btg,
        'costs': cost_t,
        'total_cost': cost_t.sum().item(),
        'lf_fraction': float((torch.tensor(actions_ell) == 0)
                              .float().mean()),
        'neg_rtg_frac': float((rtg_t < 0).float().mean()),
        'y_values': torch.tensor(y_values, dtype=dtype),
    }
    if zero_reward_frac is not None:
        traj['zero_reward_frac'] = zero_reward_frac
    if use_candidate_scoring:
        traj['candidates'] = torch.stack(candidates_list)
        # chosen_idx kept for diagnostics (and for the "thompson"/"random"
        # rollout_policy fallback), per spec -- teacher_scores is what the
        # soft-KL loss actually consumes (Change 1d).
        traj['chosen_idx'] = torch.tensor(chosen_idx_list, dtype=torch.long)
        traj['teacher_scores'] = torch.stack(teacher_scores_list)
        traj['has_soft'] = torch.tensor(has_soft_list, dtype=torch.bool)
    else:
        traj['actions_x'] = torch.stack(actions_x)
    return traj


# ════════════════════════════════════
# PART E: RTG / BTG Target Schemas
# ════════════════════════════════════

class MFTargetSchemas:
    """
    Manages RTG and BTG targets across BO iterations.
    Initialized once per MF-DRO run.

    RTG (floored dynamic, maximize): target = max(batch_max, alpha_rtg *
    running_max_rtg) -- identical formula to dro.py's _compute_rtg_target
    rtg_schema=="floored" branch (`rtg_target = max(batch_max_rtg,
    self.alpha_floor * self.running_max_rtg)`), same running_max update
    ordering (running max updated from this batch BEFORE computing the
    floor). This is a FLOOR: it prevents the target from regressing too far
    below the best-ever achieved value even when this batch's own best is
    mediocre, by never letting it drop under alpha_rtg (<1) times the
    all-time best.

    BTG (batch mean, dynamic, updated every iteration -- FIX for Bug B, see
    update_and_get_btg_target's own docstring): previously a "ceilinged
    dynamic" schema symmetric to RTG's floor (target = min(batch_min,
    (1/alpha_btg)*running_min_btg)), which collapsed to batch_min on
    essentially every iteration and, combined with run()'s own former
    freeze-after-iteration-0 behavior, permanently conditioned every real
    query on the CHEAPEST rollout ever observed -- verified directly against
    a saved run: btg_target pinned at 22.0, exactly 2*c_H+6*c_L, the
    theoretical floor under minimum_hf_fraction=0.25. That is a first-
    principles explanation for the extreme LF-bias (lf_fraction ~0.98-0.99)
    seen in every MF-DRO run this session. Now a genuinely representative,
    dynamic target (batch mean, recomputed and re-applied every iteration,
    same treatment RTG's target already got).
    """

    def __init__(self, alpha_rtg=0.5, alpha_btg=0.5,
                 c_L=1.0, c_H=8.0):
        self.alpha_rtg = alpha_rtg
        self.alpha_btg = alpha_btg
        self.c_L = c_L
        self.c_H = c_H
        self.running_max_rtg = 0.0
        self.running_min_btg = float('inf')

    def update_and_get_rtg_target(self, rollout_batch):
        """
        RTG[0] = total rollout information gain, log(b_0 / b_T) (Change 3,
        per-step relative RTG -- see simulate_mf_trajectory's docstring).
        High RTG[0] = informative rollout. Extraction/floor semantics below
        are unchanged from the prior absolute-entropy RTG.
        Floored dynamic schema (same as SF-DRO):
            target = max(batch_max_rtg, alpha * running_max_rtg)

        rollout_batch: list of trajectory dicts from
                       simulate_mf_trajectory
        Returns: rtg_target (scalar float)
        """
        rtg0_list = [traj['rtg'][0].item() for traj in rollout_batch]
        batch_max = max(rtg0_list)
        self.running_max_rtg = max(self.running_max_rtg, batch_max)
        return max(batch_max,
                    self.alpha_rtg * self.running_max_rtg)

    def update_and_get_btg_target(self, rollout_batch):
        """
        BTG[0] = total rollout cost (= sum of all step costs).

        FIX (Bug B): the previous schema, target = min(batch_min_btg,
        (1/alpha)*running_min_btg), always returns ~batch_min_btg in
        practice -- running_min_btg <= batch_min_btg by construction (it's
        updated to the min of itself and batch_min_btg on the very line
        before this returns), so the ceiling term (1/alpha)*running_min_btg
        (2x running_min_btg at alpha=0.5) only binds if a batch's min
        happens to exceed 2x the all-time min, which essentially never
        happens once minimum_hf_fraction=0.25 pins the achievable minimum
        rollout cost at a fixed floor (2*c_H + (rollout_length-2)*c_L) every
        iteration. Confirmed directly: a saved run's btg_target sequence was
        22.0 (exactly that floor, for Hartmann_6D/rollout_length=8) on
        essentially every iteration. Combined with run()'s former freeze-
        after-iteration-0 behavior, every real query for the entire run was
        permanently told "reproduce the cheapest rollout you've ever seen" --
        a first-principles account of the extreme LF-bias (lf_fraction
        ~0.98-0.99) observed in every MF-DRO run this session.

        Now the batch MEAN of btg[0], with no artificial ceiling -- a
        representative "typical" cost from the CURRENT policy's own rollout
        distribution, dynamic every iteration (see run(), which no longer
        freezes btg_target_base after iteration 0 either).
        running_min_btg is still tracked (in case other code inspects it)
        but no longer used in this formula.

        rollout_batch: list of trajectory dicts
        Returns: btg_target (scalar float)
        """
        btg0_list = [traj['btg'][0].item() for traj in rollout_batch]
        self.running_min_btg = min(self.running_min_btg, min(btg0_list))
        return sum(btg0_list) / len(btg0_list)

    def get_inference_btg(self, btg_target_base,
                           cumulative_real_cost):
        """
        NOT CALLED from DirectMFRegretOptimization._propose_next_query
        (kept here for reference only) -- decrementing btg_target_base by
        the FULL cumulative cost of the entire real BO run (including
        initial sampling) clamped this to c_L on essentially every real
        iteration in practice, since btg_target_base is derived from a
        short simulated rollout while cumulative_real_cost can exceed it
        before the first real iteration even runs. The DT then learned
        BTG=c_L as a constant signal meaning "always pick the cheapest
        action," rather than anything meaningfully tied to remaining
        budget. _propose_next_query now passes btg_target_base directly,
        treating every real query as the start of a fresh decision horizon
        (the same role btg[0] plays at tau=0 in a simulated rollout).

        At inference time, BTG decreases as real costs accumulate.
        btg_inference = btg_target_base - cumulative_real_cost
        Clamped to minimum c_L (never tell DT that budget=0).

        Returns: btg_inference (scalar float)
        """
        return max(btg_target_base - cumulative_real_cost,
                    self.c_L)


# ════════════════════════════════════
# PART G: MF-DRO Main Class
# ════════════════════════════════════

class DirectMFRegretOptimization:
    """
    Multi-fidelity Direct Regret Optimization: M independent
    KennedyOHaganGP ensemble members, simulated MF rollouts, a
    forward_mf/propose_mf-capable DecisionTransformer, and RTG/BTG target
    schemas, wired into a standard BO loop.

    Two corrections versus a literal reading of the originally given spec,
    both verified against the actual classes as built in earlier prompts
    (not stylistic changes):

    1. MFTargetSchemas' real methods are update_and_get_rtg_target /
       update_and_get_btg_target / get_inference_btg (not update_rtg /
       update_btg / inference_btg -- those names don't exist on the class).
       Also, _propose_next_query needs the CURRENT iteration's
       schema-computed RTG target (the floored-dynamic max(batch_max,
       alpha*running_max) value), not self.schemas.running_max_rtg directly
       (the raw, un-floored tracker) -- using the raw tracker would bypass
       the whole floored-dynamic formula. Stored as self._last_rtg_target
       right after computing it in run().

    2. DecisionTransformer.__init__ is (self, config, input_dim, action_dim,
       use_mf=False) -- input_dim/action_dim are separate positional
       arguments, never read from config, and config's transformer-encoder
       fields are num_layers/num_heads (not n_layer/n_head). Also explicitly
       cast to .float() after construction: importing anything under
       src.policy (this module lives at src/policy/mf_dro.py) transitively
       imports src/policy/baselines.py, which calls
       torch.set_default_dtype(torch.float64) at module level -- an
       existing, unrelated side effect in this codebase that would
       otherwise silently make the DT's own parameters float64 regardless
       of the float32 data pipeline below, causing a dtype mismatch the
       first time forward_mf/propose_mf runs (confirmed in Prompt 5).
    """

    def __init__(self, config, f_hf, f_lf, bounds):
        self.f_hf = f_hf
        self.f_lf = f_lf
        self.bounds = bounds
        self.c_L = config.c_L
        self.c_H = config.c_H
        self.M = config.M
        self.d = bounds.shape[1]
        self.config = config
        # Fixed reference grid for the state's reference-grid block (Fix for
        # Bug A -- see _reference_grid_features/_extract_mf_state's
        # docstrings). Generated ONCE per run and shared identically by
        # every training rollout (simulate_mf_trajectory's ref_grid param,
        # via _generate_rollout_batch) and by real inference
        # (_propose_next_query) -- the same fixed-points-shared-by-train-
        # and-inference design that motivated removing the OLD, candidate-
        # dependent 2*M block in the first place (that one used DIFFERENT
        # candidate distributions on each side; this one deliberately does
        # not). Sobol for low-discrepancy spatial coverage; seeded by
        # config.seed so the grid itself is reproducible run-to-run. Raw
        # domain-scale (ko.hf_posterior/lf_posterior expect raw-domain X,
        # not [0,1]^d-normalized, exactly like every other KO-model
        # consumer in this class).
        _sobol = torch.quasirandom.SobolEngine(dimension=self.d, scramble=True, seed=config.seed)
        _ref_unit = _sobol.draw(STATE_REF_GRID_R).to(dtype=torch.float64)
        self.state_ref_grid = bounds[0] + (bounds[1] - bounds[0]) * _ref_unit
        # CRITICAL-1 FIX: separate, LARGER fixed pool from which y* is drawn
        # once per (model, iteration) -- see _y_star_for_model. Deliberately
        # NOT state_ref_grid: 10 points in 6D badly underestimate max_x f(x),
        # and y*'s location/scale depend on the pool size, which is the whole
        # bug. Same seeded-Sobol construction pattern, sized Y_STAR_POOL_P.
        _sobol_ys = torch.quasirandom.SobolEngine(
            dimension=self.d, scramble=True, seed=config.seed + 991)
        _ys_unit = _sobol_ys.draw(Y_STAR_POOL_P).to(dtype=torch.float64)
        self.y_star_pool = bounds[0] + (bounds[1] - bounds[0]) * _ys_unit
        # Candidate scoring vs. regression location head -- see
        # DecisionTransformer.forward_mf/propose_mf's own docstrings.
        # False (default): original regression pipeline, bit-for-bit
        # unchanged.
        # DEFAULT: regression head (False). Reverted from Change 1a, which
        # had made candidate scoring the default.
        #
        # Change 1a's argument, kept because it is still sound and still
        # untested: MSE regression collapses to the conditional mean of the
        # teacher's argmax distribution whenever states repeat within a
        # batch -- which they structurally do at tau=0 (see
        # _reference_grid_features's docstring on the M-groups-of-
        # rollouts_per_model duplication) -- and the mean of a multimodal
        # acquisition surface can land in a region no mode occupies.
        # Softmax-over-candidates has no such failure mode: duplicated
        # inputs with different labels yield a DISTRIBUTION whose argmax is
        # always an actual candidate. The [STATE-DIAG] line printed each
        # iteration confirms the precondition: 200 rollouts routinely give
        # ~10 unique tau=0 states.
        #
        # Why the default flipped anyway: h45 measured both heads at matched
        # settings (Hartmann 6D, cost budget 200, initial_hf=36/initial_lf=60,
        # shared seeds) and the regression head won on 5/6, mean 0.3711 vs
        # 0.4523 for scoring and 0.4724 for the teacher alone. So the
        # collapse, where it happens, is not costing more than the pool +
        # linear-score restriction costs. Its signature may instead be h45's
        # variance: one seed at 1.1818 against a 0.2308 median, i.e. rare and
        # catastrophic rather than uniform. Set use_candidate_scoring=True to
        # restore the old head.
        self.use_candidate_scoring = getattr(config, 'use_candidate_scoring', False)
        # DRO paper Sec 4.2 ROI filtering of ROLLOUT candidates. Default OFF:
        # this repo removed ROI after an earlier version was measured giving
        # zero rollout steps within L2=0.2 of the optimum. roi_stats makes
        # that failure visible instead of silent.
        self.use_roi = getattr(config, 'use_roi', False)
        self.roi_stats = [] if self.use_roi else None
        _kx = getattr(config, 'known_optimal_x', None)
        self._roi_x_star = (torch.as_tensor(_kx, dtype=bounds.dtype)
                            if _kx is not None else None)
        # Change 1c/1d and Change 2 ablation flags (all default ON, per spec).
        self.use_candidate_features = getattr(config, 'use_candidate_features', True)
        # CRITICAL-2: `soft_targets` (default True) switches the whole
        # candidate-scoring branch between the standardized-KL soft target
        # and plain F.cross_entropy on chosen_idx, so the two are
        # independently ablatable. use_soft_score_target is accepted as an
        # alias for backward compatibility with configs already using it.
        self.use_soft_score_target = getattr(
            config, 'soft_targets', getattr(config, 'use_soft_score_target', True))
        self.use_state_standardization = getattr(config, 'use_state_standardization', True)
        # ITEM 2: teacher pool -- see TEACHER_POOL/_generate_rollout_batch.
        # Default OFF (single fixed self.rollout_policy) for backward
        # compatibility; the gate-check script below turns it on explicitly.
        self.use_teacher_pool = getattr(config, 'use_teacher_pool', False)
        # Inference-time candidate pool size (Change 1b). Drawn uniformly
        # over the domain, matching simulate_mf_trajectory's roi_candidates
        # distribution (also uniform over bounds).
        self.n_infer_candidates = getattr(config, 'n_infer_candidates', 200)
        # THRESHOLD-BUG FIX flag (item 2): default True (sample ell_t ~
        # Bernoulli(p_val) at real inference instead of thresholding at
        # p_val>0.5). See propose_mf's own docstring for why the threshold
        # is structurally wrong when the true tau=0 HF label rate (0.371) is
        # below 0.5.
        self.fidelity_sampling = getattr(config, 'fidelity_sampling', True)
        self.rollout_policy = getattr(config, 'rollout_policy', 'mes')
        # ITEM 1: default switched to regret-based RTG ("improvement");
        # mes_entropy still available via explicit config.rollout_reward.
        self.rollout_reward = getattr(config, 'rollout_reward', 'improvement')
        # MF-DRO-Real diagnostic (see simulate_mf_trajectory's docstring) --
        # False (default): original simulated-rollout pipeline, bit-for-bit
        # unchanged.
        self.use_real_rollout_queries = getattr(config, 'use_real_rollout_queries', False)
        self.refit_hyperparams_in_rollout = getattr(config, 'refit_hyperparams_in_rollout', False)

        # dkl_threshold: None (default) means "use KennedyOHaganGP's own
        # class default" (30) -- only overridden here when the config
        # explicitly sets one (e.g. dkl_threshold=9999 to keep DKL from
        # ever activating, for ablation control).
        _dkl_threshold = getattr(config, 'dkl_threshold', None)
        _ko_kwargs = {} if _dkl_threshold is None else {'dkl_threshold': _dkl_threshold}

        # Ensemble diversity, analogous to SF-DRO's dro.py _initialize_models
        # (np.linspace grid of initial_lengthscale across the ensemble,
        # re-anchored every refit -- see KennedyOHaganGP.__init__'s
        # initial_lengthscale docstring). Without this, every member fits
        # the SAME data from the SAME fixed initialization and converges to
        # ~the same MLE solution -- M "different" rollout generators that
        # are actually M copies of one model, defeating the purpose of an
        # ensemble (dense, diverse rollout training data for the DT).
        # rho_init ~ Uniform(0.3, 0.95) per member (distinct seed offset
        # +2000, clear of +0/+1 used by initial-point sampling and +1000
        # used by sequential-max-variance init).
        torch.manual_seed(config.seed + 2000)
        _rho_inits = (0.3 + 0.65 * torch.rand(config.M)).tolist()
        # Lengthscale diversity grid spans the LogNormalPrior's own ~95%
        # mass interval (+/- 2 sigma in log-space) -- derived from ko_gp.py's
        # LENGTHSCALE_PRIOR_LOC/SCALE, not a duplicated formula here (a
        # duplicate previously went stale the moment the old hard-bound
        # formula changed, generating grid values the constraint rejected
        # outright -- see ko_gp.py's module docstring). d-independent now,
        # matching the prior itself (unlike the old d-scaled hard bounds).
        _ls_low = math.exp(LENGTHSCALE_PRIOR_LOC - 2 * LENGTHSCALE_PRIOR_SCALE)
        _ls_high = math.exp(LENGTHSCALE_PRIOR_LOC + 2 * LENGTHSCALE_PRIOR_SCALE)
        _ls_grid = torch.linspace(_ls_low, _ls_high, config.M).tolist()
        # H49: ko_ensemble[0] is NOT just one of M rollout generators -- it is
        # the model behind every real decision (the y* pool at 2577, the state
        # reference grid at 2598, candidate features at 2659, _refine_proposal
        # at 2784, inference regret at 3009). The diversity grid anchors it at
        # the SHORTEST lengthscale, which ko_gp.py's own docstring identifies as
        # the near-interpolation / over-fitting regime. Measured cost of that
        # anchor, holding the acquisition fixed (H48 decomposition, MF-MES,
        # Hartmann 6D, 10 seeds): initial_lengthscale alone +0.1058 regret on
        # 10/10 seeds, p=0.0020; rho_init alone +0.0673, p=0.1934 (not
        # significant). So the lengthscale anchor is the knob, and only member
        # 0 needs to change -- members 1..M-1 keep the grid, preserving the
        # rollout diversity it exists for and keeping member 0 in the rollout
        # pool so the train/inference state distributions still match (the
        # point of the earlier Bug A fix at 2591).
        if getattr(config, 'natural_decision_lengthscale', False) and config.M > 1:
            _ls_grid = [None] + torch.linspace(
                _ls_low, _ls_high, config.M - 1).tolist()
        self.ko_ensemble = [
            KennedyOHaganGP(d=self.d, rho_init=_rho_inits[m],
                             initial_lengthscale=_ls_grid[m], **_ko_kwargs)
            for m in range(config.M)
        ]

        state_dim = _get_mf_state_dim(self.d, config.M)
        dt_cfg = SimpleNamespace(
            hidden_size=config.dt_hidden,
            num_layers=config.dt_layers,
            num_heads=config.dt_heads,
            max_seq_length=config.max_seq_length,
            dropout=0.1,
            # score_head input width (Change 1c): d coordinates plus the 7
            # per-candidate GP/MES features build_candidate_features adds,
            # or bare d when that block is ablated off.
            cand_feature_dim=(self.d + N_CAND_EXTRA_FEATURES
                              if self.use_candidate_features else self.d),
            score_temp=getattr(config, 'score_temp', 1.0),
            # H4: "token" (default, unchanged) or "adaln" -- see
            # DecisionTransformer.__init__ and
            # experiments/h4-adaln-rtg-conditioning/protocol.md
            rtg_conditioning=getattr(config, 'rtg_conditioning', 'token'),
            # H20 BUG FIX: this was NEVER forwarded, so DecisionTransformer's
            # `getattr(config, 'use_linear_score_head', True)` always saw a
            # dt_cfg without the attribute and silently defaulted to True.
            # `use_linear_score_head=False` -- documented in findings.md as an
            # available ablation -- was therefore UNREACHABLE. Default is
            # unchanged (True), so no previously-recorded result moves.
            use_linear_score_head=getattr(config, 'use_linear_score_head', True),
        )
        # DT weight-init seed override (weight-init-seed-sweep ablation --
        # tests whether the DT's random weight initialization, independent
        # of everything else config.seed controls, determines whether a
        # given (benchmark, seed) run lands in the frozen-incumbent
        # minimum). ONLY reseeds when config.dt_init_seed is explicitly set
        # by a diagnostic script; when absent (every existing config),
        # this is a pure no-op and self.dt's weights are initialized from
        # whatever RNG state the ensemble-diversity draws above (torch.
        # manual_seed(config.seed+2000) at the top of this method) happened
        # to leave the global generator at -- bit-for-bit unchanged from
        # before this override existed. _sample_initial_points() (run(),
        # called after __init__ returns) unconditionally calls its own
        # torch.manual_seed(self.config.seed) regardless of this, so LHS
        # initialization is never affected either way.
        dt_init_seed = getattr(config, 'dt_init_seed', None)
        if dt_init_seed is not None:
            torch.manual_seed(dt_init_seed)
        self.dt = DecisionTransformer(
            dt_cfg, input_dim=state_dim, action_dim=self.d, use_mf=True
        )
        self.dt = self.dt.float()
        # FIX 3 (post-hoc causal-mask investigation): AdamW + weight decay,
        # matching DRO paper Appendix D.1 -- was plain Adam (no decay).
        self.dt_optimizer = torch.optim.AdamW(
            self.dt.parameters(), lr=config.dt_lr, weight_decay=1e-5
        )
        self.schemas = MFTargetSchemas(
            alpha_rtg=config.alpha_rtg, alpha_btg=config.alpha_btg,
            c_L=config.c_L, c_H=config.c_H
        )

        # SLIDING-WINDOW INFERENCE: the real (state, rtg, btg) actually used at
        # each past real iteration, oldest first. inference_context_k=1 (default)
        # leaves the T=1 path bit-for-bit unchanged.
        self._real_hist = []
        self.inference_context_k = int(getattr(config, 'inference_context_k', 1))
        self.data_hf_x = []
        self.data_hf_y = []
        self.data_lf_x = []
        self.data_lf_y = []
        self.cumulative_cost = 0.0
        self.post_init_cost = 0.0
        self.btg_target_base = None
        self._last_rtg_target = 0.0
        self.iteration_log = []
        # Rolling window of the last 5 REAL fidelity choices (chronological),
        # feeding the recent_hf_frac state feature -- see _extract_mf_state.
        self.recent_ell_history = deque(maxlen=5)

        # Trajectory-level-luck diagnostics (see _generate_rollout_batch),
        # one entry appended per BO iteration (per _generate_rollout_batch
        # call). Always populated, no config gate -- cheap, no oracle access.
        self.action_reward_corr_per_iter = []
        self.rtg_frac_between_traj_var_per_iter = []
        self.rtg_gpbelief_corr_per_iter = []
        # Gradient coherency (see _train_dt): cosine similarity of per-
        # rollout gradients w.r.t. the location head, on a subsample of the
        # batch, computed once per iteration (after training, not per epoch).
        self.grad_coherency_per_iter = []
        # Real-query distance to the benchmark's known optimum location, only
        # populated when config.known_optimal_x is set (Hartmann_6D has a
        # published closed-form location; Currin_2D/Borehole_8D don't -- see
        # run()'s docstring on how those are computed when available).
        self.query_dist_to_xstar_per_iter = []
        # Secondary-basin diagnostic: same computation against a SECOND
        # known local optimum (config.known_secondary_x). Motivated by
        # measuring, on saved traces, that queries sat ~0.97-1.10 from x*
        # but only ~0.62-0.70 from Hartmann-6D's second local minimum
        # x2=[0.405,0.882,0.846,0.574,0.139,0.038] (f=3.2031 on the negated
        # scale, ||x2-x*||=1.1027) -- i.e. the policy was CONVERGING, just
        # on the wrong basin. A flat query_dist_to_xstar alone cannot
        # distinguish that from "not converging at all".
        self.query_dist_to_x2_per_iter = []
        self.p_pred_inference_per_iter = []
        # Last (L_loc, L_fid, fid_mean, fid_std); reused for logging when the
        # DT is frozen (see run()'s freeze_dt_after branch).
        self._last_train_stats = (0.0, 0.0, 0.0, 0.0)
        # H7 (decision divergence): a deepcopy of the DT taken at iteration
        # config.decision_snapshot_at, plus one record per later iteration
        # comparing what the LIVE policy and that SNAPSHOT propose on the
        # IDENTICAL state/RTG/BTG/candidate pool. Measures the DT's marginal
        # contribution as decision agreement (~50-200 paired decisions/run)
        # instead of through final regret (1 noisy scalar/run).
        self._dt_snapshot = None
        self.decision_divergence_log = []
        # Gated behind self._diag_xstar exactly like the [ROLLOUT-DIAG] print
        # block itself (see _generate_rollout_batch) -- stays empty unless a
        # diagnostic script sets _diag_xstar before calling .run(). Records
        # the same frac_rollout_near_xstar value the print statement reports,
        # one entry per BO iteration, so scripts using this hook don't have
        # to scrape stdout to get a per-iteration array back.
        self.diag_frac_rollout_near_xstar_per_iter = []
        # GP-refinement diagnostic (semi-amortized fix ablation, see
        # _refine_proposal): one entry per real BO iteration, ONLY when
        # config.use_gp_refinement is True (_refine_proposal is only ever
        # called from that branch of _propose_next_query) -- stays empty
        # otherwise.
        self.gp_refinement_log = []

    def _sample_initial_points(self):
        """
        Samples initial_points at BOTH fidelities (HF via LHS, LF via a
        separate LHS draw) -- the originally given version of this method
        only sampled HF points, leaving data_lf_x/data_lf_y empty on the
        very first _update_ko_ensemble() call. KennedyOHaganGP.fit's Step 1
        (self.gp_lf = self._build_gp(X_lf, Y_lf)) cannot fit a GP on zero
        points (confirmed: crashes with a 0-element reshape the moment
        _build_gp tries to construct train_targets from an empty tensor),
        so the KO ensemble is unusable without some LF data to seed it,
        exactly like every earlier prompt's ko.fit() calls always supplied
        non-trivial X_lf alongside X_hf.
        """
        from src.utils.init_design import make_initial_design
        torch.manual_seed(self.config.seed)

        # initial_hf/initial_lf (Song 2019 asymmetric init, e.g. 3*d HF +
        # 5*d LF) override the legacy symmetric initial_points when present,
        # so existing configs that only set initial_points keep working.
        n_hf = getattr(self.config, 'initial_hf', None) or self.config.initial_points
        n_lf = getattr(self.config, 'initial_lf', None) or self.config.initial_points

        # LF-screened HF init (attempted freeze fix): place the n_hf HF
        # points where a cheap LF surrogate says they matter, instead of
        # independent random LHS at both fidelities. False (default):
        # original pipeline, bit-for-bit unchanged. See
        # init_design.lf_screened_hf_init's docstring for the full
        # 2-phase procedure.
        use_lf_screened_init = getattr(self.config, 'use_lf_screened_init', False)
        if use_lf_screened_init:
            from src.utils.init_design import lf_screened_hf_init
            X_lf_init, Y_lf_init, X_hf_init = lf_screened_hf_init(
                self.bounds, self.d, n_hf, n_lf, self.config.seed, self.f_lf,
            )
            for x, y in zip(X_lf_init, Y_lf_init.tolist()):
                self.data_lf_x.append(x)
                self.data_lf_y.append(y)
                self.cumulative_cost += self.c_L
            for x in X_hf_init:
                y = self.f_hf(x.unsqueeze(0)).reshape(-1)[0].item()
                self.data_hf_x.append(x)
                self.data_hf_y.append(y)
                self.cumulative_cost += self.c_H
            self.initial_hf_values = list(self.data_hf_y)
            return

        # Both designs (LHS and the sequential max-variance alternative) now
        # live in src/utils/init_design.py so every optimizer in the codebase
        # -- MF-DRO here, plus the MF baselines and the KO-MES/Additive-MES/
        # SF-MES methods -- draws the identical initial design for a given
        # (benchmark, seed, fidelity). Verified bit-for-bit identical to the
        # nested implementations this replaced.
        use_sequential_init = getattr(self.config, 'use_sequential_init', False)

        def _init_points(n, seed_offset):
            return make_initial_design(
                self.bounds, self.d, n, self.config.seed, seed_offset,
                use_sequential_init=use_sequential_init,
            )

        X_hf_init = _init_points(n_hf, seed_offset=0)
        for x in X_hf_init:
            y = self.f_hf(x.unsqueeze(0)).reshape(-1)[0].item()
            self.data_hf_x.append(x)
            self.data_hf_y.append(y)
            self.cumulative_cost += self.c_H
        self.initial_hf_values = list(self.data_hf_y)

        # Distinct seed offset so LF initial points aren't identical to HF's.
        X_lf_init = _init_points(n_lf, seed_offset=1)
        for x in X_lf_init:
            y = self.f_lf(x.unsqueeze(0)).reshape(-1)[0].item()
            self.data_lf_x.append(x)
            self.data_lf_y.append(y)
            self.cumulative_cost += self.c_L

    def _update_ko_ensemble(self):
        if not self.data_hf_x:
            return
        X_hf = torch.stack(self.data_hf_x)
        Y_hf = torch.tensor(self.data_hf_y, dtype=torch.float64)
        if self.data_lf_x:
            X_lf = torch.stack(self.data_lf_x)
            Y_lf = torch.tensor(self.data_lf_y, dtype=torch.float64)
        else:
            X_lf = X_hf[:0]
            Y_lf = Y_hf[:0]
        for ko in self.ko_ensemble:
            ko.fit(X_lf, Y_lf, X_hf, Y_hf, self.bounds)

    #: ITEM 2 teacher pool. "mes" is the current/original acquisition;
    #: cost_ei and ucb_beta{1,3} are cost-normalized EI/UCB over the same
    #: roi_candidates pool; thompson is a joint LF+HF Thompson draw. "random"
    #: is deliberately excluded (no real per-candidate score exists for it,
    #: so it cannot supply a soft teacher_scores target).
    TEACHER_POOL = ["mes", "cost_ei", "ucb_beta1", "ucb_beta3", "thompson"]

    def _generate_rollout_batch(self):
        batch = []
        use_pool = getattr(self.config, 'use_teacher_pool', False)
        for ko in self.ko_ensemble:
            for _ in range(self.config.rollouts_per_model):
                # ITEM 2: teacher sampled UNIFORMLY PER ROLLOUT (one draw for
                # this whole trajectory's rollout_length steps, not per
                # step) when use_teacher_pool is set. Teacher identity is
                # used ONLY to pick which acquisition branch runs inside
                # simulate_mf_trajectory -- it is never written into the
                # state or candidate features (both are built purely from
                # GP posterior/MES quantities, with no policy-identity input
                # anywhere in _extract_mf_state/build_candidate_features).
                _policy = (self.TEACHER_POOL[torch.randint(0, len(self.TEACHER_POOL), (1,)).item()]
                           if use_pool else self.rollout_policy)
                traj = simulate_mf_trajectory(
                    ko,
                    (self.data_hf_x, self.data_hf_y),
                    (self.data_lf_x, self.data_lf_y),
                    rollout_length=self.config.rollout_length,
                    c_H=self.c_H, c_L=self.c_L,
                    bounds=self.bounds,
                    n_real_iter=len(self.data_hf_y),
                    T_real=self.config.bo_iterations,
                    ko_ensemble_full=self.ko_ensemble,
                    ref_grid=self.state_ref_grid,
                    minimum_hf_fraction=getattr(self.config, 'minimum_hf_fraction', 0.25),
                    use_rtg_grounding=getattr(self.config, 'use_rtg_grounding', False),
                    bes_delta=getattr(self.config, 'bes_delta', 0.05),
                    use_candidate_scoring=self.use_candidate_scoring,
                    rollout_policy=_policy,
                    rollout_reward=self.rollout_reward,
                    kg_signed=getattr(self.config, 'kg_signed', False),
                    kg_topk=getattr(self.config, 'kg_topk', 1),
                    fantasy_mode=getattr(self.config, 'fantasy_mode', 'sample'),
                    use_roi=self.use_roi,
                    roi_beta_sqrt=getattr(self.config, 'roi_beta_sqrt', 2.0),
                    roi_raw_pool=getattr(self.config, 'roi_raw_pool', 2000),
                    roi_x_star=self._roi_x_star,
                    roi_stats=self.roi_stats,
                    use_real_rollout_queries=self.use_real_rollout_queries,
                    f_hf_real=(self.f_hf if self.use_real_rollout_queries else None),
                    f_lf_real=(self.f_lf if self.use_real_rollout_queries else None),
                    refit_hyperparams_in_rollout=self.refit_hyperparams_in_rollout,
                    # Change 3b: seed the recent_hf_frac window from the REAL
                    # recent fidelity history, so tau=0 isn't the hardcoded
                    # 0.5 that real inference never presents.
                    recent_ell_seed=list(self.recent_ell_history),
                    use_candidate_features=self.use_candidate_features,
                    use_state_standardization=self.use_state_standardization,
                    y_star_pool=self.y_star_pool,
                    # Seed derived from the BO ITERATION index (not a run
                    # constant), so a single draw's sampling error is not
                    # frozen across the whole run while each iteration's own
                    # features stay internally consistent/deterministic.
                    y_star_seed=self.config.seed + 7919 * len(self.iteration_log),
                )
                batch.append(traj)

        # RTG-CAP FIX: batch-level normalization for rollout_reward==
        # "improvement" (see simulate_mf_trajectory's own comment on why
        # per-trajectory self-normalization was removed). Scale is a
        # RUNNING max of rtg[0] across BO iterations (not just this batch's
        # own), mirroring update_and_get_rtg_target's existing floored-
        # dynamic running-max pattern elsewhere in this class -- keeps the
        # scale stable rather than at the mercy of one iteration's luck.
        # Done here, before anything else reads t['rtg'] (the trajectory-
        # level-luck diagnostics below, update_and_get_rtg_target, and
        # _train_dt all see the corrected values).
        # H10 (rtg_target_mode): "normalized" (default, unchanged) divides by a
        # running max, which makes batch_max = this-batch-best / best-ever --
        # pinned near 1 by construction, giving the measured [0.57, 1.0] band.
        # "raw" skips normalisation entirely so the target is the raw
        # improvement, which spans orders of magnitude over a run by
        # construction. Tests whether RTG is ignored or merely starved.
        if self.rollout_reward in ("improvement", "kg_incumbent") and \
                getattr(self.config, 'rtg_target_mode', 'normalized') != 'raw':
            _batch_max = max((t['rtg'][0].item() for t in batch if t['rtg'].numel() > 0),
                              default=0.0)
            self._running_max_rtg_raw = max(
                getattr(self, '_running_max_rtg_raw', 0.0), _batch_max)
            _norm_scale = max(self._running_max_rtg_raw, 1e-8)
            for t in batch:
                if t['rtg'].numel() > 0:
                    t['rtg'] = t['rtg'] / _norm_scale

        # VERIFICATION (spec): tau=0 state duplication + reference-block
        # variation, printed once per real BO iteration.
        s0 = torch.stack([t['states'][0] for t in batch])          # [B, state_dim]
        n_uniq = len(set(tuple(r.tolist()) for r in s0))
        ref_block_0 = s0[:, -4 * STATE_REF_GRID_R:]                # [B, 4R]
        std_across = ref_block_0.std(dim=0).mean().item()
        # ref-block std ACROSS tau WITHIN a trajectory, averaged over
        # trajectories (uses each trajectory's own possibly-BES-shortened T).
        within = [t['states'][:, -4 * STATE_REF_GRID_R:].std(dim=0).mean().item()
                  for t in batch if t['states'].shape[0] > 1]
        std_within = float(np.mean(within)) if within else float('nan')
        print(f"[STATE-DIAG] n_traj={len(batch)} uniq_tau0_states={n_uniq} "
              f"ref_block_std_across_traj_at_tau0={std_across:.6f} "
              f"ref_block_std_across_tau_within_traj={std_within:.6f}", flush=True)

        # VERIFICATION (spec): the per-candidate feature tensor built on the
        # TRAINING path and the one built on the INFERENCE path must have
        # identical shape (except K, which differs by design: K_cands=20
        # while training, n_infer_candidates=200 at inference) and identical
        # dtype. Compared once per iteration against whatever
        # _propose_next_query last built.
        if self.use_candidate_scoring and getattr(self, '_last_cand_feats', None) is not None:
            train_cf = batch[0]['candidates']          # [T, K_cands, F]
            infer_cf = self._last_cand_feats           # [n_infer, F]
            assert train_cf.shape[-1] == infer_cf.shape[-1], (
                f"candidate feature width mismatch: train={train_cf.shape[-1]} "
                f"infer={infer_cf.shape[-1]}"
            )
            assert train_cf.dtype == infer_cf.dtype, (
                f"candidate feature dtype mismatch: train={train_cf.dtype} "
                f"infer={infer_cf.dtype}"
            )

        # Rollout-exploration diagnostic (oracle-init investigation): does
        # the DT's simulated-rollout training data explore new regions, or
        # collapse near the incumbent immediately after initialization?
        # Gated behind self._diag_xstar (unset/None by default -- never
        # runs/prints in the normal pipeline); a diagnostic script sets it
        # to a raw-scale [d] tensor before calling .run() to activate.
        xstar = getattr(self, '_diag_xstar', None)
        if xstar is not None and self.data_hf_y:
            best_idx = int(np.argmax(self.data_hf_y))
            best_pos_raw = self.data_hf_x[best_idx]
            best_pos_norm = (best_pos_raw - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
            xstar_norm = (xstar - self.bounds[0]) / (self.bounds[1] - self.bounds[0])

            all_x = []
            for traj in batch:
                if 'actions_x' in traj:
                    all_x.append(traj['actions_x'])
                else:
                    idx_exp = traj['chosen_idx'].unsqueeze(-1).unsqueeze(-1).expand(
                        -1, 1, traj['candidates'].shape[-1]
                    )
                    # [:, :self.d] -- candidates carry d coordinates plus
                    # Change 1c's per-candidate feature columns; only the
                    # coordinates are a location.
                    all_x.append(
                        traj['candidates'].gather(1, idx_exp).squeeze(1)[:, :self.d]
                    )
            all_x = torch.cat(all_x, dim=0)  # [total_rollout_steps, d]

            dist_incumbent = (all_x - best_pos_norm).norm(dim=-1)
            dist_xstar = (all_x - xstar_norm).norm(dim=-1)
            frac_near_xstar = (dist_xstar < 0.3).float().mean().item()
            self.diag_frac_rollout_near_xstar_per_iter.append(frac_near_xstar)
            print(
                f"[ROLLOUT-DIAG] n_hf={len(self.data_hf_y)} best_hf={max(self.data_hf_y):.4f} "
                f"mean_dist_rollout_to_incumbent={dist_incumbent.mean().item():.4f} "
                f"mean_dist_rollout_to_xstar={dist_xstar.mean().item():.4f} "
                f"frac_rollout_near_incumbent={(dist_incumbent < 0.3).float().mean().item():.3f} "
                f"frac_rollout_near_xstar={frac_near_xstar:.3f}",
                flush=True,
            )

            # RTG-vs-quality diagnostic: RTG_tau = log(b_tau/b_T) is a
            # function of current_ko's Gumbel-fit posterior spread alone
            # (see simulate_mf_trajectory's b_tau/b_values computation) --
            # it never reads x_tau or y_tau. This checks directly whether
            # that construction happens to still correlate with how GOOD
            # the step's chosen location actually was (true f_hf value,
            # oracle-only, measurement never fed back into training/cost),
            # or whether it's statistically independent of location
            # quality -- the concrete, code-grounded version of "does the
            # reward signal actually discriminate good rollout steps from
            # bad ones" this diagnostic exists to answer.
            all_rtg = torch.cat([traj['rtg'] for traj in batch], dim=0)
            if all_rtg.numel() == all_x.shape[0] and all_rtg.numel() > 1:
                all_x_raw = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * all_x
                true_y = torch.tensor(
                    [self.f_hf(x.unsqueeze(0)).reshape(-1)[0].item() for x in all_x_raw],
                    dtype=all_rtg.dtype,
                )
                def _corr(a, b):
                    a = a - a.mean(); b = b - b.mean()
                    denom = a.norm() * b.norm()
                    return (a @ b / denom).item() if denom > 1e-12 else float('nan')
                corr_rtg_true_y = _corr(all_rtg, true_y)
                corr_rtg_dist_xstar = _corr(all_rtg, dist_xstar)
                print(
                    f"[RTG-DIAG] corr(rtg, true_f_hf)={corr_rtg_true_y:.4f} "
                    f"corr(rtg, dist_to_xstar)={corr_rtg_dist_xstar:.4f} "
                    f"(expect corr(rtg,true_f_hf)>0 and corr(rtg,dist)<0 if RTG "
                    f"tracks location quality; near-0 means RTG is statistically "
                    f"independent of it)",
                    flush=True,
                )

        # Trajectory-level-luck diagnostics (always computed, cheap -- O(total
        # rollout steps), no oracle/ground-truth access, so these run on every
        # real Stage 2 job, unlike the [RTG-DIAG] block above which needs
        # self._diag_xstar. Hypothesis under test: DT's training data labels
        # each state by its OWN trajectory's RTG, mixing trajectory-level luck
        # (did THIS simulated continuation happen to sample high y) with
        # state-level quality (is x_tau intrinsically valuable) -- a state
        # embedded in an unlucky trajectory gets a low RTG label regardless of
        # its own merit, and the DT cannot tell the two apart from the label
        # alone.
        incumbent_y = max(self.data_hf_y) if self.data_hf_y else 0.0
        rtg_per_traj = [t['rtg'] for t in batch if t['rtg'].numel() > 0]
        y_per_traj = [t['y_values'] for t in batch if t['rtg'].numel() > 0]
        if rtg_per_traj:
            def _corr(a, b):
                a = a - a.mean(); b = b - b.mean()
                denom = a.norm() * b.norm()
                return (a @ b / denom).item() if denom > 1e-12 else float('nan')

            # (a) Action-reward correlation: corr(RTG_tau, y_tau - incumbent)
            # pooled across every step in the batch. Near-zero means the RTG
            # label doesn't track whether THIS step's own observation was
            # good -- consistent with the label being dominated by something
            # else (e.g. how the rest of that trajectory happened to unfold).
            all_rtg_luck = torch.cat(rtg_per_traj)
            all_y_gap = torch.cat(y_per_traj) - incumbent_y
            action_reward_corr = _corr(all_rtg_luck, all_y_gap)

            # (b) RTG variance decomposition: what fraction of total RTG
            # variance is explained by WHICH TRAJECTORY a step came from
            # (between-trajectory) vs. variance within a single trajectory
            # (across its own tau steps)? A high between-trajectory share is
            # the direct signature of trajectory-level-luck contamination --
            # "which rollout this step was simulated in" matters more to its
            # RTG label than anything about the step itself.
            traj_means = torch.stack([r.mean() for r in rtg_per_traj])
            traj_vars = torch.stack([
                r.var(unbiased=False) if r.numel() > 1 else torch.zeros((), dtype=r.dtype)
                for r in rtg_per_traj
            ])
            between_var = traj_means.var(unbiased=False).item()
            within_var = traj_vars.mean().item()
            total_var = between_var + within_var
            frac_between_traj_var = (between_var / total_var) if total_var > 1e-12 else float('nan')

            # (c) RTG vs. GP-belief correlation: corr(RTG_tau, GP posterior
            # mean at x_tau) using ko_ensemble[0] as a fixed, trajectory-
            # INDEPENDENT "how good is this state, per the model's own
            # current belief" reference -- unlike (a), which is noisy at the
            # single-trajectory level, this checks whether RTG tracks the
            # model's own aggregate notion of state quality at all.
            all_x_norm_luck = []
            for t in batch:
                if t['rtg'].numel() == 0:
                    continue
                if 'actions_x' in t:
                    all_x_norm_luck.append(t['actions_x'])
                else:
                    idx_exp = t['chosen_idx'].unsqueeze(-1).unsqueeze(-1).expand(
                        -1, 1, t['candidates'].shape[-1]
                    )
                    # Coordinates only -- see the identical slice in the
                    # [ROLLOUT-DIAG] block above.
                    all_x_norm_luck.append(
                        t['candidates'].gather(1, idx_exp).squeeze(1)[:, :self.d]
                    )
            all_x_norm_luck = torch.cat(all_x_norm_luck, dim=0)
            all_x_raw_luck = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * all_x_norm_luck
            gp_mu, _ = self.ko_ensemble[0].hf_posterior(all_x_raw_luck)
            rtg_gpbelief_corr = _corr(all_rtg_luck, gp_mu)

            self.action_reward_corr_per_iter.append(action_reward_corr)
            self.rtg_frac_between_traj_var_per_iter.append(frac_between_traj_var)
            self.rtg_gpbelief_corr_per_iter.append(rtg_gpbelief_corr)

        return batch

    def _train_dt(self, batch):
        """
        Trains on a batch of rollouts whose lengths may differ (Change 1,
        Bayesian Early Stopping can stop a rollout short of
        self.config.rollout_length). Every trajectory is right-padded with
        zeros up to T_max=rollout_length and a per-position valid_mask marks
        which positions are real vs padding; forward_mf uses valid_mask to
        exclude padded positions from L_loc/L_fid (see its own docstring)
        instead of the padding zeros silently diluting the loss. A batch
        where every trajectory already has length rollout_length (BES
        disabled, or never triggered) produces an all-True valid_mask,
        which is NOT the same code path as valid_mask=None -- see CHECK 5
        for why that's still equivalent to the pre-padding behavior.

        self.use_candidate_scoring=False (default): regression path,
        bit-for-bit unchanged from before this flag existed. =True: builds
        candidates/chosen_idx tensors instead of actions_x (see
        simulate_mf_trajectory's docstring for what each trajectory dict
        contains in which mode).
        """
        self.dt.train()
        T_max = self.config.rollout_length
        B = len(batch)
        use_cs = self.use_candidate_scoring

        state_dim = _get_mf_state_dim(self.d, self.config.M)
        states = torch.zeros(B, T_max, state_dim)
        actions_ell = torch.zeros(B, T_max, dtype=torch.long)
        rtg = torch.zeros(B, T_max)
        btg = torch.zeros(B, T_max)
        valid_mask = torch.zeros(B, T_max, dtype=torch.bool)

        if use_cs:
            K = batch[0]['candidates'].shape[1]
            # Last dim is the per-candidate FEATURE width (d + 7 when
            # use_candidate_features, else d) -- read it off the data rather
            # than assuming self.d, so the ablation flag needs no special
            # case here.
            cand_dim = batch[0]['candidates'].shape[-1]
            candidates = torch.zeros(B, T_max, K, cand_dim)
            chosen_idx = torch.zeros(B, T_max, dtype=torch.long)
            teacher_scores = torch.zeros(B, T_max, K)
            has_soft = torch.zeros(B, T_max, dtype=torch.bool)
            actions_x = None
        else:
            actions_x = torch.zeros(B, T_max, self.d)
            candidates = None
            chosen_idx = None
            teacher_scores = None
            has_soft = None

        for i, t in enumerate(batch):
            T_i = t['states'].shape[0]  # actual (possibly BES-shortened) length
            states[i, :T_i] = t['states']
            actions_ell[i, :T_i] = t['actions_ell']
            rtg[i, :T_i] = t['rtg']
            btg[i, :T_i] = t['btg']
            valid_mask[i, :T_i] = True
            if use_cs:
                candidates[i, :T_i] = t['candidates']
                chosen_idx[i, :T_i] = t['chosen_idx']
                teacher_scores[i, :T_i] = t['teacher_scores']
                has_soft[i, :T_i] = t['has_soft']
            else:
                actions_x[i, :T_i] = t['actions_x']

        timesteps = torch.arange(T_max).unsqueeze(0).repeat(B, 1)

        # ── Required instrumentation (CRITICAL-2 + ISSUE-4), first BO
        # iteration only. Reports (a) teacher entropy before vs after
        # within-set standardization against log(K), so a near-uniform
        # (i.e. useless) soft target cannot pass silently as a small L_loc,
        # and (b) L_loc vs lambda_fid*L_fid magnitudes, since L_loc moved
        # from MSE on [0,1]^d (O(0.01-0.1)) to KL/CE over K candidates
        # (O(0-3)) and lambda_fid was NOT retuned to match.
        if use_cs and not getattr(self, '_loss_diag_done', False):
            self._loss_diag_done = True
            with torch.no_grad():
                vm = valid_mask.reshape(B * T_max)
                tf = teacher_scores.reshape(B * T_max, K)[vm]
                hs = has_soft.reshape(B * T_max)[vm]
                if hs.any():
                    ts = tf[hs]
                    p_raw = torch.softmax(ts / self.dt.score_temp, dim=-1)
                    H_raw = -(p_raw * p_raw.clamp_min(1e-12).log()).sum(-1).mean().item()
                    zs = (ts - ts.mean(-1, keepdim=True)) / (ts.std(-1, keepdim=True) + 1e-8)
                    p_std = torch.softmax(zs / self.dt.score_temp, dim=-1)
                    H_std = -(p_std * p_std.clamp_min(1e-12).log()).sum(-1).mean().item()
                else:
                    H_raw = H_std = float('nan')
                logK = math.log(K)
                # Same batch, both objectives, for a like-for-like L_loc read.
                _, L_soft, L_fid_d, _, _ = self.dt.forward_mf(
                    states.float(), actions_ell, rtg.float(), btg.float(), timesteps,
                    candidates=candidates.float(), chosen_idx=chosen_idx,
                    valid_mask=valid_mask, use_candidate_scoring=True,
                    teacher_scores=teacher_scores.float(), has_soft=has_soft)
                _, L_hard, _, _, _ = self.dt.forward_mf(
                    states.float(), actions_ell, rtg.float(), btg.float(), timesteps,
                    candidates=candidates.float(), chosen_idx=chosen_idx,
                    valid_mask=valid_mask, use_candidate_scoring=True,
                    teacher_scores=None, has_soft=None)
                print(f"[LOSS-DIAG] K={K} log(K)={logK:.4f} | teacher_entropy "
                      f"raw={H_raw:.4f} ({H_raw / logK * 100:.1f}% of log K) "
                      f"standardized={H_std:.4f} ({H_std / logK * 100:.1f}% of log K)",
                      flush=True)
                print(f"[LOSS-DIAG] L_loc(soft KL)={L_soft.item():.6f} "
                      f"L_loc(hard CE)={L_hard.item():.6f} "
                      f"L_fid={L_fid_d.item():.6f} lambda_fid={self.dt.lambda_fid} "
                      f"lambda_fid*L_fid={self.dt.lambda_fid * L_fid_d.item():.6f}",
                      flush=True)

        L_loc_final = L_fid_final = 0.0
        for epoch in range(self.config.num_epochs):
            self.dt_optimizer.zero_grad()
            loss, L_loc, L_fid, _, p_pred = self.dt.forward_mf(
                states.float(), actions_ell,
                rtg.float(), btg.float(), timesteps,
                actions_x=(actions_x.float() if actions_x is not None else None),
                candidates=(candidates.float() if candidates is not None else None),
                chosen_idx=chosen_idx,
                valid_mask=valid_mask,
                use_candidate_scoring=use_cs,
                teacher_scores=(teacher_scores.float()
                                if (teacher_scores is not None
                                    and self.use_soft_score_target) else None),
                has_soft=(has_soft if self.use_soft_score_target else None),
            )
            loss.backward()

            # Gradient-norm diagnostic (policy-collapse vs data-quality
            # investigation): does the DT's location/fidelity heads
            # actually receive gradient signal through the RTG embedding,
            # or does the state embedding dominate? Read BEFORE optimizer
            # .step() (standard convention -- .step() doesn't touch
            # .grad, but the gradient is only meaningful pre-update).
            # Gated behind self._diag_grad_norms (unset/False by default).
            if getattr(self, '_diag_grad_norms', False):
                rtg_grad_norm = self.dt.reward_embedding.weight.grad.norm().item()
                state_grad_norm = self.dt.state_embedding.weight.grad.norm().item()
                print(
                    f"[GRAD-DIAG] epoch={epoch} L_loc={L_loc.item():.4f} "
                    f"L_fid={L_fid.item():.4f} "
                    f"gradient_norm_rtg_embedding={rtg_grad_norm:.6f} "
                    f"gradient_norm_state_embedding={state_grad_norm:.6f}",
                    flush=True,
                )

            self.dt_optimizer.step()
            L_loc_final = L_loc.item()
            L_fid_final = L_fid.item()

        valid_p = p_pred[valid_mask]
        fid_mean = valid_p.detach().mean().item() if valid_p.numel() > 0 else 0.5
        fid_std = valid_p.detach().std().item() if valid_p.numel() > 1 else 0.0

        # Gradient coherency (always computed, once per iteration -- not per
        # epoch -- on a K=8 subsample of the batch, to bound cost against the
        # much larger num_epochs full-batch training loop above). Takes K
        # individual rollouts, computes grad(L_loc_i, location_head_params)
        # for each independently via torch.autograd.grad (no .backward(),
        # doesn't touch the optimizer's own .grad state), and measures mean
        # pairwise cosine similarity between the flattened per-example
        # gradients. High coherency = every rollout's own loss pulls the
        # location head in roughly the same direction; collapsing toward ~0
        # or negative = rollouts disagree with each other -- consistent with
        # trajectory-level-luck contaminating individual examples' training
        # signal in conflicting directions (see _generate_rollout_batch's
        # trajectory-level-luck diagnostics for the related RTG-side checks).
        # ITEM 5: when the linear coefficient score head is active, score_head
        # itself receives no gradient (it isn't used in the forward pass),
        # so this diagnostic must target coef_head+bias_head instead -- or
        # torch.autograd.grad returns all-None and the per-example gradient
        # stack below has nothing to concatenate.
        if use_cs and getattr(self.dt, 'use_linear_score_head', False):
            target_params = list(self.dt.coef_head.parameters()) + list(self.dt.bias_head.parameters())
        elif use_cs:
            target_params = list(self.dt.score_head.parameters())
        else:
            target_params = list(self.dt.action_head.parameters())
        K_grad = min(8, B)
        grad_idx = torch.randperm(B)[:K_grad].tolist()
        per_example_grads = []
        for i in grad_idx:
            _, L_loc_i, _, _, _ = self.dt.forward_mf(
                states[i:i + 1].float(), actions_ell[i:i + 1],
                rtg[i:i + 1].float(), btg[i:i + 1].float(), timesteps[i:i + 1],
                actions_x=(actions_x[i:i + 1].float() if actions_x is not None else None),
                candidates=(candidates[i:i + 1].float() if candidates is not None else None),
                chosen_idx=(chosen_idx[i:i + 1] if chosen_idx is not None else None),
                valid_mask=valid_mask[i:i + 1],
                use_candidate_scoring=use_cs,
                teacher_scores=(teacher_scores[i:i + 1].float()
                                if (teacher_scores is not None
                                    and self.use_soft_score_target) else None),
                # has_soft is None on the REGRESSION path (no candidate sets
                # exist to carry soft targets), so guard on the tensor itself
                # and not only on the flag -- otherwise
                # use_candidate_scoring=False raises TypeError here.
                has_soft=(has_soft[i:i + 1]
                          if (has_soft is not None and self.use_soft_score_target)
                          else None),
            )
            grads_i = torch.autograd.grad(L_loc_i, target_params, allow_unused=True)
            flat_i = torch.cat([g.flatten() for g in grads_i if g is not None])
            per_example_grads.append(flat_i)

        if len(per_example_grads) >= 2:
            G = torch.stack(per_example_grads)
            G = G / (G.norm(dim=1, keepdim=True) + 1e-12)
            cos_sim = G @ G.T
            n_pairs = G.shape[0] * (G.shape[0] - 1)
            grad_coherency = ((cos_sim.sum() - torch.diagonal(cos_sim).sum()) / n_pairs).item()
        else:
            grad_coherency = float('nan')
        self.grad_coherency_per_iter.append(grad_coherency)

        return L_loc_final, L_fid_final, fid_mean, fid_std

    def _propose_next_query(self):
        # recent_hf_frac reflects real queries 0..t-1 only (history BEFORE
        # this iteration's own decision) -- exposed via _last_recent_hf_frac
        # for diagnostics, same pattern as _last_btg_now/_last_rtg_target.
        self._last_recent_hf_frac = (
            sum(self.recent_ell_history) / max(len(self.recent_ell_history), 1)
            if self.recent_ell_history else 0.5
        )
        # No roi_candidates here (Fix 1): _extract_mf_state no longer needs
        # any candidate set (see its docstring), so the ROI-filtered pool
        # this used to build purely to feed it is gone too.
        _y_star_now = _y_star_for_model(
            self.ko_ensemble[0], self.y_star_pool,
            # Same iteration-derived seed as this iteration's training
            # rollouts (see _generate_rollout_batch), so train and inference
            # share one y* draw per BO iteration.
            seed=self.config.seed + 7919 * len(self.iteration_log))
        state = _extract_mf_state(
            self.data_hf_x, self.data_hf_y,
            self.data_lf_x, self.data_lf_y,
            self.ko_ensemble,
            len(self.data_hf_y), self.config.bo_iterations,
            self.c_L, self.c_H,
            device='cpu', dtype=torch.float64,
            recent_hf_frac=self._last_recent_hf_frac,
            bounds=self.bounds,
            # Fix for Bug A: ko_ensemble[0] as the single representative
            # model for the reference-grid block, matching the same
            # single-model convention _refine_proposal already uses for
            # real-inference GP queries. self.state_ref_grid is the SAME
            # fixed grid every training rollout uses (see
            # _generate_rollout_batch) -- shared reference distribution
            # between train and inference is the whole point of this fix.
            ref_grid=self.state_ref_grid, ref_model=self.ko_ensemble[0],
            # CRITICAL-1: ONE y* draw for this whole proposal, from the same
            # fixed pool and same seed the training path uses, shared by the
            # ref-grid state block and the candidate block below.
            y_star_arr=_y_star_now,
            standardize=self.use_state_standardization,
        )
        self._last_state = state  # exposed for the RTG sensitivity probe diagnostic
        # btg_target_base comes from a short simulated rollout (rollout_length
        # steps, representing a brief lookahead budget), while cumulative_cost
        # accumulates over the ENTIRE real BO run including initial sampling
        # -- decrementing the former by the latter is a category mismatch,
        # not the "budget remaining in this episode" the get_inference_btg
        # decay was designed for. With initial-sampling cost alone often
        # exceeding btg_target_base (e.g. Hartmann initial_points=3, c_H=8 ->
        # 27 before iteration 0 even starts), this clamped btg_now to c_L on
        # every real iteration, training the DT to associate BTG=c_L with
        # "always pick the cheapest action" regardless of what was actually
        # happening. Each real BO iteration triggers a FRESH rollout batch
        # (see run()), so there is no single depleting episode budget to
        # decrement against -- every real query is the start of a new
        # decision horizon, analogous to tau=0 in a simulated rollout, where
        # btg[0] == btg_target_base by construction. btg_target_base is now
        # updated every iteration (Bug B fix, see run()/
        # update_and_get_btg_target) rather than frozen at iteration 0's
        # value -- self.btg_target_base always reflects the MOST RECENT
        # iteration's batch-mean rollout cost, not a permanent constant.
        btg_now = self.btg_target_base
        self._last_btg_now = btg_now  # exposed for diagnostics/verification
        rtg_tgt = self._last_rtg_target
        # Position embeddings are only ever trained on rollout step indices
        # 0..rollout_length-1 (see _train_dt: timesteps = arange(T)). At real
        # inference the DT is always at the start of a fresh decision horizon
        # -- same "tau=0" framing already used for btg_now above -- not at
        # whatever position len(self.data_hf_y) happens to be (which starts
        # at initial_points, already past the trained range on iteration 0,
        # and only grows further). Passing that count sent the DT an
        # out-of-distribution position index it never received gradient
        # updates for.
        # Change 1b/1c: inference candidate pool -- N_infer fresh uniform
        # draws over the domain, then the SAME build_candidate_features the
        # training path uses. simulate_mf_trajectory's roi_candidates is
        # also uniform over bounds, so the two paths score structurally and
        # distributionally identical inputs.
        cand_feats = None
        if self.use_candidate_scoring:
            X_cand_raw = (self.bounds[0] + (self.bounds[1] - self.bounds[0])
                          * torch.rand(self.n_infer_candidates, self.d,
                                        dtype=torch.float64))
            if self.data_hf_y:
                _bi = int(np.argmax(self.data_hf_y))
                _best_pos_norm = ((self.data_hf_x[_bi] - self.bounds[0])
                                  / (self.bounds[1] - self.bounds[0])).clamp(0, 1)
            else:
                _best_pos_norm = torch.zeros(self.d, dtype=torch.float64)
            if self.use_candidate_features:
                _ym = (float(np.mean(self.data_hf_y))
                       if (self.use_state_standardization and self.data_hf_y) else 0.0)
                _ys = (max(float(np.std(self.data_hf_y)), 1e-6)
                       if (self.use_state_standardization and self.data_hf_y) else 1.0)
                cand_feats = build_candidate_features(
                    self.ko_ensemble[0], X_cand_raw, self.bounds,
                    self.c_H, self.c_L, _best_pos_norm,
                    y_star_arr=_y_star_now, y_mean=_ym, y_std=_ys,
                )
            else:
                cand_feats = ((X_cand_raw - self.bounds[0])
                              / (self.bounds[1] - self.bounds[0])).clamp(0, 1)
            self._last_cand_feats = cand_feats  # for the shape/dtype verification

        # Change 3a: eval mode + no_grad around the real proposal.
        # propose_mf already does both internally (verified -- it calls
        # self.eval(), wraps in torch.no_grad(), then restores self.train()),
        # so this is belt-and-braces: it makes the guarantee explicit at the
        # call site and holds even if propose_mf's internals change.
        self.dt.eval()
        with torch.no_grad():
            _K = self.inference_context_k
            _hist = None
            if _K > 1 and self._real_hist:
                _hist = [{'state': h['state'].float(), 'rtg': h['rtg'], 'btg': h['btg']}
                          for h in self._real_hist[-(_K - 1):]]
            self._last_ctx_len = (len(_hist) + 1) if _hist else 1
            x_t, ell_t = self.dt.propose_mf(
                state.float(), rtg_tgt, btg_now,
                timestep=0,
                use_candidate_scoring=self.use_candidate_scoring,
                candidate_features=(cand_feats.float() if cand_feats is not None else None),
                fidelity_sampling=self.fidelity_sampling,
                hist=_hist,
            )
            # Record AFTER proposing, so the current step never sees itself.
            self._real_hist.append({'state': state.detach().clone(),
                                     'rtg': float(rtg_tgt), 'btg': float(btg_now)})
        # H7: replay the SAME inputs through the iteration-k snapshot. Nothing
        # here is executed -- only the LIVE x_t/ell_t below drive the run --
        # so the trajectory is an ordinary MF-DRO run and the evaluation is
        # untouched. Records whether the extra training changed the decision.
        if self._dt_snapshot is not None:
            with torch.no_grad():
                x_s, ell_s = self._dt_snapshot.propose_mf(
                    state.float(), rtg_tgt, btg_now,
                    timestep=0,
                    use_candidate_scoring=self.use_candidate_scoring,
                    candidate_features=(cand_feats.float() if cand_feats is not None else None),
                    fidelity_sampling=False,   # deterministic: isolate the policy, not the coin flip
                )
            _dist = (x_t.double() - x_s.double()).norm().item()
            self.decision_divergence_log.append({
                'iter': len(self.iteration_log),
                'argmax_agree': bool(_dist < 1e-9),
                'dist': _dist,
                'fid_agree': bool(int(ell_t) == int(ell_s)),
            })

        # propose_mf's location head is normalized to [0,1]^d (see its own
        # x_pred.clamp(0.0, 1.0)) -- rescale to the benchmark's actual
        # domain bounds HERE, the single exit point for real x_t, so every
        # downstream consumer (f_hf/f_lf evaluation, data_hf_x/data_lf_x
        # storage, x_t_trace logging) sees genuine domain-scale
        # coordinates. Previously missing entirely: silently correct only
        # because Currin_2D/Hartmann_6D's domain IS [0,1]^d; on
        # Borehole_8D every real query was being evaluated inside [0,1]^8
        # instead of the actual domain (e.g. rw in [0,1] instead of
        # [0.05,0.15], r in [0,1] instead of [100,50000]). See
        # REVISION_LOG.md.
        x_t = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * x_t.double()

        # Semi-amortized GP-refinement ablation (see _refine_proposal's own
        # docstring): warm-starts gradient ascent on EI from the DT's
        # proposal, replacing the QUERIED location only -- ell_t (fidelity)
        # still comes from the DT untouched. Off (getattr default False) for
        # every existing config/run; only set True by this ablation's own
        # worker script.
        if getattr(self.config, 'use_gp_refinement', False):
            x_t = self._refine_proposal(x_t, ell_t)

        return x_t, ell_t

    def _refine_proposal(self, x_init, ell_t):
        """
        Warm-start gradient ascent on an acquisition function (of the HF
        posterior) starting from the DT's proposal x_init, for
        config.gp_refinement_steps Adam steps at lr=config.gp_refinement_lr.
        config.gp_refinement_acquisition selects the acquisition: "ei"
        (default -- Expected Improvement, the original version of this
        ablation) or "ucb" (mu + beta*sigma, config.gp_refinement_ucb_beta,
        default 2.0) -- added when EI's gradient was found to vanish almost
        everywhere once the GP's HF posterior gets confident (median EI
        collapsed to ~0.0 by the back half of a 380-iteration run; UCB's
        sigma term stays gradient-informative even when EI's improvement
        term does not). x_init/return value are RAW domain-scale (matching
        self.bounds), NOT normalized [0,1]^d -- called from
        _propose_next_query AFTER that method's own [0,1]^d -> raw-domain
        rescale, not before. Reason: ko.gp_lf/gp_delta (the actual BoTorch
        SingleTaskGPs backing the KO ensemble) are built with
        input_transform=Normalize(d, bounds=self.bounds) (src/models/
        ko_gp.py) -- their own .posterior(X) calls expect raw domain-scale
        X and normalize internally, exactly like every other consumer of
        this ensemble in this class (hf_posterior, sample_fantasy,
        make_fantasy_ko all take raw-domain X too). Refining a [0,1]^d-
        normalized x through them would be silently wrong on any benchmark
        whose raw domain isn't already [0,1]^d -- a non-issue for THIS
        ablation's only benchmark (Hartmann_6D's raw domain IS [0,1]^6, so
        normalized and raw-domain-scale coincide here), but matters for any
        future use on a benchmark like Borehole_8D.

        Does NOT call self.ko_ensemble[0].hf_posterior(...) despite the
        original EI spec calling for it: hf_posterior wraps its entire body
        in `with torch.no_grad(), gpytorch.settings.fast_pred_var():` (see
        ko_gp.py), which silently severs the gradient back to x --
        confirmed directly: (-acq).backward() through hf_posterior raises
        "element 0 of tensors does not require grad and does not have a
        grad_fn". This method instead reimplements the SAME mu_H/var_H
        combination (mu_H = rho*mu_L + mu_delta, var_H = rho^2*var_L +
        var_delta) by calling ko.gp_lf.posterior/ko.gp_delta.posterior
        directly, without the no_grad wrapper -- verified this preserves
        gradients w.r.t. x. rho itself is still detached (ko.rho.detach(),
        matching hf_posterior's own rho_val = self.rho.detach()): only x is
        being optimized here, not the GP's fitted hyperparameters.

        Logs one entry per call to self.gp_refinement_log (acq_type,
        acq_at_x_dt, acq_at_x_refined, sigma_at_x_dt, and, when
        config.known_optimal_x is set, x_dt_dist/x_refined_dist to it) --
        see __init__.
        """
        ko = self.ko_ensemble[0]
        best = max(self.data_hf_y) if self.data_hf_y else -1e9
        lo, hi = self.bounds[0], self.bounds[1]
        normal = torch.distributions.Normal(0, 1)
        acq_type = getattr(self.config, 'gp_refinement_acquisition', 'ei')
        beta = getattr(self.config, 'gp_refinement_ucb_beta', 2.0)

        def _posterior(x_pt):
            with gpytorch.settings.fast_pred_var():
                post_lf = ko.gp_lf.posterior(x_pt.unsqueeze(0))
                post_delta = ko.gp_delta.posterior(x_pt.unsqueeze(0))
                mu_lf = post_lf.mean.reshape(-1)
                var_lf = post_lf.variance.clamp_min(1e-12).reshape(-1)
                mu_delta = post_delta.mean.reshape(-1)
                var_delta = post_delta.variance.clamp_min(1e-12).reshape(-1)
                rho_val = ko.rho.detach()
                mu = rho_val * mu_lf + mu_delta
                var = rho_val ** 2 * var_lf + var_delta
            sigma = var.clamp_min(1e-8).sqrt()
            return mu, sigma

        def _acq(x_pt):
            mu, sigma = _posterior(x_pt)
            if acq_type == 'ucb':
                val = mu + beta * sigma
            else:
                z = (mu - best) / sigma
                val = (mu - best) * normal.cdf(z) + sigma * torch.exp(normal.log_prob(z))
            return val.reshape(())

        with torch.no_grad():
            acq_init = _acq(x_init).item()
            sigma_at_x_dt = _posterior(x_init)[1].item()

        x = x_init.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=self.config.gp_refinement_lr)
        for _ in range(self.config.gp_refinement_steps):
            opt.zero_grad()
            acq = _acq(x)
            (-acq).backward()
            opt.step()
            with torch.no_grad():
                x.clamp_(lo, hi)

        with torch.no_grad():
            acq_final = _acq(x).item()

        entry = {
            'acq_type': acq_type,
            'acq_at_x_dt': acq_init,
            'acq_at_x_refined': acq_final,
            'sigma_at_x_dt': sigma_at_x_dt,
        }
        known_x = getattr(self.config, 'known_optimal_x', None)
        if known_x is not None:
            xstar_t = torch.as_tensor(known_x, dtype=x_init.dtype, device=x_init.device)
            entry['x_dt_dist'] = (x_init - xstar_t).norm().item()
            entry['x_refined_dist'] = (x.detach() - xstar_t).norm().item()
        self.gp_refinement_log.append(entry)

        return x.detach()

    def _check_lf_pathology(self):
        # Warning only -- does NOT stop the run. Previously raised
        # RuntimeError here to terminate early the moment 10 consecutive
        # real queries were all LF; for Currin_2D that fired within the
        # first 10 iterations essentially every time, before the run ever
        # got far enough to show whether the resulting all-LF policy
        # actually finds a good optimum or is genuinely stuck -- the two
        # look identical from an early-stopped trace alone. Let the full
        # bo_iterations budget run and evaluate regret at the end instead.
        if len(self.iteration_log) < 10:
            return
        recent = [l['ell_t'] for l in self.iteration_log[-10:]]
        if all(e == 0 for e in recent):
            print(f"WARNING iter {len(self.iteration_log)}: "
                  f"last 10 queries all LF. "
                  f"Continuing run -- will evaluate regret at end.")

    def run(self):
        self._sample_initial_points()

        cost_budget = getattr(self.config, 'cost_budget', float('inf'))
        for t in range(self.config.bo_iterations):
            if self.post_init_cost >= cost_budget:
                print(f"iter {t}: cost budget reached "
                      f"(post_init_cost={self.post_init_cost:.1f} >= "
                      f"{cost_budget:.1f}), stopping.")
                break
            self._update_ko_ensemble()
            batch = self._generate_rollout_batch()
            rtg_target = self.schemas.update_and_get_rtg_target(batch)
            btg_target = self.schemas.update_and_get_btg_target(batch)
            self._last_rtg_target = rtg_target
            # FIX (Bug B): previously only set once (`if self.btg_target_base
            # is None`), freezing every subsequent real query's BTG
            # conditioning at whatever near-minimum value iteration 0
            # happened to produce. Now updated every iteration, the same
            # treatment _last_rtg_target already gets above.
            self.btg_target_base = btg_target

            # H6 (freeze_dt_after): train normally through iteration k, then
            # stop updating the DT entirely for the rest of the run. Rollouts
            # are STILL generated above (they supply the RTG/BTG targets) and
            # the KO ensemble still refits -- only the policy weights freeze.
            # Tests whether the DT's continued learning contributes anything,
            # given that H4/H5 showed the proposal is near-independent of the
            # conditioning within a single trained model. None (default) =
            # unchanged behaviour.
            _snap_at = getattr(self.config, 'decision_snapshot_at', None)
            _freeze_after = getattr(self.config, 'freeze_dt_after', None)
            if _freeze_after is not None and t >= _freeze_after:
                L_loc, L_fid, fid_mean, fid_std = self._last_train_stats
            else:
                L_loc, L_fid, fid_mean, fid_std = self._train_dt(batch)
                self._last_train_stats = (L_loc, L_fid, fid_mean, fid_std)
            if _snap_at is not None and t == _snap_at and self._dt_snapshot is None:
                import copy as _copy
                self._dt_snapshot = _copy.deepcopy(self.dt)
                self._dt_snapshot.eval()
            x_t, ell_t = self._propose_next_query()
            self._last_p_pred = self.dt.last_p_pred
            # Fidelity-bottleneck measurement: p_pred_inference was only
            # ever printed to console before -- accumulate it so it can be
            # analyzed from the saved result without scraping logs.
            self.p_pred_inference_per_iter.append(self._last_p_pred)

            # Real-query distance to the benchmark's known optimum location
            # (only when config.known_optimal_x is set -- Hartmann_6D has a
            # published closed-form location; Currin_2D/Borehole_8D don't).
            # Same config field name/dict as dro_runner.py's _KNOWN_OPTIMAL_X
            # (used for SF-DRO's own Track-1 diagnostic).
            known_x = getattr(self.config, 'known_optimal_x', None)
            if known_x is not None:
                xstar_t = torch.as_tensor(known_x, dtype=x_t.dtype, device=x_t.device)
                x_t_norm = (x_t - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
                xstar_norm = (xstar_t - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
                self.query_dist_to_xstar_per_iter.append((x_t_norm - xstar_norm).norm().item())
            known_x2 = getattr(self.config, 'known_secondary_x', None)
            if known_x2 is not None:
                x2_t = torch.as_tensor(known_x2, dtype=x_t.dtype, device=x_t.device)
                x_t_norm2 = (x_t - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
                x2_norm = (x2_t - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
                self.query_dist_to_x2_per_iter.append((x_t_norm2 - x2_norm).norm().item())

            # RTG sensitivity probe (diagnostic only, does not affect the
            # real query above): at the requested iterations, re-run
            # propose_mf on the SAME state at several RTG multipliers and
            # measure how much the predicted location/fidelity moves.
            # Gated behind self._diag_probe_iters (unset/None by default).
            probe_iters = getattr(self, '_diag_probe_iters', None)
            if probe_iters is not None and t in probe_iters:
                multipliers = [0.25, 0.5, 1.0, 2.0, 4.0]
                probe_x, probe_ell = [], []
                for m in multipliers:
                    with torch.no_grad():
                        px, pell = self.dt.propose_mf(
                            self._last_state.float(),
                            self._last_rtg_target * m,
                            self._last_btg_now,
                            timestep=0,
                            use_candidate_scoring=self.use_candidate_scoring,
                        )
                    probe_x.append(px)
                    probe_ell.append(pell)
                probe_x = torch.stack(probe_x)  # [5, d], normalized [0,1]^d
                pairwise = torch.cdist(probe_x, probe_x)
                n = pairwise.shape[0]
                mean_probe_spread = (pairwise.sum() / (n * (n - 1))).item()
                fidelity_varies = len(set(probe_ell)) > 1
                print(
                    f"[RTG-PROBE] iter={t} rtg_multipliers={multipliers} "
                    f"mean_probe_spread={mean_probe_spread:.4f} "
                    f"probe_ell={probe_ell} fidelity_varies={fidelity_varies}",
                    flush=True,
                )

            # Cold-start override: force HF for the first real_hf_warmup
            # real iterations regardless of DT output, so recent_hf_frac
            # (and n_hf/n_lf-derived GP fit) never start from an empty/
            # all-LF history in the first place.
            real_hf_warmup = getattr(self.config, 'real_hf_warmup', 2)
            if t < real_hf_warmup:
                ell_t = 1
                print(f"iter {t}: cold-start HF override "
                      f"(p_pred was {self._last_p_pred:.4f})")

            if ell_t == 1:
                y_t = self.f_hf(x_t.unsqueeze(0)).reshape(-1)[0].item()
                self.data_hf_x.append(x_t.double())
                self.data_hf_y.append(y_t)
            else:
                y_t = self.f_lf(x_t.unsqueeze(0)).reshape(-1)[0].item()
                self.data_lf_x.append(x_t.double())
                self.data_lf_y.append(y_t)
            step_cost = self.c_H if ell_t else self.c_L
            self.cumulative_cost += step_cost
            self.post_init_cost += step_cost
            self.recent_ell_history.append(ell_t)

            print(f"[FIX-DIAG] iter={t} ell_t={ell_t} "
                  f"p_pred_before_override={self._last_p_pred:.4f} "
                  f"recent_hf_frac={self._last_recent_hf_frac:.4f}")

            best_hf = max(self.data_hf_y)
            # Raw scale: f_hf is already the negated (maximization-ready)
            # objective (see benchmarks.py), and config.true_opt is the
            # RAW pre-negation optimum -- matching dro.py's own maximize-mode
            # regret convention (-best_observed - known_optimal_value), not
            # a naive abs-difference on mismatched scales (see the same bug
            # already caught and fixed in mf_baselines.py's regret calc).
            regret = -best_hf - self.config.true_opt

            # INFERENCE REGRET (Takeno et al. 2020's second metric, and the one
            # the MFBO literature reports alongside simple regret): the regret of
            # the model's RECOMMENDATION x_hat = argmax mu_H, not of the best
            # point queried. SR can only fall when a good point is *evaluated*;
            # IR falls as soon as the surrogate *believes* the right thing, so
            # the two separate a method that finds good points from one that
            # merely models well. Evaluated on the fixed y_star_pool reference
            # set so it is comparable across iterations and methods.
            # Takeno's convention: if IR > SR at an iteration, report SR (the
            # recommendation is never worse than the best point actually seen).
            try:
                with torch.no_grad():
                    _mu = self.ko_ensemble[0].hf_posterior(self.y_star_pool)[0].flatten()
                    _xhat = self.y_star_pool[int(_mu.argmax())]
                    _f_xhat = float(self.f_hf(_xhat.unsqueeze(0)).reshape(-1)[0])
                inf_regret = min(-_f_xhat - self.config.true_opt, regret)
            except Exception:
                inf_regret = float('nan')
            neg_rtg_frac = float(np.mean([tr['neg_rtg_frac'] for tr in batch]))

            self.iteration_log.append({
                'iter': t, 'ell_t': ell_t, 'y_t': y_t,
                'x_t': x_t.tolist(),
                'cumulative_cost': self.cumulative_cost,
                'post_init_cost': self.post_init_cost,
                'regret': regret, 'inference_regret': inf_regret,
                'rtg_target': rtg_target,
                'btg_target': btg_target,
                'fid_mean': fid_mean, 'fid_std': fid_std,
                'L_loc': L_loc, 'L_fid': L_fid,
                'neg_rtg_frac': neg_rtg_frac,
            })

            print(f"iter {t:3d} | ell={ell_t} y={y_t:.4f} "
                  f"cost={self.cumulative_cost:.1f} "
                  f"regret={regret:.4f} "
                  f"fid_mean={fid_mean:.3f} "
                  f"btg_now={self._last_btg_now:.4f} "
                  f"p_pred_inference={self.dt.last_p_pred:.4f} "
                  f"L_loc={L_loc:.4f} L_fid={L_fid:.4f}")

            try:
                self._check_lf_pathology()
            except RuntimeError:
                break

        return self._build_result()

    def _build_result(self):
        L = self.iteration_log
        return {
            'hf_regret_curve': [l['regret'] for l in L],
            'inference_regret_curve': [l.get('inference_regret') for l in L],
            # cost_curve is POST-INIT cost (starts near 0, excludes
            # initialization spending) so methods with different
            # initialization sizes are comparable on the same x-axis.
            'cost_curve': [l['post_init_cost'] for l in L],
            'cumulative_cost_curve': [l['cumulative_cost'] for l in L],
            'fidelity_trace': [l['ell_t'] for l in L],
            'x_t_trace': [l['x_t'] for l in L],
            'y_t_trace': [l['y_t'] for l in L],
            'initial_hf_values': self.initial_hf_values,
            'lf_fraction': sum(1 for l in L if l['ell_t'] == 0)
                / max(len(L), 1),
            'rtg_target': [l['rtg_target'] for l in L],
            'btg_target': [l['btg_target'] for l in L],
            'fid_mean_per_iter': [l['fid_mean'] for l in L],
            'fid_std_per_iter': [l['fid_std'] for l in L],
            'L_loc_per_iter': [l['L_loc'] for l in L],
            'L_fid_per_iter': [l['L_fid'] for l in L],
            'neg_rtg_frac_per_iter': [l['neg_rtg_frac'] for l in L],
            # Trajectory-level-luck / DT-policy-layer diagnostics (see
            # _generate_rollout_batch and _train_dt for what each measures).
            # One entry per _generate_rollout_batch/_train_dt call, i.e. per
            # BO iteration that actually ran (same length as L_loc_per_iter).
            'action_reward_corr_per_iter': self.action_reward_corr_per_iter,
            'rtg_frac_between_traj_var_per_iter': self.rtg_frac_between_traj_var_per_iter,
            'rtg_gpbelief_corr_per_iter': self.rtg_gpbelief_corr_per_iter,
            'grad_coherency_per_iter': self.grad_coherency_per_iter,
            # Real-query distance to the known optimum location, only
            # non-empty when config.known_optimal_x is set.
            'query_dist_to_xstar_per_iter': self.query_dist_to_xstar_per_iter,
            'query_dist_to_x2_per_iter': self.query_dist_to_x2_per_iter,
            'p_pred_inference_per_iter': self.p_pred_inference_per_iter,
            'decision_divergence_log': self.decision_divergence_log,
            # Rollout-exploration diagnostic, only non-empty when a
            # diagnostic script set self._diag_xstar before .run() (see
            # __init__ and the [ROLLOUT-DIAG] block in _generate_rollout_batch).
            'diag_frac_rollout_near_xstar_per_iter': self.diag_frac_rollout_near_xstar_per_iter,
            # GP-refinement ablation diagnostic, only non-empty when
            # config.use_gp_refinement was True (see _refine_proposal).
            'gp_refinement_log': self.gp_refinement_log,
        }
