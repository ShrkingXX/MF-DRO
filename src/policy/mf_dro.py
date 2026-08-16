"""
Multi-fidelity DRO state extraction and joint (location, fidelity) MES
acquisition, built on top of the Kennedy-O'Hagan two-fidelity GP
(src/models/ko_gp.py).

Deliberately independent of src/policy/dro.py's `_extract_state` -- the MF
state vector has a completely different structure (per-ensemble-member HF/LF
sigma summaries instead of raw GP kernel hyperparameters, plus fidelity-cost
features with no SF-DRO analogue). `_extract_state` was read for style
reference only and is never called from this module.
"""
import math
from types import SimpleNamespace

import torch
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.stats import norm as scipy_norm

from gumbel_thompson import thompson_sample_y_star, fit_gumbel_to_samples
from mes_reward import compute_mes_reward
from src.models.ko_gp import KennedyOHaganGP
from src.model.decisionTransformer import DecisionTransformer

EULER_GAMMA = 0.5772156649015329


# ════════════════════════════════════
# PART A: State Dimension Helper
# ════════════════════════════════════

def _get_mf_state_dim(d, M=10):
    """
    MF-DRO state dimension.
    Breakdown:
        M slots:  mean HF sigma per ensemble member
        M slots:  mean LF sigma per ensemble member
        1 slot:   best_value_HF
        1 slot:   step_norm
        d slots:  best_position_HF
        1 slot:   mean_rho across ensemble
        1 slot:   n_hf / (n_hf + n_lf)
        1 slot:   c_L / c_H
    Total: 2*M + d + 5
    """
    return 2 * M + d + 5


# ════════════════════════════════════
# PART B: MF State Extraction
# ════════════════════════════════════

def _extract_mf_state(data_hf_x, data_hf_y,
                       data_lf_x, data_lf_y,
                       ko_ensemble, roi_candidates,
                       n_real_iter, T_real,
                       c_L, c_H, device, dtype):
    """
    Build MF state vector. Does NOT call _extract_state from dro.py -- this
    is an entirely independent function.

    data_hf_x: list of [d] tensors (may be empty)
    data_hf_y: list of scalar floats (may be empty)
    data_lf_x: list of [d] tensors (may be empty)
    data_lf_y: list of scalar floats (may be empty)
    ko_ensemble: list of KennedyOHaganGP (length M)
    roi_candidates: [N_roi, d] tensor or None
    Returns: Tensor [2*M + d + 5]
    """
    M = len(ko_ensemble)
    d = ko_ensemble[0].d
    state_list = []

    # Slots 0..M-1: mean HF sigma per ensemble member
    # Slots M..2M-1: mean LF sigma per ensemble member
    if roi_candidates is not None:
        with torch.no_grad():
            for ko in ko_ensemble:
                _, var_hf = ko.hf_posterior(roi_candidates)
                state_list.append(var_hf.clamp_min(0).sqrt()
                                   .mean().item())
            for ko in ko_ensemble:
                _, var_lf = ko.lf_posterior(roi_candidates)
                state_list.append(var_lf.clamp_min(0).sqrt()
                                   .mean().item())
    else:
        state_list.extend([0.0] * (2 * M))

    # Slot 2M: best_value_HF
    best_val = max(data_hf_y) if data_hf_y else 0.0
    state_list.append(float(best_val))

    # Slot 2M+1: step_norm
    state_list.append(float(n_real_iter) / max(float(T_real), 1.0))

    # Slots 2M+2 .. 2M+1+d: best_position_HF
    if data_hf_y:
        best_idx = int(np.argmax(data_hf_y))
        best_pos = data_hf_x[best_idx].tolist()
    else:
        best_pos = [0.0] * d
    state_list.extend(best_pos)

    # Slot 2M+2+d: mean_rho across ensemble
    rhos = [torch.sigmoid(ko.log_rho).item() for ko in ko_ensemble]
    state_list.append(float(np.mean(rhos)))

    # Slot 2M+3+d: n_hf / (n_hf + n_lf)
    n_hf = len(data_hf_y) if data_hf_y else 0
    n_lf = len(data_lf_y) if data_lf_y else 0
    total = n_hf + n_lf
    state_list.append(n_hf / total if total > 0 else 0.5)

    # Slot 2M+4+d: c_L / c_H
    state_list.append(float(c_L) / float(c_H))

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


def compute_joint_mf_mes(ko_model, roi_candidates, c_H, c_L, K=10):
    """
    Joint MF-MES acquisition (Takeno et al. 2020, arXiv:1901.08275,
    Algorithm 1). Selects (x_tau, ell_tau) = argmax_{x,ell}
    InfoGain(x,ell)/c(ell) using the EXACT MF-MES formula for LF (via
    _lf_mes_info_gain), not the variance-ratio approximation this function
    used previously.

    HF branch: standard MES via compute_mes_reward (unchanged).
    LF branch: _lf_mes_info_gain, 1D Gauss-Hermite quadrature over the LF
    posterior, using SHARED Thompson-sampled y*_H draws (same K samples
    feed both branches, avoiding a second independent Monte Carlo estimate
    of the same global-optimum distribution).

    Returns:
        x_tau:   [d] tensor
        ell_tau: int 0=L, 1=H
        scores:  [N_roi, 2] tensor (col0=LF, col1=HF)
    """
    N = roi_candidates.shape[0]

    # Shared HF y* samples (used for both HF and LF branches)
    hf_proxy = _build_hf_proxy_model(ko_model)
    y_star_arr = thompson_sample_y_star(hf_proxy, roi_candidates, K=K)

    # HF MES at all candidates (existing implementation, unchanged)
    mes_hf_list = []
    for i in range(N):
        x_i = roi_candidates[i].unsqueeze(0)
        s = compute_mes_reward(x_i, hf_proxy, roi_candidates)
        mes_hf_list.append(s.item() if hasattr(s, 'item') else float(s))
    mes_hf = torch.tensor(mes_hf_list, dtype=roi_candidates.dtype)

    # LF MES at all candidates (Takeno 2020 Lemma 3.1, numerically corrected)
    mes_lf_list = []
    for i in range(N):
        s = _lf_mes_info_gain(roi_candidates[i], ko_model, y_star_arr, n_quad=32)
        mes_lf_list.append(s)
    mes_lf = torch.tensor(mes_lf_list, dtype=roi_candidates.dtype)

    # Cost-normalized scores [N, 2]
    scores = torch.stack([mes_lf / c_L, mes_hf / c_H], dim=1)

    flat_best = scores.reshape(-1).argmax()
    cand_idx = (flat_best // 2).item()
    fid_idx = (flat_best % 2).item()  # 0=L, 1=H

    return roi_candidates[cand_idx], fid_idx, scores


def _compute_mf_roi_candidates(ko_model, bounds, n_samples=500):
    """
    UCB >= max(LCB) filter on HF posterior. Mirrors SF-DRO ROI logic
    (dro.py's _optimize_acquisition).

    bounds: [2, d] -- BoTorch convention (row 0 = lower, row 1 = upper),
    matching every other bounds consumer in this codebase (ko.fit()'s
    Normalize transform, dro.py's `domain_min, domain_max = self.bounds[0],
    self.bounds[1]`). NOTE: indexes bounds[0]/bounds[1] (row-wise), not
    bounds[:,0]/bounds[:,1] (column-wise) -- the latter, tried against a
    [2,d]-shaped bounds tensor, collapses every sampled candidate to a
    single degenerate point (verified: bounds[:,1]-bounds[:,0] is the zero
    vector whenever bounds is [2,d] BoTorch-convention, since both columns
    of such a tensor share the same per-dimension value pattern only
    coincidentally for symmetric domains, but the *range* comes out zero in
    general -- confirmed directly for Currin_2D's own bounds).
    Returns: [N_roi, d] filtered candidates (min 10)
    """
    d = ko_model.d
    X_cand = (bounds[0]
              + (bounds[1] - bounds[0])
              * torch.rand(n_samples, d,
                           device=ko_model.device,
                           dtype=ko_model.dtype))
    kappa = 2.0
    with torch.no_grad():
        mu, var = ko_model.hf_posterior(X_cand)
    sigma = var.clamp_min(0).sqrt()
    ucb = mu + kappa * sigma
    lcb = mu - kappa * sigma
    max_lcb = lcb.max()
    mask = ucb >= max_lcb
    roi = X_cand[mask]
    return roi if roi.shape[0] >= 10 else X_cand


def simulate_mf_trajectory(ko_model, real_data_hf, real_data_lf,
                            rollout_length, c_H, c_L, bounds,
                            n_real_iter, T_real,
                            ko_ensemble_full,
                            K_rtg=100,
                            device='cpu', dtype=torch.float64,
                            minimum_hf_fraction=0.25):
    """
    One MF rollout of fixed length.

    real_data_hf: (list_of_x, list_of_y) -- actual real HF data
    real_data_lf: (list_of_x, list_of_y) -- actual real LF data
    ko_ensemble_full: M KO models for state extraction
    K_rtg: Thompson samples for RTG Gumbel estimation

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

    Returns dict:
        states:       [T, state_dim]
        actions_x:    [T, d]
        actions_ell:  [T] int64, 0=L 1=H
        rtg:          [T] float64  H(y*_H|D_tau) AFTER conditioning
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

    # roi_candidates ONCE before the loop
    roi_candidates = _compute_mf_roi_candidates(ko_model, bounds)

    states = []
    actions_x = []
    actions_ell = []
    rtg_values = []
    costs = []

    current_ko = ko_model

    # Local sim data (starts from real data, grows with fantasy obs)
    sim_hf_x = list(real_hf_x_list)
    sim_hf_y = list(real_hf_y_list)
    sim_lf_x = list(real_lf_x_list)
    sim_lf_y = list(real_lf_y_list)

    for tau in range(rollout_length):

        # 1. State BEFORE conditioning
        s_tau = _extract_mf_state(
            sim_hf_x, sim_hf_y, sim_lf_x, sim_lf_y,
            ko_ensemble_full, roi_candidates,
            n_real_iter, T_real,
            c_L, c_H, device, dtype
        )

        # 2. Joint MES: (x_tau, ell_tau)
        x_tau, ell_tau, _ = compute_joint_mf_mes(
            current_ko, roi_candidates, c_H, c_L
        )

        # 2b. Minimum-HF-fraction override (training-data-generation only --
        # see minimum_hf_fraction's docstring above). Must happen before
        # Steps 3/4/6 below, which all consume ell_tau (fantasy sampling,
        # KO conditioning, and cost/action bookkeeping must all reflect
        # whichever fidelity is ACTUALLY simulated this step).
        hf_steps_so_far = sum(1 for e in actions_ell if e == 1)
        if (tau > 0 and
                hf_steps_so_far / tau < minimum_hf_fraction and
                ell_tau == 0):
            ell_tau = 1  # forced HF to ensure training diversity

        # 3. Sample fantasy observation
        y_tau = current_ko.sample_fantasy(x_tau, 'LH'[ell_tau])

        # 4. Condition (returns new object, does not mutate)
        current_ko = current_ko.make_fantasy_ko(
            x_tau.unsqueeze(0),
            torch.tensor([y_tau], device=device, dtype=dtype),
            'LH'[ell_tau]
        )

        # 5. RTG AFTER conditioning
        hf_proxy = _build_hf_proxy_model(current_ko)
        y_star_arr = thompson_sample_y_star(
            hf_proxy, roi_candidates, K=K_rtg
        )  # np.ndarray [K_rtg]
        _, b_tau = fit_gumbel_to_samples(y_star_arr)
        rtg_tau = math.log(max(b_tau, 1e-12)) + EULER_GAMMA + 1

        # 6. Record
        states.append(s_tau)
        actions_x.append(x_tau.detach())
        actions_ell.append(ell_tau)
        rtg_values.append(rtg_tau)
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
    rtg_t = torch.tensor(rtg_values, dtype=dtype)

    return {
        'states': torch.stack(states),
        'actions_x': torch.stack(actions_x),
        'actions_ell': torch.tensor(actions_ell, dtype=torch.long),
        'rtg': rtg_t,
        'btg': btg,
        'costs': cost_t,
        'total_cost': cost_t.sum().item(),
        'lf_fraction': float((torch.tensor(actions_ell) == 0)
                              .float().mean()),
        'neg_rtg_frac': float((rtg_t < 0).float().mean())
    }


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

    BTG (ceilinged dynamic, minimize): the symmetric analog under
    maximize<->minimize, floor<->ceiling, max<->min: target = min(batch_min,
    (1/alpha_btg) * running_min_btg). This is a CEILING: it prevents the
    target from rising too far above the best-ever (lowest) achieved cost
    even when this batch's own best is unusually expensive (e.g. a round of
    rollouts that happened to lean HF-heavy), by never letting it exceed
    1/alpha_btg (>1) times the all-time-best (lowest) cost -- keeping
    inference-time BTG conditioning within a range the DT has plausibly
    seen during training, rather than an inflated, never-demonstrated
    budget.
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
        RTG[0] = H(y*_H | D_0) at rollout start.
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
        Ceilinged dynamic schema (symmetric to RTG floor):
            target = min(batch_min_btg, (1/alpha) * running_min_btg)

        The (1/alpha) ceiling prevents the BTG target from rising too far
        above the minimum (best) cost seen in training -- avoids OOD
        conditioning on an inflated, never-demonstrated budget.

        rollout_batch: list of trajectory dicts
        Returns: btg_target (scalar float)
        """
        btg0_list = [traj['btg'][0].item() for traj in rollout_batch]
        batch_min = min(btg0_list)
        self.running_min_btg = min(self.running_min_btg, batch_min)
        return min(batch_min,
                    (1.0 / self.alpha_btg) * self.running_min_btg)

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

        self.ko_ensemble = [
            KennedyOHaganGP(d=self.d)
            for _ in range(config.M)
        ]

        state_dim = _get_mf_state_dim(self.d, config.M)
        dt_cfg = SimpleNamespace(
            hidden_size=config.dt_hidden,
            num_layers=config.dt_layers,
            num_heads=config.dt_heads,
            max_seq_length=config.max_seq_length,
            dropout=0.1,
        )
        self.dt = DecisionTransformer(
            dt_cfg, input_dim=state_dim, action_dim=self.d, use_mf=True
        )
        self.dt = self.dt.float()
        self.dt_optimizer = torch.optim.Adam(
            self.dt.parameters(), lr=config.dt_lr
        )
        self.schemas = MFTargetSchemas(
            alpha_rtg=config.alpha_rtg, alpha_btg=config.alpha_btg,
            c_L=config.c_L, c_H=config.c_H
        )

        self.data_hf_x = []
        self.data_hf_y = []
        self.data_lf_x = []
        self.data_lf_y = []
        self.cumulative_cost = 0.0
        self.btg_target_base = None
        self._last_rtg_target = 0.0
        self.iteration_log = []

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
        from scipy.stats.qmc import LatinHypercube
        torch.manual_seed(self.config.seed)

        def _lhs_points(seed_offset):
            sampler = LatinHypercube(d=self.d, seed=self.config.seed + seed_offset)
            unit_X = torch.tensor(
                sampler.random(n=self.config.initial_points),
                dtype=torch.float64
            )
            # Rescale from the sampler's [0,1]^d unit hypercube to the actual
            # domain -- happens to be a no-op for Currin_2D/Hartmann_6D (both
            # already [0,1]^d) but keeps this correct for any other domain.
            return self.bounds[0] + (self.bounds[1] - self.bounds[0]) * unit_X

        X_hf_init = _lhs_points(seed_offset=0)
        for x in X_hf_init:
            y = self.f_hf(x.unsqueeze(0)).reshape(-1)[0].item()
            self.data_hf_x.append(x)
            self.data_hf_y.append(y)
            self.cumulative_cost += self.c_H

        # Distinct seed offset so LF initial points aren't identical to HF's.
        X_lf_init = _lhs_points(seed_offset=1)
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

    def _generate_rollout_batch(self):
        batch = []
        for ko in self.ko_ensemble:
            for _ in range(self.config.rollouts_per_model):
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
                    minimum_hf_fraction=getattr(self.config, 'minimum_hf_fraction', 0.25)
                )
                batch.append(traj)
        return batch

    def _train_dt(self, batch):
        self.dt.train()
        T = self.config.rollout_length
        B = len(batch)
        states = torch.stack([t['states'] for t in batch]).float()
        actions_x = torch.stack([t['actions_x'] for t in batch]).float()
        actions_ell = torch.stack([t['actions_ell'] for t in batch])
        rtg = torch.stack([t['rtg'] for t in batch]).float()
        btg = torch.stack([t['btg'] for t in batch]).float()
        timesteps = torch.arange(T).unsqueeze(0).repeat(B, 1)

        L_loc_final = L_fid_final = 0.0
        for epoch in range(self.config.num_epochs):
            self.dt_optimizer.zero_grad()
            loss, L_loc, L_fid, _, p_pred = self.dt.forward_mf(
                states, actions_x, actions_ell,
                rtg, btg, timesteps
            )
            loss.backward()
            self.dt_optimizer.step()
            L_loc_final = L_loc.item()
            L_fid_final = L_fid.item()

        fid_mean = p_pred.detach().mean().item()
        fid_std = p_pred.detach().std().item()
        return L_loc_final, L_fid_final, fid_mean, fid_std

    def _propose_next_query(self):
        roi_cands = _compute_mf_roi_candidates(
            self.ko_ensemble[0], self.bounds
        )
        state = _extract_mf_state(
            self.data_hf_x, self.data_hf_y,
            self.data_lf_x, self.data_lf_y,
            self.ko_ensemble, roi_cands,
            len(self.data_hf_y), self.config.bo_iterations,
            self.c_L, self.c_H,
            device='cpu', dtype=torch.float64
        )
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
        # btg[0] == btg_target_base by construction.
        btg_now = self.btg_target_base
        self._last_btg_now = btg_now  # exposed for diagnostics/verification
        rtg_tgt = self._last_rtg_target
        x_t, ell_t = self.dt.propose_mf(
            state.float(), rtg_tgt, btg_now,
            timestep=len(self.data_hf_y)
        )
        return x_t, ell_t

    def _check_lf_pathology(self):
        if len(self.iteration_log) < 10:
            return
        recent = [l['ell_t'] for l in self.iteration_log[-10:]]
        if all(e == 0 for e in recent):
            print("WARNING: last 10 iterations all LF. "
                  "Pathological all-LF run detected. "
                  "Stopping -- report to user.")
            raise RuntimeError("all_lf_pathology")

    def run(self):
        self._sample_initial_points()

        for t in range(self.config.bo_iterations):
            self._update_ko_ensemble()
            batch = self._generate_rollout_batch()
            rtg_target = self.schemas.update_and_get_rtg_target(batch)
            btg_target = self.schemas.update_and_get_btg_target(batch)
            self._last_rtg_target = rtg_target

            if self.btg_target_base is None:
                self.btg_target_base = btg_target

            L_loc, L_fid, fid_mean, fid_std = self._train_dt(batch)
            x_t, ell_t = self._propose_next_query()

            if ell_t == 1:
                y_t = self.f_hf(x_t.unsqueeze(0)).reshape(-1)[0].item()
                self.data_hf_x.append(x_t.double())
                self.data_hf_y.append(y_t)
            else:
                y_t = self.f_lf(x_t.unsqueeze(0)).reshape(-1)[0].item()
                self.data_lf_x.append(x_t.double())
                self.data_lf_y.append(y_t)
            self.cumulative_cost += (self.c_H if ell_t else self.c_L)

            best_hf = max(self.data_hf_y)
            # Raw scale: f_hf is already the negated (maximization-ready)
            # objective (see benchmarks.py), and config.true_opt is the
            # RAW pre-negation optimum -- matching dro.py's own maximize-mode
            # regret convention (-best_observed - known_optimal_value), not
            # a naive abs-difference on mismatched scales (see the same bug
            # already caught and fixed in mf_baselines.py's regret calc).
            regret = -best_hf - self.config.true_opt
            neg_rtg_frac = float(np.mean([tr['neg_rtg_frac'] for tr in batch]))

            self.iteration_log.append({
                'iter': t, 'ell_t': ell_t, 'y_t': y_t,
                'cumulative_cost': self.cumulative_cost,
                'regret': regret, 'rtg_target': rtg_target,
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
            'cost_curve': [l['cumulative_cost'] for l in L],
            'fidelity_trace': [l['ell_t'] for l in L],
            'lf_fraction': sum(1 for l in L if l['ell_t'] == 0)
                / max(len(L), 1),
            'rtg_target': [l['rtg_target'] for l in L],
            'btg_target': [l['btg_target'] for l in L],
            'fid_mean_per_iter': [l['fid_mean'] for l in L],
            'fid_std_per_iter': [l['fid_std'] for l in L],
            'L_loc_per_iter': [l['L_loc'] for l in L],
            'L_fid_per_iter': [l['L_fid'] for l in L],
            'neg_rtg_frac_per_iter': [l['neg_rtg_frac'] for l in L],
        }
