"""
Experiment C: MF-MES under an ADDITIVE fidelity model (Song et al. 2019 /
Takeno et al. 2020), plus a single-fidelity MES reference.

The point of this file is a controlled contrast against KO-MES
(GreedyMFMESOptimizer). Every non-surrogate component -- initial design,
candidate pool, Thompson-sampled y*, cost-normalized argmax, cost
accounting, regret convention -- is inherited unchanged from GreedyMFBase,
and the acquisition itself is the model-agnostic Takeno Lemma 3.1
implementation in mes_common.py. The ONLY thing that varies is the
surrogate, so a gap in the results is attributable to the surrogate.

Two additive variants are provided, because the specification this was
built from describes two different things and they answer different
questions:

  RHO1 ("f_H = f_L + delta")
      The KO model with rho pinned to 1 instead of fitted. This is the
      tightest possible ablation: the model class, the fitting procedure,
      the alternating-optimization schedule and the acquisition are all
      literally the same code as KO-MES, and the single difference is
      whether the global fidelity correlation is learned. Any KO-MES
      advantage that survives this contrast is attributable to the fitted
      rho and nothing else.

  SONG ("f_L = f_H + e")
      The two-stage additive model as usually written down: a target GP on
      the HF observations and an error GP on the LF residuals against it.
      Same structure as MFMIGreedyOptimizer's model in mf_baselines.py.
      Note this variant's HF posterior is fitted on HF data ALONE -- LF
      observations never inform it (they only shape the error GP) -- which
      is a real handicap relative to KO, where LF data flows into the HF
      posterior through rho*mu_L. That is a property of the model as
      specified, not an implementation shortcut, but it does mean RHO1 is
      the fairer of the two contrasts and SONG is closer to how the
      additive baseline is usually reported.

On the cross-covariance each variant feeds to mes_common.mes_lf:
  RHO1: f_H = f_L + delta, f_L indep delta
        => Cov(f_L, f_H) = Var(f_L); Var(f_H) = Var(f_L) + Var(delta)
  SONG: f_L = f_H + e, f_H indep e
        => Cov(f_L, f_H) = Var(f_H); Var(f_L) = Var(f_H) + Var(e)
Both are then handled by the same general bivariate conditional, so neither
needs a bespoke information-gain formula.
"""
import numpy as np
import torch

from gumbel_thompson import thompson_sample_y_star
from src.baselines.greedy_mf import GreedyMFBase, cost_normalized_argmax
from src.baselines.mes_common import mes_hf, mes_lf
from src.models.ko_gp import KennedyOHaganGP

DEFAULT_DTYPE = torch.float64


def _build_ko_style_gp(d, X, Y, bounds):
    """
    Build one SingleTaskGP through KennedyOHaganGP's own _build_gp, so the
    additive model's component GPs get bit-for-bit the same construction as
    the KO model's: same RBF+ARD kernel with the same Interval lengthscale
    constraint and geometric-mean initialization, same noise floor, same
    Normalize+Standardize transforms, same Adam schedule. Using a
    differently-configured GP here would reintroduce exactly the confound
    this file exists to avoid.
    """
    helper = KennedyOHaganGP(d=d, dkl_threshold=float('inf'))
    helper.bounds = bounds.to(dtype=DEFAULT_DTYPE)
    return helper._build_gp(X, Y, use_dkl=False)


class _AdditiveSongGP:
    """
    Two-stage additive surrogate: f_L = f_H + e, with f_H independent of e.

        gp_H : fitted on (X_hf, Y_hf)                    -- target GP
        gp_e : fitted on (X_lf, Y_lf - mu_H(X_lf))       -- error GP

    Exposes the same (mean, variance) accessors the acquisition needs, plus
    the cross-covariance Cov(f_L, f_H) = Var(f_H).
    """

    def __init__(self, d, bounds):
        self.d = d
        self.bounds = bounds
        self.gp_H = None
        self.gp_e = None

    def fit(self, X_lf, Y_lf, X_hf, Y_hf):
        self.gp_H = _build_ko_style_gp(self.d, X_hf, Y_hf, self.bounds)
        if X_lf.shape[0] > 0:
            with torch.no_grad():
                mu_H_at_lf = self.gp_H.posterior(X_lf).mean.reshape(-1)
            self.gp_e = _build_ko_style_gp(
                self.d, X_lf, Y_lf - mu_H_at_lf, self.bounds
            )

    def _post(self, gp, X):
        with torch.no_grad():
            p = gp.posterior(X)
            return (p.mean.reshape(-1).numpy(),
                    p.variance.clamp_min(1e-12).reshape(-1).numpy())

    def moments(self, X):
        """Returns (mu_L, var_L, mu_H, var_H, cov_LH), each np.ndarray [N]."""
        mu_H, var_H = self._post(self.gp_H, X)
        if self.gp_e is None:
            mu_e = np.zeros_like(mu_H)
            var_e = np.ones_like(var_H)
        else:
            mu_e, var_e = self._post(self.gp_e, X)
        return mu_H + mu_e, var_H + var_e, mu_H, var_H, var_H

    def hf_model(self):
        """The GP whose joint posterior supplies Thompson samples of y*_H."""
        return self.gp_H


class AdditiveMESOptimizer(GreedyMFBase):
    """
    Additive-model MF-MES. variant: 'rho1' or 'song' (see module docstring).
    Everything outside the surrogate is inherited from GreedyMFBase and is
    therefore identical to KO-MES.
    """

    def __init__(self, benchmark, n_initial_hf=5, n_initial_lf=5, seed=0,
                 cost_budget=None, use_sequential_init=False,
                 n_candidates=200, variant='rho1'):
        super().__init__(benchmark, n_initial_hf=n_initial_hf,
                         n_initial_lf=n_initial_lf, seed=seed,
                         cost_budget=cost_budget,
                         use_sequential_init=use_sequential_init,
                         n_candidates=n_candidates)
        if variant not in ('rho1', 'song'):
            raise ValueError(f"unknown variant {variant!r}, expected 'rho1' or 'song'")
        self.variant = variant
        if variant == 'rho1':
            # rho pinned to 1.0 => f_H = f_L + delta, i.e. the additive model,
            # sharing KO-MES's entire fitting path.
            self.model = KennedyOHaganGP(d=self.d, rho_fixed=1.0,
                                          dkl_threshold=float('inf'))
        else:
            self.model = _AdditiveSongGP(self.d, benchmark.bounds)

    def _training_tensors(self):
        X_hf = torch.stack(self.data_hf_x)
        Y_hf = torch.tensor(self.data_hf_y, dtype=DEFAULT_DTYPE)
        if self.data_lf_x:
            X_lf = torch.stack(self.data_lf_x)
            Y_lf = torch.tensor(self.data_lf_y, dtype=DEFAULT_DTYPE)
        else:
            X_lf, Y_lf = X_hf[:0], Y_hf[:0]
        return X_lf, Y_lf, X_hf, Y_hf

    def _update_model(self):
        X_lf, Y_lf, X_hf, Y_hf = self._training_tensors()
        if self.variant == 'rho1':
            self.model.fit(X_lf, Y_lf, X_hf, Y_hf, self.bounds)
        else:
            self.model.fit(X_lf, Y_lf, X_hf, Y_hf)

    def _moments_and_hf_model(self, X_cand):
        if self.variant == 'rho1':
            ko = self.model
            with torch.no_grad():
                mu_L, var_L = ko.lf_posterior(X_cand)
                mu_H, var_H = ko.hf_posterior(X_cand)
            mu_L, var_L = mu_L.numpy(), var_L.numpy()
            mu_H, var_H = mu_H.numpy(), var_H.numpy()
            # rho == 1 exactly here, so Cov(f_L, f_H) = rho*Var(f_L) = Var(f_L).
            cov_LH = var_L
            from src.policy.mf_dro import _build_hf_proxy_model
            return mu_L, var_L, mu_H, var_H, cov_LH, _build_hf_proxy_model(ko)
        mu_L, var_L, mu_H, var_H, cov_LH = self.model.moments(X_cand)
        return mu_L, var_L, mu_H, var_H, cov_LH, self.model.hf_model()

    def _propose_greedy(self, X_cand):
        mu_L, var_L, mu_H, var_H, cov_LH, hf_model = \
            self._moments_and_hf_model(X_cand)
        y_star = thompson_sample_y_star(hf_model, X_cand, K=10)

        acq_hf = mes_hf(mu_H, np.sqrt(var_H), y_star)
        acq_lf = mes_lf(mu_L, np.sqrt(var_L), mu_H, np.sqrt(var_H), cov_LH,
                        y_star, n_quad=32)
        return cost_normalized_argmax(acq_lf, acq_hf, self.c_L, self.c_H, X_cand)

    def final_rho(self):
        return 1.0 if self.variant == 'rho1' else None


class SFMESOptimizer(GreedyMFBase):
    """
    Single-fidelity MES reference: one RBF GP on HF observations only,
    greedy MES acquisition, no LF queries and no fidelity switching. Shares
    GreedyMFBase's loop and regret convention so it lands on the same
    post-init cost axis as every multi-fidelity method; `uses_lf = False`
    suppresses the LF initial design, so its initialization spends only
    n_initial_hf * c_H.

    Answers "does multi-fidelity help at all here?" without the confound of
    also changing the acquisition -- the HF branch is the same mes_hf the
    multi-fidelity methods use.
    """

    uses_lf = False

    def __init__(self, benchmark, n_initial_hf=5, n_initial_lf=0, seed=0,
                 cost_budget=None, use_sequential_init=False,
                 n_candidates=200):
        super().__init__(benchmark, n_initial_hf=n_initial_hf,
                         n_initial_lf=0, seed=seed, cost_budget=cost_budget,
                         use_sequential_init=use_sequential_init,
                         n_candidates=n_candidates)
        self.gp = None

    def _update_model(self):
        X_hf = torch.stack(self.data_hf_x)
        Y_hf = torch.tensor(self.data_hf_y, dtype=DEFAULT_DTYPE)
        self.gp = _build_ko_style_gp(self.d, X_hf, Y_hf, self.bounds)

    def _propose_greedy(self, X_cand):
        with torch.no_grad():
            p = self.gp.posterior(X_cand)
            mu = p.mean.reshape(-1).numpy()
            sigma = p.variance.clamp_min(1e-12).reshape(-1).sqrt().numpy()
        y_star = thompson_sample_y_star(self.gp, X_cand, K=10)
        acq = mes_hf(mu, sigma, y_star)
        # HF-only: never propose fidelity 0.
        return X_cand[int(np.argmax(acq))], 1
