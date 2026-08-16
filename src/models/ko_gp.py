"""
Kennedy-O'Hagan (2000) autoregressive two-fidelity GP:

    f_H(x) = rho * f_L(x) + delta(x)
    f_L   ~ GP(mu_L, k_L)      LF surrogate, fit independently on LF data
    delta ~ GP(0, k_delta)     HF residual, fit on Y_hf - rho*mu_L(X_hf)

GP construction and the fantasy/conditioning logic below deliberately mirror
src/policy/dro.py's `_construct_gp_model`/`_make_fantasy_model` exactly (same
transform choice, same "rebuild from scratch with frozen hyperparameters
instead of condition_on_observations" pattern -- see `_make_fantasy_model`'s
docstring in dro.py for why condition_on_observations/get_fantasy_model are
broken for Normalize+Standardize-transformed SingleTaskGPs).
"""
import math
import warnings

import torch
import torch.nn as nn
import gpytorch

from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.exceptions import InputDataWarning
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.mlls import ExactMarginalLogLikelihood


class KennedyOHaganGP:
    """
    d: input dimension
    All GPs use: Normalize(d) + Standardize(m=1) transforms, RBF kernel with
    ARD=True, ScaleKernel wrapper, GaussianLikelihood with noise >= noise_lb.
    Lengthscale is constrained to [0.05*sqrt(d), 2*sqrt(d)] (via an Interval
    constraint on the RBF kernel, not just an initial value) and initialized
    at that range's geometric mean.
    """

    def __init__(self, d, rho_init=0.8, lr=0.1, train_iter=50,
                 noise_lb=1e-4, device='cpu', dtype=torch.float64):
        self.d = d
        self.log_rho = nn.Parameter(
            torch.tensor(math.log(rho_init / (1.0 - rho_init)))
        )
        self.gp_lf = None
        self.gp_delta = None
        self.device = device
        self.dtype = dtype
        # Store raw training data for make_fantasy_ko
        self.train_x_lf = None     # [N_lf, d] raw scale
        self.train_y_lf = None     # [N_lf] raw scale
        self.train_x_hf = None     # [N_hf, d] raw scale
        self.train_y_hf = None     # [N_hf] raw scale
        self.train_y_delta = None  # [N_hf] residuals, raw scale
        self.bounds = None         # [d, 2]... actually [2, d], see fit()
        self.lr = lr
        self.train_iter = train_iter
        self.noise_lb = noise_lb

    @property
    def rho(self):
        return torch.sigmoid(self.log_rho)

    def _lengthscale_bounds(self):
        # Deterministic in d -- recomputed (not stored) so _build_gp and
        # _rebuild_frozen_gp always construct an IDENTICAL Interval constraint
        # object for a given self.d, which matters for _rebuild_frozen_gp:
        # load_state_dict copies the raw (pre-constraint-transform) parameter
        # value, so decoding it through a differently-bounded Interval would
        # silently produce a different actual lengthscale than the one being
        # "frozen" from the source model.
        return 0.05 * math.sqrt(self.d), 2.0 * math.sqrt(self.d)

    def _build_gp(self, X_raw, Y_raw):
        """
        Build a fresh SingleTaskGP on (X_raw, Y_raw), MLL-fit via Adam for
        train_iter steps. Mirrors dro.py's _construct_gp_model exactly (same
        likelihood/kernel/transform choices), except the kernel is always RBF
        with ARD and a bounded lengthscale, per this class's spec.
        X_raw: [N, d], Y_raw: [N]. Returns model in eval() mode.
        """
        low, high = self._lengthscale_bounds()
        likelihood = GaussianLikelihood(noise_constraint=GreaterThan(self.noise_lb))
        base_kernel = RBFKernel(ard_num_dims=self.d, lengthscale_constraint=Interval(low, high))
        base_kernel.initialize(lengthscale=math.sqrt(low * high))
        covar_module = ScaleKernel(base_kernel)

        train_x = X_raw.to(device=self.device, dtype=self.dtype)
        train_y = Y_raw.to(device=self.device, dtype=self.dtype).reshape(-1, 1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=InputDataWarning)
            model = SingleTaskGP(
                train_X=train_x,
                train_Y=train_y,
                likelihood=likelihood,
                covar_module=covar_module,
                input_transform=Normalize(d=self.d, bounds=self.bounds),
                outcome_transform=Standardize(m=1),
            )
            model = model.to(device=self.device, dtype=self.dtype)

        model.train()
        model.likelihood.train()
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        gp_optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        for _ in range(self.train_iter):
            gp_optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, model.train_targets)
            loss.backward()
            gp_optimizer.step()

        model.eval()
        model.likelihood.eval()
        return model

    def _rebuild_frozen_gp(self, old_model, X_aug_raw, Y_aug_raw):
        """
        Rebuild a fresh SingleTaskGP on (X_aug_raw, Y_aug_raw), reusing
        old_model's frozen kernel/likelihood hyperparameters (no MLL
        refitting) -- the exact same pattern as dro.py's
        _make_fantasy_model, generalized from that method's RBF-or-Matern
        branch to this class's always-RBF-with-ARD kernel.
        """
        low, high = self._lengthscale_bounds()
        X_aug = X_aug_raw.to(device=self.device, dtype=self.dtype)
        Y_aug = Y_aug_raw.to(device=self.device, dtype=self.dtype).reshape(-1, 1)

        new_likelihood = GaussianLikelihood(noise_constraint=GreaterThan(self.noise_lb))
        new_likelihood.noise = old_model.likelihood.noise.detach().clone()

        new_covar_module = ScaleKernel(
            RBFKernel(ard_num_dims=self.d, lengthscale_constraint=Interval(low, high))
        )
        new_covar_module.load_state_dict(old_model.covar_module.state_dict())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=InputDataWarning)
            new_model = SingleTaskGP(
                train_X=X_aug,
                train_Y=Y_aug,
                likelihood=new_likelihood,
                covar_module=new_covar_module,
                input_transform=Normalize(d=self.d, bounds=self.bounds),
                outcome_transform=Standardize(m=1),
            )
            new_model = new_model.to(device=self.device, dtype=self.dtype)
        new_model.eval()
        return new_model

    def fit(self, X_lf, Y_lf, X_hf, Y_hf, bounds):
        """
        Fit all parameters via 3 rounds of alternating optimization:
            Step 1: fit gp_lf on (X_lf, Y_lf)
            Step 2: residuals at HF:   Y_delta = Y_hf - rho*mu_L(X_hf)
            Step 3: fit gp_delta on (X_hf, Y_delta)
            Step 4: one Adam step on rho, fitting rho*mu_L + mu_delta to Y_hf
        bounds: [2, d] (row 0 = lower, row 1 = upper -- botorch convention).
        Stores X_lf, Y_lf, X_hf, Y_hf, Y_delta, bounds as raw-scale attrs.

        Step 1 (gp_lf) runs only once, before the round loop: its training
        data (X_lf, Y_lf) never depends on rho, so re-running its MLL
        optimization every round would just reproduce the identical fit
        (same data, same init) at 3x the cost. Steps 2-4 run once per round:
        round 0 fits gp_delta via a fresh MLL optimization (_build_gp);
        rounds 1-2 rebuild gp_delta with FROZEN hyperparameters copied from
        round 0 (_rebuild_frozen_gp) on the newly-recomputed Y_delta target
        (rho shifts slightly each round, so the residual VALUES change, but
        there's no reason to assume delta's overall smoothness/kernel shape
        needs re-learning every round too). rho gets one Adam step per
        round (3 total, via a single persistent optimizer instance so its
        momentum carries across rounds), not just one -- with only a single
        step, rho barely moves from its rho_init prior regardless of how
        much the data actually supports a different LF/HF correlation.
        """
        self.bounds = bounds.to(device=self.device, dtype=self.dtype)
        X_lf = X_lf.to(device=self.device, dtype=self.dtype)
        Y_lf = Y_lf.to(device=self.device, dtype=self.dtype).reshape(-1)
        X_hf = X_hf.to(device=self.device, dtype=self.dtype)
        Y_hf = Y_hf.to(device=self.device, dtype=self.dtype).reshape(-1)

        # Step 1 (once)
        self.gp_lf = self._build_gp(X_lf, Y_lf)

        rho_optimizer = torch.optim.Adam([self.log_rho], lr=self.lr)
        Y_delta = None
        for round_idx in range(3):
            # Step 2: residuals at HF, using rho as of the START of this
            # round (rho_init on round 0, updated by the previous round's
            # Step 4 otherwise).
            rho_val = torch.sigmoid(self.log_rho).item()
            with torch.no_grad():
                mu_lf = self.gp_lf.posterior(X_hf).mean.reshape(-1)
            Y_delta = Y_hf - rho_val * mu_lf

            # Step 3
            if round_idx == 0:
                self.gp_delta = self._build_gp(X_hf, Y_delta)
            else:
                self.gp_delta = self._rebuild_frozen_gp(self.gp_delta, X_hf, Y_delta)

            # Step 4: one Adam step tuning rho against the current gp_lf/gp_delta
            with torch.no_grad():
                mu_lf_at_hf = self.gp_lf.posterior(X_hf).mean.reshape(-1)
                mu_delta_at_hf = self.gp_delta.posterior(X_hf).mean.reshape(-1)
            rho_optimizer.zero_grad()
            pred = self.rho * mu_lf_at_hf + mu_delta_at_hf
            loss = torch.nn.functional.mse_loss(pred, Y_hf)
            loss.backward()
            rho_optimizer.step()

        self.train_x_lf = X_lf
        self.train_y_lf = Y_lf
        self.train_x_hf = X_hf
        self.train_y_hf = Y_hf
        self.train_y_delta = Y_delta

    def hf_posterior(self, X):
        """
        mu_H(x)  = rho * mu_L(x) + mu_delta(x)
        var_H(x) = rho^2 * var_L(x) + var_delta(x)
        (f_L and delta are independent GPs, so their posterior variances add
        under this linear combination.)
        Returns: (mean [N], variance [N])
        """
        X = X.to(device=self.device, dtype=self.dtype)
        if X.ndim == 1:
            X = X.unsqueeze(0)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            post_lf = self.gp_lf.posterior(X)
            post_delta = self.gp_delta.posterior(X)
            mu_lf = post_lf.mean.reshape(-1)
            var_lf = post_lf.variance.clamp_min(1e-12).reshape(-1)
            mu_delta = post_delta.mean.reshape(-1)
            var_delta = post_delta.variance.clamp_min(1e-12).reshape(-1)
            rho_val = self.rho.detach()
            mu_h = rho_val * mu_lf + mu_delta
            var_h = rho_val ** 2 * var_lf + var_delta
        return mu_h, var_h

    def lf_posterior(self, X):
        """LF posterior from gp_lf. Returns: (mean [N], var [N])"""
        X = X.to(device=self.device, dtype=self.dtype)
        if X.ndim == 1:
            X = X.unsqueeze(0)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            post_lf = self.gp_lf.posterior(X)
            mean = post_lf.mean.reshape(-1)
            var = post_lf.variance.clamp_min(1e-12).reshape(-1)
        return mean, var

    def sample_fantasy(self, x, fidelity):
        """
        Draw one fantasy observation at x.
        x: [d] or [1, d]
        fidelity: 'L' or 'H' -- 'H' samples f_L and delta independently (per
        the model's independence assumption) and combines them as
        rho*y_L_sample + y_delta_sample.
        Returns: scalar float y (raw scale)
        """
        x = x.to(device=self.device, dtype=self.dtype)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if fidelity == 'L':
                posterior = self.gp_lf.posterior(x, observation_noise=True)
                y = posterior.sample().reshape(-1)
                return y.item()
            elif fidelity == 'H':
                post_lf = self.gp_lf.posterior(x, observation_noise=True)
                post_delta = self.gp_delta.posterior(x, observation_noise=True)
                sample_lf = post_lf.sample().reshape(-1)
                sample_delta = post_delta.sample().reshape(-1)
                y = self.rho.detach() * sample_lf + sample_delta
                return y.item()
            else:
                raise ValueError(f"Unknown fidelity: {fidelity!r}, expected 'L' or 'H'")

    def make_fantasy_ko(self, x_new, y_new_raw, fidelity):
        """
        Return a NEW KennedyOHaganGP conditioned on one obs. DOES NOT mutate
        self. Follows dro.py's _make_fantasy_model pattern exactly: rebuild
        the affected GP(s) from scratch on augmented raw data, reusing frozen
        hyperparameters (no MLL refitting) via _rebuild_frozen_gp.

        fidelity == 'L': gp_lf is rebuilt on the augmented LF data; since
        gp_delta's own training targets (Y_delta = Y_hf - rho*mu_L(X_hf))
        depend on gp_lf's posterior mean at the (unchanged) HF locations,
        Y_delta is recomputed against the NEW gp_lf and gp_delta is rebuilt
        (frozen hyperparameters) on that updated residual target.

        fidelity == 'H': gp_lf is unaffected (referenced directly, not
        rebuilt) -- only gp_delta is rebuilt, on the augmented HF residual
        y_delta_new = y_new_raw - rho*mu_L(x_new).

        rho is copied frozen (log_rho.detach().clone()) in both cases -- no
        re-optimization of rho happens here, matching "hyperparameters are
        frozen" for the whole model, not just the GPs.
        """
        new_ko = KennedyOHaganGP(self.d, device=self.device, dtype=self.dtype)
        new_ko.log_rho = nn.Parameter(self.log_rho.detach().clone())
        new_ko.bounds = self.bounds
        new_ko.lr = self.lr
        new_ko.train_iter = self.train_iter
        new_ko.noise_lb = self.noise_lb

        x_new = x_new.to(device=self.device, dtype=self.dtype)
        if x_new.ndim == 1:
            x_new = x_new.unsqueeze(0)
        y_new_raw = y_new_raw.to(device=self.device, dtype=self.dtype).reshape(-1)

        rho_val = torch.sigmoid(self.log_rho).detach().item()

        if fidelity == 'L':
            aug_x_lf = torch.cat([self.train_x_lf, x_new], dim=0)
            aug_y_lf = torch.cat([self.train_y_lf, y_new_raw], dim=0)
            new_gp_lf = self._rebuild_frozen_gp(self.gp_lf, aug_x_lf, aug_y_lf)

            with torch.no_grad():
                mu_lf_at_hf = new_gp_lf.posterior(self.train_x_hf).mean.reshape(-1)
            new_y_delta = self.train_y_hf - rho_val * mu_lf_at_hf
            new_gp_delta = self._rebuild_frozen_gp(self.gp_delta, self.train_x_hf, new_y_delta)

            new_ko.gp_lf = new_gp_lf
            new_ko.gp_delta = new_gp_delta
            new_ko.train_x_lf = aug_x_lf
            new_ko.train_y_lf = aug_y_lf
            new_ko.train_x_hf = self.train_x_hf
            new_ko.train_y_hf = self.train_y_hf
            new_ko.train_y_delta = new_y_delta

        elif fidelity == 'H':
            with torch.no_grad():
                mu_lf_at_new = self.gp_lf.posterior(x_new).mean.item()
            y_delta_new = y_new_raw.item() - rho_val * mu_lf_at_new

            aug_x_hf = torch.cat([self.train_x_hf, x_new], dim=0)
            aug_y_delta = torch.cat(
                [self.train_y_delta,
                 torch.tensor([y_delta_new], device=self.device, dtype=self.dtype)],
                dim=0,
            )
            new_gp_delta = self._rebuild_frozen_gp(self.gp_delta, aug_x_hf, aug_y_delta)

            new_ko.gp_lf = self.gp_lf  # reference directly, not rebuilt
            new_ko.gp_delta = new_gp_delta
            new_ko.train_x_lf = self.train_x_lf
            new_ko.train_y_lf = self.train_y_lf
            new_ko.train_x_hf = aug_x_hf
            new_ko.train_y_hf = torch.cat([self.train_y_hf, y_new_raw], dim=0)
            new_ko.train_y_delta = aug_y_delta

        else:
            raise ValueError(f"Unknown fidelity: {fidelity!r}, expected 'L' or 'H'")

        return new_ko
