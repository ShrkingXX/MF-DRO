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
import time
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
from gpytorch.constraints import GreaterThan
from gpytorch.priors import LogNormalPrior
from gpytorch.mlls import ExactMarginalLogLikelihood

# Lengthscale prior (replaces the old hard Interval constraint -- see
# KennedyOHaganGP's docstring). Module-level so DirectMFRegretOptimization's
# ensemble-diversity lengthscale grid (mf_dro.py) can derive a sensible
# spread from these SAME two numbers instead of duplicating them -- a
# duplicate is exactly the bug class that broke when the old
# _lengthscale_bounds() formula last changed (mf_dro.py kept generating
# grid values the updated bound rejected).
LENGTHSCALE_PRIOR_LOC = math.log(0.5)
LENGTHSCALE_PRIOR_SCALE = 0.5


# ════════════════════════════════════
# Deep Kernel Learning (Wilson et al. 2015, DKL.pdf) -- see docstrings below
# for the sparse-data RBF/DKL switching rationale (DRO.pdf Appendix C.2
# cites Zhang, Desautels & Chen 2025 [44] for deep kernels in MF-BO).
# ════════════════════════════════════

class DeepKernelFeatureExtractor(nn.Module):
    """
    Small MLP mapping d-dimensional input to d_feature-dimensional feature
    space. Jointly learned with GP hyperparameters through MLL
    backpropagation (Wilson et al. 2015, Eq. 5: k(x,x'|theta) ->
    k(g(x,w), g(x',w)|theta,w)).

    Architecture: d -> hidden -> d_feature, single hidden layer of FIXED
    width (not d-dependent). Tanh activation (bounded, unlike ReLU, which
    matters here since feature-space magnitude directly scales the RBF
    base kernel's distance computation).

    Previously d -> [d*2, d*2] -> d_feature (two d-dependent hidden
    layers): the middle layer's O(d^2) parameter scaling (d*2 x d*2) blew
    up to 266-450 params for d=6/d=8 benchmarks even at d_feature=2,
    breaking the fixed dkl_threshold=30 design (30/266=0.11x,
    30/450=0.07x data-to-parameter ratio). A single fixed-width hidden
    layer keeps parameter count roughly constant across benchmark
    dimensions (38-46 params at hidden=4, d_feature=2, threshold=30 -->
    ~0.65-0.79x ratio, just under 1:1) instead of scaling with d.

    d_feature default: max(4, d // 2) -- small enough to avoid overfitting
    on the 5-50 point regime KO GP operates in (see KennedyOHaganGP's
    dkl_threshold), expressive enough to learn nontrivial structure.
    """

    def __init__(self, d, d_feature=None, hidden=4):
        super().__init__()
        if d_feature is None:
            d_feature = max(4, d // 2)
        self.d_feature = d_feature
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.Tanh(),
            nn.Linear(hidden, d_feature),
        )

    def forward(self, x):
        return self.net(x)


class DeepKernel(gpytorch.kernels.Kernel):
    """
    Deep kernel: k_DKL(x, x') = k_base(g(x,w), g(x',w)) where g is
    DeepKernelFeatureExtractor and k_base is RBF (Wilson et al. 2015).
    Drop-in replacement for ScaleKernel(RBFKernel) in
    KennedyOHaganGP._build_gp() when DKL is active. DNN weights w and base
    RBF hyperparameters are jointly optimized through the same MLL
    backprop loop _build_gp already uses -- no separate pretraining step
    (unlike Wilson et al.'s own large-data DNN-pretrain-then-joint-tune
    procedure, unnecessary here since the base network is tiny and the
    whole point is joint learning on scarce data).

    NOT combined with KISS-GP/inducing-point scalability machinery
    (Wilson et al.'s Section 4 scalability extension) -- irrelevant at the
    N<50 point regime this class operates in; exact GP inference is
    already trivial at this scale.
    """
    has_lengthscale = False  # base kernel handles this

    def __init__(self, d, d_feature=None, **kwargs):
        super().__init__(**kwargs)
        self.feature_extractor = DeepKernelFeatureExtractor(d, d_feature)
        d_feat = self.feature_extractor.d_feature
        self.base_kernel = ScaleKernel(RBFKernel(ard_num_dims=d_feat))

    def forward(self, x1, x2, **params):
        z1 = self.feature_extractor(x1)
        z2 = self.feature_extractor(x2)
        return self.base_kernel(z1, z2)


class KennedyOHaganGP:
    """
    d: input dimension
    All GPs use: Normalize(d) + Standardize(m=1) transforms, RBF kernel with
    ARD=True, ScaleKernel wrapper, GaussianLikelihood with noise >= noise_lb.
    Lengthscale has NO hard constraint (GPyTorch's default Positive/softplus
    transform) -- instead a LogNormalPrior(loc=log(0.5), scale=0.5) softly
    regularizes it toward 0.5, letting MLL settle shorter or longer if the
    data genuinely supports it, rather than the previous hard Interval
    constraint, which a diagnostic found MLL pinning against exactly (first
    at 0.05*sqrt(d)=0.12 for d=6, then again at a raised floor of 0.2 --
    the optimizer wanted shorter than *any* hard floor tried, on this
    sparse-data/noise_lb regime, so a soft prior replaces the floor
    entirely rather than raising it again). See LENGTHSCALE_PRIOR_LOC/
    LENGTHSCALE_PRIOR_SCALE below.
    """

    def __init__(self, d, rho_init=0.8, lr=0.1, train_iter=50,
                 noise_lb=1e-2, device='cpu', dtype=torch.float64,
                 dkl_threshold=30, dkl_train_iter=100, d_feature=None,
                 rho_fixed=None, initial_lengthscale=None):
        # noise_lb default raised from 1e-4: a near-zero noise floor made
        # near-interpolation (very short lengthscale, near-exact fit to
        # each training point) more attractive to the MLL optimizer on
        # sparse high-D data -- see the class docstring's LogNormalPrior
        # note for the diagnostic that found this. 1e-2 is light
        # regularization, not expected to meaningfully change behavior
        # once data is denser.
        self.d = d
        self.rho_init = rho_init  # re-applied every fit() call when
        # initial_lengthscale is set -- see fit()'s reset and
        # initial_lengthscale's docstring below for why.
        self.log_rho = nn.Parameter(
            torch.tensor(math.log(rho_init / (1.0 - rho_init)))
        )
        # rho_fixed: pin the autoregressive coefficient instead of fitting it.
        # rho_fixed=1.0 degenerates the KO model f_H = rho*f_L + delta into the
        # purely additive f_H = f_L + delta, which is the Additive-MES ablation
        # in src/baselines/additive_mes.py -- with rho pinned, that variant
        # differs from the KO model in exactly one respect (whether the global
        # fidelity correlation is learned from data), so any performance gap is
        # attributable to the fitted rho alone. sigmoid(log_rho) can never
        # actually reach 1.0, so this is a genuine override of the `rho`
        # property rather than an initialization, and fit()'s Step 4 skips the
        # rho update entirely when it is set.
        self.rho_fixed = rho_fixed
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

        # GP warm-starting (Change 2, DRO.pdf Section D.2: "GPs are not
        # retrained from scratch... retrain: False"). fit() stores each
        # round-0 gp_lf/gp_delta's state_dict here after every call; the
        # NEXT fit() call's round 0 loads it back via _build_gp's
        # prev_state_dict, in place of a random cold-start init, and uses
        # fewer Adam iterations (train_iter_warm instead of train_iter) --
        # see fit()/_build_gp docstrings for exactly which round loads what.
        # DISABLED (never populated/consumed) when initial_lengthscale is
        # set -- see that param's docstring below for why.
        self.prev_state_dict_lf = None
        self.prev_state_dict_delta = None
        self.train_iter_warm = max(10, train_iter // 2)

        # Ensemble-diversity anchor (analogous to SF-DRO's dro.py
        # _initialize_models/_update_models: each ensemble member gets a
        # DIFFERENT fixed initial_lengthscale from a grid, re-applied via
        # _build_gp's RBF branch on EVERY fit() call -- not just the first).
        # None (default) means "use the fixed 0.5 init, the LogNormalPrior's
        # own center" (unchanged behavior for any non-ensemble/single-
        # instance use). The prior itself (regularization target) stays
        # shared across ensemble members regardless of this setting -- only
        # the Adam search's STARTING point differs per member.
        # When set, this ALSO disables warm-starting for this instance:
        # SF-DRO's own _update_models docstring is explicit that rebuilding
        # from scratch, anchored to each member's own designated
        # initial_lengthscale every call, is *why* its ensemble stays
        # diverse -- warm-starting would let members drift from their
        # distinct anchors toward whatever the (possibly shared) MLL
        # landscape's gradient points to, exactly the "gradually converge
        # toward each other over many iterations" failure this is meant to
        # prevent. Applied uniformly (not just to the RBF lengthscale): the
        # DKL branch also cold-restarts every call under this flag, rather
        # than selectively warm-starting the DNN while the lengthscale
        # resets -- a partial reset would be an inconsistent, hard-to-reason
        # -about hybrid of the two designs. rho is reset the same way (see
        # fit()) -- log_rho is a persistent nn.Parameter that otherwise
        # keeps taking Adam steps every fit() call with nothing to re-anchor
        # it, so even with per-member rho_init, ensemble rho std was
        # observed to decay monotonically toward 0 over iterations (a
        # diagnostic finding, not a hypothetical) if left unreset -- the
        # exact convergence-toward-each-other failure this whole mechanism
        # exists to prevent, just via a different parameter than lengthscale.
        self.initial_lengthscale = initial_lengthscale

        # DKL config -- see DeepKernel/fit() docstrings. Starts False;
        # fit() activates it once n_hf reaches dkl_threshold.
        # d_feature=2 (not max(4,d//2)) and dkl_threshold=30 (not 15):
        # under the fixed-width single-hidden-layer architecture (see
        # DeepKernelFeatureExtractor's docstring), d_feature=2/hidden=4
        # gives 38-46 DNN params across d=6/d=8, and at threshold=30 the
        # ~0.65-0.79x data-to-parameter ratio is in the regime where MLL
        # regularization can actually constrain the DNN. The DKL paper's
        # own results (Wilson et al. 2015) cover 2,565+ points -- its
        # architecture choices aren't calibrated for n<50.
        self.use_dkl = False
        self.dkl_threshold = dkl_threshold  # HF obs needed before switching
        self.dkl_train_iter = dkl_train_iter  # more iterations for DKL fitting
        self.d_feature = d_feature if d_feature is not None else 2
        self._last_rbf_fit_time = None  # for the DKL-activation timing check

    @property
    def rho(self):
        if self.rho_fixed is not None:
            return torch.tensor(self.rho_fixed, device=self.device, dtype=self.dtype)
        return torch.sigmoid(self.log_rho)

    def _build_gp(self, X_raw, Y_raw, use_dkl=False, prev_state_dict=None):
        """
        Build a fresh SingleTaskGP on (X_raw, Y_raw), MLL-fit via Adam.
        Mirrors dro.py's _construct_gp_model exactly (same
        likelihood/transform choices); covariance is either RBF+ARD with a
        bounded lengthscale (use_dkl=False, this class's original spec) or
        DeepKernel (use_dkl=True, requires more data -- see fit()).
        X_raw: [N, d], Y_raw: [N]. Returns model in eval() mode.

        prev_state_dict (Change 2, GP warm-starting -- DRO.pdf Section D.2):
        if given, loaded into the freshly-constructed model via
        load_state_dict(..., strict=False) BEFORE the Adam MLL loop below,
        and the loop then runs self.train_iter_warm iterations instead of
        the cold-start self.train_iter/self.dkl_train_iter count. The
        caller (fit()) only ever passes a prev_state_dict captured from a
        model with the SAME architecture (same use_dkl, same d) as the one
        being built here, so shapes normally match -- strict=False guards
        against the one thing that legitimately still differs, buffers
        derived from train_inputs/train_targets (N changes every BO
        iteration as data accumulates), skipping those instead of raising.
        The except is a last-resort fallback (an incompatible dict slipping
        through some other way) that falls back to a full cold-start init
        rather than let fit() crash.
        """
        if use_dkl:
            # Deep kernel -- no explicit lengthscale constraint (base
            # kernel inside DeepKernel handles its own).
            covar_module = DeepKernel(d=self.d, d_feature=self.d_feature)
            train_iters = self.dkl_train_iter
        else:
            # No hard Interval constraint (GPyTorch's default Positive/
            # softplus transform instead) -- a LogNormalPrior softly
            # regularizes toward 0.5 without hard-blocking the optimizer
            # from going shorter or longer if the data supports it. See
            # KennedyOHaganGP's class docstring and LENGTHSCALE_PRIOR_LOC/
            # LENGTHSCALE_PRIOR_SCALE above for why this replaced the old
            # hard floor (MLL kept pinning against it, at two different
            # floor values, rather than finding a genuine data-driven
            # optimum below it).
            base_kernel = RBFKernel(ard_num_dims=self.d)
            base_kernel.register_prior(
                'lengthscale_prior',
                LogNormalPrior(loc=LENGTHSCALE_PRIOR_LOC, scale=LENGTHSCALE_PRIOR_SCALE),
                'lengthscale'
            )
            # Ensemble-diversity anchor (see __init__'s initial_lengthscale
            # docstring): each member re-anchors to its OWN fixed value on
            # every call, instead of the shared 0.5 default -- the PRIOR
            # (regularization target) is shared across members, but the
            # starting point Adam searches from stays diverse.
            init_ls = self.initial_lengthscale if self.initial_lengthscale is not None else 0.5
            base_kernel.initialize(lengthscale=init_ls)
            covar_module = ScaleKernel(base_kernel)
            train_iters = self.train_iter

        likelihood = GaussianLikelihood(noise_constraint=GreaterThan(self.noise_lb))
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

        if prev_state_dict is not None:
            try:
                model.load_state_dict(prev_state_dict, strict=False)
                train_iters = self.train_iter_warm
            except Exception as e:
                print(f"[KO GP] warm-start load_state_dict failed "
                      f"({e!r}), falling back to cold-start init")

        model.train()
        model.likelihood.train()
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        gp_optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        for _ in range(train_iters):
            gp_optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, model.train_targets)
            loss.backward()
            # Prevents NaN from poorly-conditioned DNN initialization
            # (DeepKernel's feature_extractor) propagating into the GP fit.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        branch to this class's RBF-with-ARD-or-DeepKernel covariance.

        For DKL: copies both DNN weights (in covar_module.feature_extractor)
        and GP hyperparameters via load_state_dict -- works for both cases
        because DNN weights are ordinary registered nn.Module parameters,
        included in covar_module's state_dict alongside the base kernel's.
        """
        X_aug = X_aug_raw.to(device=self.device, dtype=self.dtype)
        Y_aug = Y_aug_raw.to(device=self.device, dtype=self.dtype).reshape(-1, 1)

        new_likelihood = GaussianLikelihood(noise_constraint=GreaterThan(self.noise_lb))
        new_likelihood.noise = old_model.likelihood.noise.detach().clone()

        is_dkl = isinstance(old_model.covar_module, DeepKernel)
        if is_dkl:
            new_covar_module = DeepKernel(d=self.d, d_feature=self.d_feature)
        else:
            # No explicit lengthscale_constraint here either (must match
            # _build_gp's construction exactly -- GPyTorch's default
            # Positive/softplus transform, not the old Interval/sigmoid
            # one): load_state_dict copies the RAW (pre-transform)
            # parameter value, so decoding it through a differently-shaped
            # transform would silently produce a different actual
            # lengthscale than the one being "frozen" from old_model.
            new_covar_module = ScaleKernel(RBFKernel(ard_num_dims=self.d))
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

        Activates DKL when n_hf >= self.dkl_threshold (both gp_lf and
        gp_delta switch together): below threshold, RBF (stable for sparse
        data); at/above, DeepKernel (learns feature structure once enough
        HF data exists to support it -- see DeepKernel's docstring).

        All 4 steps run fresh EVERY round (3 rounds total): gp_lf and
        gp_delta are both rebuilt via a full MLL fit (_build_gp) each round
        -- not the frozen-hyperparameter rebuild used elsewhere in this
        class for fantasy conditioning -- and rho gets one Adam step per
        round via a SINGLE optimizer constructed once before the round loop,
        so momentum accumulates across all 3 rounds (a fresh optimizer per
        round would discard momentum state, making the 3 rounds behave like
        plain SGD instead of Adam). Re-fitting gp_lf on the same (X_lf, Y_lf)
        every round is redundant computation under RBF (deterministic Adam
        trajectory, so it reproduces the same fit each time) but kept for
        exact parity with the specified alternating-optimization procedure;
        under DKL it is NOT redundant (fresh random DNN init each round).
        """
        self.bounds = bounds.to(device=self.device, dtype=self.dtype)
        X_lf = X_lf.to(device=self.device, dtype=self.dtype)
        Y_lf = Y_lf.to(device=self.device, dtype=self.dtype).reshape(-1)
        X_hf = X_hf.to(device=self.device, dtype=self.dtype)
        Y_hf = Y_hf.to(device=self.device, dtype=self.dtype).reshape(-1)

        # Activate DKL if enough HF data.
        n_hf = X_hf.shape[0]
        use_dkl = (n_hf >= self.dkl_threshold)
        just_activated = use_dkl and not self.use_dkl
        if just_activated:
            print(f"[KO GP] Activating DKL at n_hf={n_hf} "
                  f"(threshold={self.dkl_threshold})")
            # Change 2: a just-activated DKL model has a different
            # architecture (DeepKernel vs ScaleKernel(RBFKernel)) than
            # whatever produced the stored warm-start state dicts, so they
            # are no longer valid initializations -- discard rather than
            # let _build_gp's strict=False silently no-op-load them.
            self.prev_state_dict_lf = None
            self.prev_state_dict_delta = None
        self.use_dkl = use_dkl

        # rho re-anchor (see initial_lengthscale's docstring above): reset
        # log_rho to this instance's own rho_init EVERY fit() call, for the
        # same diversity-preservation reason lengthscale re-anchors via
        # _build_gp -- log_rho is otherwise a persistent nn.Parameter with
        # no reset, so it keeps drifting via Step 4's Adam updates below
        # regardless of how distinct rho_init was at construction.
        if self.initial_lengthscale is not None and self.rho_fixed is None:
            with torch.no_grad():
                self.log_rho.fill_(math.log(self.rho_init / (1.0 - self.rho_init)))

        _fit_t0 = time.time()
        Y_delta = None
        rho_optimizer = torch.optim.Adam([self.log_rho], lr=self.lr)
        for round_idx in range(3):
            # Step 1: fit gp_lf on LF data. Warm-start (Change 2) only on
            # round 0, from the PREVIOUS fit() call's final state -- rounds
            # 1-2 stay cold-start, unchanged from before (see fit()'s
            # docstring on why every round rebuilds from scratch). Warm-start
            # is disabled entirely (lf_prev always None) when
            # initial_lengthscale is set -- see __init__'s docstring on why
            # ensemble-diversity anchors and warm-starting are mutually
            # exclusive for a given instance.
            _warm_ok = self.initial_lengthscale is None
            lf_prev = self.prev_state_dict_lf if (round_idx == 0 and _warm_ok) else None
            self.gp_lf = self._build_gp(X_lf, Y_lf, use_dkl=use_dkl, prev_state_dict=lf_prev)

            # Step 2: recompute residuals using updated gp_lf and rho
            rho_val = self.rho.item()
            with torch.no_grad():
                mu_lf_at_hf = self.gp_lf.posterior(X_hf).mean.reshape(-1)
            Y_delta = Y_hf - rho_val * mu_lf_at_hf

            # Step 3: fit gp_delta on updated residuals (same round-0-only
            # warm-start policy as gp_lf above).
            delta_prev = self.prev_state_dict_delta if (round_idx == 0 and _warm_ok) else None
            self.gp_delta = self._build_gp(X_hf, Y_delta, use_dkl=use_dkl, prev_state_dict=delta_prev)

            # Step 4: one Adam step on rho -- skipped entirely when rho is
            # pinned (rho_fixed), since the `rho` property ignores log_rho in
            # that case and stepping it would be a no-op that still burns a
            # posterior evaluation per round.
            if self.rho_fixed is None:
                with torch.no_grad():
                    mu_lf_at_hf = self.gp_lf.posterior(X_hf).mean.reshape(-1)
                    mu_delta_at_hf = self.gp_delta.posterior(X_hf).mean.reshape(-1)
                rho_optimizer.zero_grad()
                pred = (torch.sigmoid(self.log_rho)
                        * mu_lf_at_hf + mu_delta_at_hf)
                loss = torch.nn.functional.mse_loss(pred, Y_hf)
                loss.backward()
                rho_optimizer.step()

        self.train_x_lf = X_lf
        self.train_y_lf = Y_lf
        self.train_x_hf = X_hf
        self.train_y_hf = Y_hf
        self.train_y_delta = Y_delta

        # Change 2: snapshot this call's final hyperparameters for the
        # NEXT fit() call's round-0 warm-start (see round loop above).
        # Skipped when initial_lengthscale is set -- warm-starting is
        # disabled for this instance, so there's nothing to read it back.
        if self.initial_lengthscale is None:
            self.prev_state_dict_lf = self.gp_lf.state_dict()
            self.prev_state_dict_delta = self.gp_delta.state_dict()

        # Timing checkpoint: verify DKL overhead is acceptable at the
        # moment it first activates, comparing against the most recent
        # RBF-mode fit() call's time (tracks the same accumulating data
        # size the switch itself was triggered by).
        fit_elapsed = time.time() - _fit_t0
        if not use_dkl:
            self._last_rbf_fit_time = fit_elapsed
        elif just_activated:
            rbf_str = (f"{self._last_rbf_fit_time:.2f}s"
                       if self._last_rbf_fit_time is not None else "n/a")
            print(f"DKL fit time: {fit_elapsed:.2f}s  (RBF was {rbf_str})")

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

    def sample_fantasy(self, x, fidelity, mode='sample'):
        """
        Draw one fantasy observation at x.
        x: [d] or [1, d]
        fidelity: 'L' or 'H' -- 'H' samples f_L and delta independently (per
        the model's independence assumption) and combines them as
        rho*y_L_sample + y_delta_sample.
        mode='sample' (default, unchanged): draw from the posterior.
        mode='mean':   return the posterior MEAN instead -- a
            certainty-equivalent transition. This makes the rollout's
            dynamics DETERMINISTIC given the action sequence, which is the
            condition Brandfonbrener et al. (2022, arXiv:2206.01079)
            Corollary 1 requires for return-conditioned supervised learning
            to be near-optimal (their bound scales with epsilon, the degree
            of departure from deterministic dynamics). It does NOT make the
            behaviour policy deterministic -- their epsilon concerns the
            MDP's transition/reward, not beta -- so rollout DIVERSITY is
            preserved. See literature/rcsl-necessary-conditions.md.

        Returns: scalar float y (raw scale)
        """
        if mode not in ('sample', 'mean'):
            raise ValueError(f"mode must be 'sample' or 'mean', got {mode!r}")
        x = x.to(device=self.device, dtype=self.dtype)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if fidelity == 'L':
                posterior = self.gp_lf.posterior(x, observation_noise=True)
                y = (posterior.mean if mode == 'mean'
                     else posterior.sample()).reshape(-1)
                return y.item()
            elif fidelity == 'H':
                post_lf = self.gp_lf.posterior(x, observation_noise=True)
                post_delta = self.gp_delta.posterior(x, observation_noise=True)
                sample_lf = (post_lf.mean if mode == 'mean'
                             else post_lf.sample()).reshape(-1)
                sample_delta = (post_delta.mean if mode == 'mean'
                                else post_delta.sample()).reshape(-1)
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
        new_ko = KennedyOHaganGP(self.d, device=self.device, dtype=self.dtype,
                                  rho_fixed=self.rho_fixed)
        new_ko.log_rho = nn.Parameter(self.log_rho.detach().clone())
        new_ko.bounds = self.bounds
        new_ko.lr = self.lr
        new_ko.train_iter = self.train_iter
        new_ko.noise_lb = self.noise_lb
        # DKL config/status propagated too -- _rebuild_frozen_gp below
        # determines DKL-vs-RBF per rebuilt GP via isinstance(old_model.
        # covar_module, DeepKernel) regardless of this flag, so the actual
        # covariance type is already correct without this; without it,
        # though, new_ko.use_dkl would read back False even when its own
        # gp_lf/gp_delta are genuinely DeepKernel-based, which would mislead
        # any future code (or debugging) that trusts this flag directly.
        new_ko.use_dkl = self.use_dkl
        new_ko.dkl_threshold = self.dkl_threshold
        new_ko.dkl_train_iter = self.dkl_train_iter
        new_ko.d_feature = self.d_feature

        x_new = x_new.to(device=self.device, dtype=self.dtype)
        if x_new.ndim == 1:
            x_new = x_new.unsqueeze(0)
        y_new_raw = y_new_raw.to(device=self.device, dtype=self.dtype).reshape(-1)

        rho_val = self.rho.detach().item()

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
