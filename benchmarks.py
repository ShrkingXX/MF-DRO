"""
Benchmark registry for Experiment 1 (MES reward ablation) and Experiment 2
(RTG schema comparison). Each entry has everything run_single_seed needs to
set up a run: dimension, domain bounds, the true optimum of the *raw*
(pre-negation) objective (for correct regret logging via
DirectRegretOptimization's known_optimal_value), and a factory that builds
the actual negated-for-maximization BoTorch objective.

Note on "Shekel": the classical Shekel function is inherently 4-dimensional
(BoTorch's implementation fixes dim=4; there is no standard d=10 variant), so
this uses the standard Shekel(m=5), dim=4, domain [0,10]^4 rather than the
d=10 figure floated earlier in this session -- confirmed with the user.
"""
import itertools

import torch
from scipy.optimize import minimize

from botorch.test_functions.synthetic import EggHolder, Shekel, Michalewicz, Hartmann

from src.objectives import (Ackley, Rosenbrock, CurrinExpHF, CurrinExpLF, HartmannLF,
                             BoreholeFunctionHF, BoreholeFunctionLF,
                             AckleyFunctionHF, AckleyFunctionLF,
                             HartmannWidened6D, HartmannWidenedLF6D)


def _ackley_2d():
    return Ackley(dim=2, bounds=[(-32.768, 32.768)] * 2, negate=True)


def _ackley_5d():
    return Ackley(dim=5, bounds=[(-5.0, 5.0)] * 5, negate=True)


def _ackley_10d():
    # Classic bounds (matches Ackley_2D's convention), not Ackley_5D's tighter
    # [-5,5] -- explicit user choice for the mes_switching_v2 cluster experiment.
    return Ackley(dim=10, bounds=[(-32.768, 32.768)] * 10, negate=True)


def _rosenbrock_2d():
    return Rosenbrock(dim=2, bounds=[(-5.0, 10.0), (-5.0, 10.0)], negate=True)


def _eggholder():
    return EggHolder(negate=True) # dim fixed at 2, bounds fixed at [-512,512]^2


def _shekel_m5():
    return Shekel(m=5, negate=True) # dim fixed at 4, bounds [0,10]^4


def _michalewicz_10d():
    return Michalewicz(dim=10, negate=True)


def _hartmann_6d():
    return Hartmann(dim=6, negate=True)


def _currin_2d():
    return CurrinExpHF(negate=True)


BENCHMARKS = {
    # --- Experiment 1 only ---
    "Eggholder": {
        "dim": 2, "domain_min": [-512.0, -512.0], "domain_max": [512.0, 512.0],
        "known_optimal_value": EggHolder(negate=False).optimal_value,
        "make_objective": _eggholder,
    },
    "Shekel": {
        "dim": 4, "domain_min": [0.0] * 4, "domain_max": [10.0] * 4,
        "known_optimal_value": Shekel(m=5, negate=False).optimal_value,
        "make_objective": _shekel_m5,
    },
    "Michalewicz": {
        "dim": 10, "domain_min": [0.0] * 10, "domain_max": [3.141592653589793] * 10,
        "known_optimal_value": Michalewicz(dim=10, negate=False).optimal_value,
        "make_objective": _michalewicz_10d,
    },
    "Rosenbrock_2D": {
        "dim": 2, "domain_min": [-5.0, -5.0], "domain_max": [10.0, 10.0],
        "known_optimal_value": 0.0,
        "make_objective": _rosenbrock_2d,
    },
    # --- Shared between Experiment 1 and Experiment 2 ---
    "Ackley_2D": {
        "dim": 2, "domain_min": [-32.768, -32.768], "domain_max": [32.768, 32.768],
        "known_optimal_value": 0.0,
        "make_objective": _ackley_2d,
    },
    "Ackley_5D": {
        "dim": 5, "domain_min": [-5.0] * 5, "domain_max": [5.0] * 5,
        "known_optimal_value": 0.0,
        "make_objective": _ackley_5d,
    },
    "Ackley_10D": {
        "dim": 10, "domain_min": [-32.768] * 10, "domain_max": [32.768] * 10,
        "known_optimal_value": 0.0,
        "make_objective": _ackley_10d,
    },
    "Hartmann_6D": {
        "dim": 6, "domain_min": [0.0] * 6, "domain_max": [1.0] * 6,
        "known_optimal_value": Hartmann(dim=6, negate=False).optimal_value,
        "make_objective": _hartmann_6d,
    },
    # --- Experiment 1 only ---
    "Currin_2D": {
        "dim": 2, "domain_min": [0.0, 0.0], "domain_max": [1.0, 1.0],
        "known_optimal_value": CurrinExpHF(negate=False).optimal_value,
        "make_objective": _currin_2d,
    },
}


# --- Two-fidelity benchmark pairs (Currin_2D_{HF,LF}, Hartmann_6D_{HF,LF}) ---
#
# CurrinExpLF/HartmannLF (src/objectives/synthetic.py) deliberately do NOT
# define _optimal_value/_optimizers: neither LF variant's global optimum has
# a known closed form (CurrinExpLF's 4-corner-average construction and
# HartmannLF's alpha substitution both distort the surface away from a
# derivable argmax), unlike Currin_2D/Hartmann_6D's HF optima above (which
# come from CurrinExpHF's own analytic derivation / the literature's known
# Hartmann-6 global minimum respectively). So known_optimal_value for the LF
# entries is instead determined empirically via dense random-grid search
# over the negated (maximization-ready) objective, mirroring how a real BO
# run would only ever observe empirical bests, not a closed-form value.
def _find_lf_optimum(f_lf, d, n=10000, seed=0):
    torch.manual_seed(seed)
    X = torch.rand(n, d)
    vals = f_lf(X)  # negated, so max = best
    return vals.max().item()


currin_lf_opt = _find_lf_optimum(CurrinExpLF(negate=True), d=2)
hartmann_lf_opt = _find_lf_optimum(HartmannLF(negate=True), d=6)
print(f"Currin LF optimum:  {currin_lf_opt:.6f}")
print(f"Hartmann LF optimum: {hartmann_lf_opt:.6f}")

BENCHMARKS.update({
    "Currin_2D_HF": {
        "dim": 2,
        "domain_min": [0.0, 0.0],
        "domain_max": [1.0, 1.0],
        "known_optimal_value": BENCHMARKS["Currin_2D"]["known_optimal_value"],
        "make_objective": lambda: CurrinExpHF(negate=True),
        "cost": 3.0,
        "fidelity": "H",
    },
    "Currin_2D_LF": {
        "dim": 2,
        "domain_min": [0.0, 0.0],
        "domain_max": [1.0, 1.0],
        "known_optimal_value": currin_lf_opt,
        "make_objective": lambda: CurrinExpLF(negate=True),
        "cost": 1.0,
        "fidelity": "L",
    },
    "Hartmann_6D_HF": {
        "dim": 6,
        "domain_min": [0.0] * 6,
        "domain_max": [1.0] * 6,
        "known_optimal_value": BENCHMARKS["Hartmann_6D"]["known_optimal_value"],
        "make_objective": lambda: Hartmann(dim=6, negate=True),
        "cost": 8.0,
        "fidelity": "H",
    },
    "Hartmann_6D_LF": {
        "dim": 6,
        "domain_min": [0.0] * 6,
        "domain_max": [1.0] * 6,
        "known_optimal_value": hartmann_lf_opt,
        "make_objective": lambda: HartmannLF(negate=True),
        "cost": 1.0,
        "fidelity": "L",
    },
})


# --- Borehole 8D benchmark pair (Borehole_8D_{HF,LF}) ---
#
# Neither BoreholeFunctionHF nor BoreholeFunctionLF has a known closed-form
# optimum in this codebase (see their docstrings in synthetic.py) -- unlike
# _find_lf_optimum above, Borehole's domain is NOT [0,1]^8 (each of the 8
# dimensions has its own physical range), so sampling needs to respect the
# actual per-dimension bounds rather than a unit hypercube.
#
# NORMALIZATION FIX: this was previously a plain 10k-uniform-sample max, which
# in 8D understates the true optimum badly (256.343 vs the true 309.576 for HF,
# 203.990 vs 246.352 for LF). Since regret = -best_observed - known_optimal_value,
# an understated optimum makes regret go NEGATIVE as soon as any optimizer finds
# a point better than the reference sample -- observed directly in the Stage 2
# Borehole runs (Greedy-MES final regrets of -9.3/-11.3/-10.0, MF-MI-Greedy
# -7.88 on every seed), which also makes log10(simple regret), the primary
# metric here, undefined. Borehole's optimum sits at a vertex of the box (both
# fidelities: u* = [1,0,1,1,1,0,0,1] in unit coordinates), so exhaustive corner
# enumeration finds it exactly; the random sample and L-BFGS-B polish below are
# kept as a guard against assuming vertex-optimality for any future objective
# registered through this helper.
def _find_optimum_bounded(f, domain_min, domain_max, n=20000, seed=0, n_polish=5):
    d = len(domain_min)
    lo = torch.tensor(domain_min, dtype=torch.float64)
    hi = torch.tensor(domain_max, dtype=torch.float64)

    corners = torch.tensor(
        list(itertools.product(*[[0.0, 1.0]] * d)), dtype=torch.float64
    )
    torch.manual_seed(seed)
    unit_X = torch.cat([corners, torch.rand(n, d, dtype=torch.float64)], dim=0)
    vals = f(lo + (hi - lo) * unit_X)  # negated, so max = best

    def _neg(u):
        x = (lo + (hi - lo) * torch.tensor(u, dtype=torch.float64)).unsqueeze(0)
        return -f(x).reshape(-1)[0].item()

    best = vals.max().item()
    for idx in torch.topk(vals.reshape(-1), n_polish).indices.tolist():
        res = minimize(_neg, unit_X[idx].numpy(), bounds=[(0.0, 1.0)] * d,
                       method="L-BFGS-B")
        best = max(best, -res.fun)
    return best


_BOREHOLE_DOMAIN_MIN = [0.05, 100.0, 63070.0, 990.0, 63.1, 700.0, 1120.0, 9855.0]
_BOREHOLE_DOMAIN_MAX = [0.15, 50000.0, 115600.0, 1110.0, 116.0, 820.0, 1680.0, 12045.0]

borehole_hf_opt = _find_optimum_bounded(
    BoreholeFunctionHF(negate=True), _BOREHOLE_DOMAIN_MIN, _BOREHOLE_DOMAIN_MAX
)
borehole_lf_opt = _find_optimum_bounded(
    BoreholeFunctionLF(negate=True), _BOREHOLE_DOMAIN_MIN, _BOREHOLE_DOMAIN_MAX
)

# SIGN CONVENTION FIX: known_optimal_value must be stored on the SAME scale
# dro.py/mf_dro.py's shared regret formula (-best_observed - known_optimal_value)
# expects -- confirmed by Currin_2D/Hartmann_6D's plain (non-HF/LF) registry
# entries, which deliberately use negate=False (Hartmann(dim=6,
# negate=False).optimal_value), the OPPOSITE sign from best_observed (which
# comes from a negate=True objective). borehole_hf_opt/borehole_lf_opt above
# are computed via _find_optimum_bounded(..., negate=True) -- the natural/
# positive scale, matching best_observed's sign, NOT known_optimal_value's
# expected sign. Storing them un-negated made every regret computation on
# this benchmark wildly wrong (observed: final_regret around -400 to -500
# instead of a small positive number, confirmed directly from the first
# Stage 2 MF-GP-UCB/SF-DRO Borehole_8D runs). Negating them here is the fix.
BENCHMARKS.update({
    "Borehole_8D_HF": {
        "dim": 8,
        "domain_min": _BOREHOLE_DOMAIN_MIN,
        "domain_max": _BOREHOLE_DOMAIN_MAX,
        "known_optimal_value": -borehole_hf_opt,
        "make_objective": lambda: BoreholeFunctionHF(negate=True),
        "cost": 2.0,
        "fidelity": "H",
    },
    "Borehole_8D_LF": {
        "dim": 8,
        "domain_min": _BOREHOLE_DOMAIN_MIN,
        "domain_max": _BOREHOLE_DOMAIN_MAX,
        "known_optimal_value": -borehole_lf_opt,
        "make_objective": lambda: BoreholeFunctionLF(negate=True),
        "cost": 1.0,
        "fidelity": "L",
    },
    # Standalone single-fidelity entry (for SF-DRO, which looks up
    # "Borehole_8D" with no _HF/_LF suffix) -- mirrors Currin_2D/Hartmann_6D,
    # which both have a plain entry alongside their _HF/_LF pair. Simply
    # aliases the HF variant since there's no separate single-fidelity
    # Borehole objective distinct from BoreholeFunctionHF.
    "Borehole_8D": {
        "dim": 8,
        "domain_min": _BOREHOLE_DOMAIN_MIN,
        "domain_max": _BOREHOLE_DOMAIN_MAX,
        "known_optimal_value": -borehole_hf_opt,
        "make_objective": lambda: BoreholeFunctionHF(negate=True),
    },
})


# --- Ackley 10D benchmark pair (Ackley_10D_{HF,LF}) ---
#
# Distinct from the pre-existing standalone "Ackley_10D" entry above (SF-only,
# [-32.768,32.768]^10 bounds, unrelated experiment) -- this pair deliberately
# stays on [0,1]^10 (AckleyFunctionHF/LF's own domain, see synthetic.py),
# matching Currin_2D/Hartmann_6D's convention rather than that entry's or
# Borehole_8D's. Built to test whether MF-DRO's incumbent-freeze pathology is
# specific to narrow-basin benchmarks (Hartmann_6D) or architectural: Ackley
# is smooth and radially symmetric around its optimum rather than built from
# several competing local-optima bumps.
#
# known_optimal_value=0.0 for BOTH HF and LF (not empirically searched like
# Currin_2D_LF/Hartmann_6D_LF's known_optimal_value): AckleyFunctionLF's raw
# (pre-negation) value is raw_ackley(x) + bias*||x-0.5||, a sum of two
# non-negative terms that are BOTH independently minimized (at 0) only at
# x=[0.5]*10 -- the same point as HF's own minimum -- so LF's true optimum is
# analytically 0 at the same location, not merely empirically close to it.
BENCHMARKS.update({
    "Ackley_10D_HF": {
        "dim": 10,
        "domain_min": [0.0] * 10,
        "domain_max": [1.0] * 10,
        "known_optimal_value": 0.0,
        "make_objective": lambda: AckleyFunctionHF(negate=True),
        # cost 10.0 -> 5.0 (user-directed, 2026-08-26): the pitch-talk comparison
        # specifies a 5:1 HF:LF ratio for Ackley. Nothing had been run on this
        # pair, so no prior result is invalidated. Note the four benchmarks now
        # span 2:1 (Borehole), 3:1 (Currin), 5:1 (Ackley), 8:1 (Hartmann).
        "cost": 5.0,
        "fidelity": "H",
    },
    "Ackley_10D_LF": {
        "dim": 10,
        "domain_min": [0.0] * 10,
        "domain_max": [1.0] * 10,
        "known_optimal_value": 0.0,
        "make_objective": lambda: AckleyFunctionLF(negate=True),
        "cost": 1.0,
        "fidelity": "L",
    },
    # Standalone single-fidelity entry for SF-DRO (which looks up a bare
    # benchmark name, no _HF/_LF suffix) -- aliases THIS pair's own HF
    # variant, matching Borehole_8D's identical pattern above. NOT named
    # bare "Ackley_10D": that key is already taken by the older, unrelated
    # standalone Ackley_10D entry earlier in this file ([-32.768,32.768]^10
    # domain, actively used by run_experiment.py's EXP9_BENCHMARKS and
    # several _diag_sf_*.py scripts) -- reusing it here would silently make
    # SF-DRO solve a different problem than MF-DRO/MF-DRO+LFScreen do on
    # "Ackley_10D" in the same Stage 2 comparison table.
    "Ackley_10D_MF": {
        "dim": 10,
        "domain_min": [0.0] * 10,
        "domain_max": [1.0] * 10,
        "known_optimal_value": 0.0,
        "make_objective": lambda: AckleyFunctionHF(negate=True),
    },
})


# --- Basin-width-parametrized Hartmann-6 pairs (Hartmann6D_w{10,03,01}_{HF,LF}) ---
#
# HartmannWidened6D/HartmannWidenedLF6D (src/objectives/synthetic.py) scale
# the FULL Hartmann-6 A matrix by a single alpha_basin scalar to make the
# optimum's basin wider (smaller alpha_basin) or narrower. Built to test
# whether MF-DRO's incumbent-freeze pathology is sensitive to landscape/
# basin width specifically, independent of the cost-ratio/LF-correlation
# mechanism investigated separately (see mfdro_hartmann_cost_ratio results).
#
# IMPORTANT -- optimum is NOT invariant to alpha_basin, despite the
# alpha_basin=1.0 case matching standard Hartmann-6 exactly: scaling the
# whole A matrix by one scalar is not a coordinate reparametrization around
# x*, since Hartmann-6 is a weighted MIXTURE of 4 anisotropic Gaussian bumps
# centered at 4 different P rows, not one bump -- multiplying every bump's
# exponent by the same alpha_basin does not preserve the mixture's argmax or
# peak height in general. Verified numerically before registering these
# benchmarks: f(x_star_standard; alpha_basin=1.0/0.3/0.1) =
# 3.322/4.127/5.683 (evaluated at the FIXED standard-Hartmann-6 optimum
# location), and the true argmax also shifts location as alpha_basin shrinks
# (confirmed via random-sample + L-BFGS-B polish, same methodology as
# _find_optimum_bounded above). So -- mirroring HartmannLF/Currin_2D_LF's
# own established convention for optima with no reliable closed form --
# known_optimal_value AND known_optimal_x are determined EMPIRICALLY per
# alpha_basin below, not assumed to be standard Hartmann-6's published
# x*=[0.2017,...]/f=3.3224 for all three variants.
def _find_optimum_unit_cube(f, d, n=20000, seed=0, n_polish=5):
    torch.manual_seed(seed)
    X = torch.rand(n, d, dtype=torch.float64)
    vals = f(X).reshape(-1)  # already maximization-ready (negate=False here is correct -- see synthetic.py)

    def _neg(x):
        xt = torch.tensor(x, dtype=torch.float64).unsqueeze(0)
        return -f(xt).reshape(-1)[0].item()

    best_val = vals.max().item()
    best_x = X[vals.argmax()].numpy()
    for idx in torch.topk(vals, n_polish).indices.tolist():
        res = minimize(_neg, X[idx].numpy(), bounds=[(0.0, 1.0)] * d, method="L-BFGS-B")
        if -res.fun > best_val:
            best_val = -res.fun
            best_x = res.x
    return best_val, best_x


_HARTMANN_WIDENED_ALPHAS = {"Hartmann6D_w10": 1.0, "Hartmann6D_w03": 0.3, "Hartmann6D_w01": 0.1}
HARTMANN_WIDENED_OPTIMA = {}  # name -> (known_optimal_value, known_optimal_x list) -- used by the sweep worker

for _name, _alpha in _HARTMANN_WIDENED_ALPHAS.items():
    _hf_val, _hf_x = _find_optimum_unit_cube(HartmannWidened6D(alpha_basin=_alpha), d=6)
    _lf_val, _lf_x = _find_optimum_unit_cube(HartmannWidenedLF6D(alpha_basin=_alpha), d=6)
    HARTMANN_WIDENED_OPTIMA[_name] = (_hf_val, _hf_x.tolist())
    print(f"{_name} (alpha_basin={_alpha}): HF optimum={_hf_val:.6f} at x={_hf_x.round(4).tolist()}, "
          f"LF optimum={_lf_val:.6f}")
    # SIGN CONVENTION (same fix as Borehole_8D above, verified via a smoke
    # test on this exact registration -- storing the positive/maximization
    # scale here produced final_regret=-5.39, the same wrong-sign failure
    # mode the Borehole comment describes): mf_dro.py/dro.py's shared regret
    # formula is `-best_observed - known_optimal_value`, and best_observed
    # comes from calling make_objective() at runtime, which is ALREADY on
    # the maximization-ready scale (HartmannWidened6D/LF's evaluate_true is
    # hard-coded positive, with no negate=True/False toggle available -- see
    # synthetic.py). known_optimal_value must therefore be stored as the
    # NEGATIVE of _hf_val/_lf_val (the raw/pre-negation-style scale), mirroring
    # Hartmann_6D's own plain entry (Hartmann(dim=6, negate=False).optimal_value,
    # which is -3.32237, not +3.32237) -- NOT HARTMANN_WIDENED_OPTIMA's own
    # (_hf_val, _hf_x) tuple above, which intentionally stays on the positive
    # scale for the sweep worker's own diagnostics (_diag_xstar, reporting).
    BENCHMARKS[f"{_name}_HF"] = {
        "dim": 6,
        "domain_min": [0.0] * 6,
        "domain_max": [1.0] * 6,
        "known_optimal_value": -_hf_val,
        "make_objective": (lambda a=_alpha: HartmannWidened6D(alpha_basin=a)),
        "cost": 8.0,   # same c_H as standard Hartmann_6D_HF
        "fidelity": "H",
    }
    BENCHMARKS[f"{_name}_LF"] = {
        "dim": 6,
        "domain_min": [0.0] * 6,
        "domain_max": [1.0] * 6,
        "known_optimal_value": -_lf_val,
        "make_objective": (lambda a=_alpha: HartmannWidenedLF6D(alpha_basin=a)),
        "cost": 1.0,   # same c_L as standard Hartmann_6D_LF
        "fidelity": "L",
    }


def get_benchmark(name: str) -> dict:
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {name!r}. Available: {sorted(BENCHMARKS.keys())}")
    return BENCHMARKS[name]
