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
import torch

from botorch.test_functions.synthetic import EggHolder, Shekel, Michalewicz, Hartmann

from src.objectives import Ackley, Rosenbrock, CurrinExpHF, CurrinExpLF, HartmannLF


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


def get_benchmark(name: str) -> dict:
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {name!r}. Available: {sorted(BENCHMARKS.keys())}")
    return BENCHMARKS[name]
