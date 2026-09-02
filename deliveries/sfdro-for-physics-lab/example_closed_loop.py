"""
example_closed_loop.py -- the other mode: a simulated objective the code can
call itself, and the plain-BO baseline to compare against.

Use this when the "experiment" is a piece of software (a physics simulation, a
surrogate fitted to past data, a benchmark function). Both optimizers here take
a callable and run the whole campaign without you in the loop.

    python example_closed_loop.py

`run_naive_bo` (naive_bo.py) is the basic BO algorithm: one GP, expected
improvement, one proposal per iteration. It is the reference point SF-DRO is
measured against -- start by reading that file, it is 140 lines and contains no
surprises. SF-DRO replaces the acquisition step with the rollout + Decision
Transformer machinery described in the README.
"""
import numpy as np
import torch
from botorch.test_functions.synthetic import SyntheticTestFunction

from naive_bo import run_naive_bo
from sfdro import build_config
from src.policy.dro import DirectRegretOptimization

torch.set_default_dtype(torch.float64)

DIM = 3
LO = [0.0, 0.2, 4.0]
HI = [5.0, 3.0, 4.5]


class SimulatedCell(SyntheticTestFunction):
    """
    Both optimizers expect a BoTorch-style callable that MAXIMIZES. Wrap your
    simulator in a class like this one: `evaluate_true` returns the quantity you
    care about, already signed so that larger is better.
    """
    def __init__(self):
        self._bounds = list(zip(LO, HI))
        self.dim = DIM
        super().__init__(noise_std=None, negate=False, bounds=self._bounds)

    def evaluate_true(self, X):
        X = X.reshape(-1, DIM)
        additive, c_rate, v_cut = X[:, 0], X[:, 1], X[:, 2]
        return (900.0
                - 60.0 * (additive - 2.0) ** 2
                - 180.0 * (c_rate - 0.8) ** 2
                - 4000.0 * (v_cut - 4.2) ** 2
                + 40.0 * torch.sin(3.0 * additive) * torch.cos(2.0 * c_rate))


def main():
    objective = SimulatedCell()
    n_init, n_iter, seed = 8, 10, 42

    # --- basic BO -----------------------------------------------------------
    print("=== basic BO (GP + expected improvement) ===")
    res = run_naive_bo(
        objective_function=objective,
        domain_min=LO, domain_max=HI, dim=DIM, seed=seed,
        max_iterations=n_iter, initial_points=n_init,
        iter_callback=lambda t, r, b, dt: print(f"  iter {t:2d}: best = {b:8.1f}  ({dt:.1f}s)"),
    )
    print(f"  final best = {res['best_observed'][-1]:.1f}\n")

    # --- SF-DRO -------------------------------------------------------------
    print("=== SF-DRO ===")
    cfg = build_config(
        input_dim=DIM, domain_min=LO, domain_max=HI, seed=seed,
        n_initial=n_init, max_iterations=n_iter,
        gp_num_models=5, rollouts_per_iter=20, rollout_length=4,   # cheap demo settings
        verbose=False,
    )
    dro = DirectRegretOptimization(cfg, objective)
    # Draws its own LHS initial design, then runs n_iter propose/evaluate steps.
    result = dro.run_optimization()
    print(f"  final best = {result['best_y']:.1f}")
    print(f"  at x = {np.round(result['best_x'], 4)}")


if __name__ == "__main__":
    main()
