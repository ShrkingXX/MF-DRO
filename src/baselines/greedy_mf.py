"""
Shared driver for every greedy (DT-free) multi-fidelity MES method.

KO-MES, Additive-MES and SF-MES differ ONLY in their surrogate model and,
consequently, in how the two MES branches are evaluated. Everything else --
the initial design, the cost accounting, the budget-termination rule, the
regret convention, the result schema -- has to be identical for the
cross-method comparison to mean anything, so all of it lives here exactly
once and each method supplies just `_update_model` and `_propose_greedy`.

Extracted from GreedyMFMESOptimizer in mf_baselines.py, which now subclasses
this; verified to reproduce that class's pre-refactor Stage 2 output
bit-for-bit (same regret curve, same fidelity trace) on Currin_2D seed 42.
"""
import numpy as np
import torch

from src.utils.init_design import make_initial_design

DEFAULT_DTYPE = torch.float64


class GreedyMFBase:
    """
    benchmark: a MultiFidelityBenchmark (mf_baselines.py).
    n_initial_hf / n_initial_lf: Song 2019 asymmetric init (3*d HF, 5*d LF).
    cost_budget: POST-INIT cost budget; initialization spending is excluded
        so methods with different init sizes stay comparable on one x-axis.
    use_sequential_init: sequential max-variance design instead of LHS (see
        src/utils/init_design.py). Applied identically to every method in a
        given experiment.
    n_candidates: size of the uniform candidate pool the greedy argmax runs
        over, sampled fresh each iteration across the FULL domain (not
        ROI-filtered), guaranteeing full-domain coverage regardless of which
        basin contains the optimum.
    """

    #: overridden to False by single-fidelity methods, which never query LF
    #: and therefore never draw an LF initial design either.
    uses_lf = True

    def __init__(self, benchmark, n_initial_hf=5, n_initial_lf=5, seed=0,
                 cost_budget=None, use_sequential_init=False,
                 n_candidates=200):
        self.benchmark = benchmark
        self.d = benchmark.dim
        self.bounds = benchmark.bounds
        self.c_L = benchmark.c_L
        self.c_H = benchmark.c_H
        self.n_initial_hf = n_initial_hf
        self.n_initial_lf = n_initial_lf
        self.seed = seed
        self.cost_budget = cost_budget
        self.use_sequential_init = use_sequential_init
        self.n_candidates = n_candidates

        self.data_hf_x, self.data_hf_y = [], []
        self.data_lf_x, self.data_lf_y = [], []
        self.initial_hf_values = []

    # ---- hooks each concrete method implements ----

    def _update_model(self):
        """Refit the surrogate on all accumulated data."""
        raise NotImplementedError

    def _propose_greedy(self, X_cand):
        """
        Return (x, ell) maximizing InfoGain(x, ell)/c(ell) over X_cand,
        with ell 0=LF, 1=HF.
        """
        raise NotImplementedError

    # ---- shared machinery ----

    def _init_points(self, n, seed_offset):
        return make_initial_design(
            self.bounds, self.d, n, self.seed, seed_offset,
            use_sequential_init=self.use_sequential_init,
        )

    def _sample_candidates(self):
        X = torch.rand(self.n_candidates, self.d, dtype=DEFAULT_DTYPE)
        return self.bounds[0] + (self.bounds[1] - self.bounds[0]) * X

    def _sample_initial(self):
        torch.manual_seed(self.seed)
        for x in self._init_points(self.n_initial_hf, seed_offset=0):
            self.data_hf_x.append(x)
            self.data_hf_y.append(self.benchmark.evaluate(x, 'H'))
        self.initial_hf_values = list(self.data_hf_y)

        if not self.uses_lf:
            return
        # Distinct seed offset so LF initial points aren't identical to HF's.
        for x in self._init_points(self.n_initial_lf, seed_offset=1):
            self.data_lf_x.append(x)
            self.data_lf_y.append(self.benchmark.evaluate(x, 'L'))

    def run(self, bo_iterations):
        """
        cost_curve is POST-INIT cost (starts near 0, excludes initialization
        spending). Terminates once post_init_cost reaches self.cost_budget;
        bo_iterations is a safety cap only.
        """
        self._sample_initial()
        n_lf_init = len(self.data_lf_y)
        cumulative_cost = (self.n_initial_hf * self.c_H + n_lf_init * self.c_L)
        post_init_cost = 0.0
        budget = self.cost_budget if self.cost_budget is not None else float('inf')

        regret_curve, cost_curve, fidelity_trace = [], [], []
        x_t_trace, y_t_trace = [], []

        for _ in range(bo_iterations):
            if post_init_cost >= budget:
                break
            self._update_model()
            x_t, ell_t = self._propose_greedy(self._sample_candidates())

            if ell_t == 1:
                y_t = self.benchmark.evaluate(x_t, 'H')
                self.data_hf_x.append(x_t)
                self.data_hf_y.append(y_t)
                cost = self.c_H
            else:
                y_t = self.benchmark.evaluate(x_t, 'L')
                self.data_lf_x.append(x_t)
                self.data_lf_y.append(y_t)
                cost = self.c_L
            cumulative_cost += cost
            post_init_cost += cost

            # known_optimal_value_hf is on the RAW (pre-negation) scale
            # (benchmarks.py's convention); data_hf_y holds NEGATED
            # (maximization-ready) values, so best_hf is negated back before
            # comparing -- matching dro.py's maximize-mode regret convention.
            best_hf = max(self.data_hf_y)
            regret_curve.append(-best_hf - self.benchmark.known_optimal_value_hf)
            cost_curve.append(post_init_cost)
            fidelity_trace.append(ell_t)
            x_t_trace.append(x_t.tolist())
            y_t_trace.append(y_t)

        return {
            'regret_curve': regret_curve,
            'cost_curve': cost_curve,
            'fidelity_trace': fidelity_trace,
            'x_t_trace': x_t_trace,
            'y_t_trace': y_t_trace,
            'lf_fraction': sum(1 for e in fidelity_trace if e == 0)
                / max(len(fidelity_trace), 1),
            'initial_hf_values': self.initial_hf_values,
            'final_rho': self.final_rho(),
        }

    def final_rho(self):
        """
        The surrogate's fitted fidelity-correlation parameter at the end of
        the run, or None for surrogates that have no such scalar. Reported in
        the comparison table (KO's rho is the quantity the KO-vs-additive
        contrast is about).
        """
        return None


def cost_normalized_argmax(mes_lf_arr, mes_hf_arr, c_L, c_H, X_cand):
    """
    Joint argmax over (candidate, fidelity) of InfoGain/cost -- the selection
    rule shared by every method here. Returns (x, ell) with ell 0=LF, 1=HF.
    """
    scores = np.stack([mes_lf_arr / c_L, mes_hf_arr / c_H], axis=1)  # [N, 2]
    flat_best = scores.reshape(-1).argmax()
    return X_cand[flat_best // 2], int(flat_best % 2)
