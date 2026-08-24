"""
D_nodt + Greedy-MES hybrid: location proposed by 5-restart UCB gradient
ascent (no DT, no MES for location) on a single-member KO-GP; fidelity
decided by Greedy-MES's own cost-normalized joint MF-MES rule
(compute_joint_mf_mes), evaluated AT that chosen location.

Subclasses GreedyMFBase (src/baselines/greedy_mf.py) directly, reusing its
exact init sampling / cost accounting / termination / regret convention --
the SAME machinery that produced the existing MF-DRO (mean regret 4.564)
and Greedy-MES (mean regret 5.729) results in results/mfdro_ackley_test/.
Only _propose_greedy's LOCATION mechanism differs from GreedyMFMESOptimizer;
_update_model is identical (single-member KO-GP, use_dkl=False).

UCB gradient ascent reuses the exact mu_H/sigma_H reimplementation pattern
from DirectMFRegretOptimization._refine_proposal in mf_dro.py (calling
ko.gp_lf.posterior/ko.gp_delta.posterior directly, NOT ko.hf_posterior,
which wraps its body in torch.no_grad() and silently kills the gradient
back to x -- confirmed in that method's own docstring).

Benchmark: Ackley_10D, seeds 42/43/44, cost_budget=500, initial_hf=
initial_lf=30 -- matching _ackley_test_worker.py's exact protocol.
"""
import sys
import os
import json

import torch
import gpytorch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

seed = int(sys.argv[1])

BENCHMARK = "Ackley_10D"
COST_BUDGET = 500
INITIAL_HF = 30
INITIAL_LF = 30
EXP_NAME = "mfdro_ackley_test"
CHECKPOINTS = [50, 150, 500]
UCB_BETA = 2.0
REFINE_STEPS = 30
REFINE_LR = 0.05
N_RESTARTS = 5

EXP_DIR = os.path.join("results", EXP_NAME, "checkpoints")
os.makedirs(EXP_DIR, exist_ok=True)
out_path = os.path.join(EXP_DIR, f"D_nodt+Greedy-MES__{BENCHMARK}__seed{seed}.json")
tag = f"[D_nodt+Greedy-MES {BENCHMARK} seed{seed}]"

if os.path.exists(out_path):
    print(f"{tag} SKIPPED (already exists)", flush=True)
    sys.exit(0)

print(f"{tag} Starting (cost_budget={COST_BUDGET}, initial_hf={INITIAL_HF}, "
      f"initial_lf={INITIAL_LF}, n_restarts={N_RESTARTS})", flush=True)

from src.baselines.greedy_mf import GreedyMFBase
from src.baselines.mf_baselines import MultiFidelityBenchmark
from src.models.ko_gp import KennedyOHaganGP
from src.policy.mf_dro import compute_joint_mf_mes

DEFAULT_DTYPE = torch.float64


def _mf_posterior(ko, x_pt):
    """mu_H(x), sigma_H(x), differentiable w.r.t. x -- same reimplementation
    DirectMFRegretOptimization._refine_proposal uses (NOT ko.hf_posterior,
    which is wrapped in torch.no_grad())."""
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


def _ucb_value(ko, x_pt, beta):
    mu, sigma = _mf_posterior(ko, x_pt)
    return (mu + beta * sigma).reshape(())


def _refine_ucb_single(ko, x_init, beta, steps, lr, lo, hi):
    with torch.enable_grad():
        x = x_init.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            val = _ucb_value(ko, x, beta)
            (-val).backward()
            opt.step()
            with torch.no_grad():
                x.clamp_(lo, hi)
    return x.detach()


class DnodtGreedyMES(GreedyMFBase):
    def __init__(self, benchmark, **kwargs):
        super().__init__(benchmark, **kwargs)
        self.ko_ensemble = [KennedyOHaganGP(d=self.d, dkl_threshold=float('inf'))]

    def _update_model(self):
        X_hf = torch.stack(self.data_hf_x)
        Y_hf = torch.tensor(self.data_hf_y, dtype=DEFAULT_DTYPE)
        if self.data_lf_x:
            X_lf = torch.stack(self.data_lf_x)
            Y_lf = torch.tensor(self.data_lf_y, dtype=DEFAULT_DTYPE)
        else:
            X_lf = X_hf[:0]
            Y_lf = Y_hf[:0]
        self.ko_ensemble[0].fit(X_lf, Y_lf, X_hf, Y_hf, self.bounds)

    def _propose_greedy(self, X_cand):
        # LOCATION: 5-restart UCB gradient ascent, no DT, no MES.
        ko = self.ko_ensemble[0]
        lo, hi = self.bounds[0], self.bounds[1]
        starts = [lo + (hi - lo) * torch.rand(self.d, dtype=DEFAULT_DTYPE) for _ in range(N_RESTARTS)]
        best_x, best_val = None, -float('inf')
        for s in starts:
            x_ref = _refine_ucb_single(ko, s, UCB_BETA, REFINE_STEPS, REFINE_LR, lo, hi)
            with torch.no_grad():
                v = _ucb_value(ko, x_ref, UCB_BETA).item()
            if v > best_val:
                best_val, best_x = v, x_ref
        # FIDELITY: Greedy-MES's own cost-normalized joint MF-MES rule,
        # evaluated at the single chosen location.
        _, ell_t, _ = compute_joint_mf_mes(ko, best_x.unsqueeze(0), self.c_H, self.c_L, K=10)
        return best_x, ell_t

    def final_rho(self):
        return float(self.ko_ensemble[0].rho.item())


def regret_at_cost(cost_curve, regret_curve, c_ref):
    idx = None
    for i, c in enumerate(cost_curve):
        if c <= c_ref:
            idx = i
        else:
            break
    return regret_curve[idx] if idx is not None else (regret_curve[0] if regret_curve else float('nan'))


def incumbent_improved_count(regret_curve):
    return sum(1 for i in range(1, len(regret_curve)) if regret_curve[i] < regret_curve[i - 1] - 1e-12)


torch.manual_seed(seed)
bench = MultiFidelityBenchmark(BENCHMARK)
opt = DnodtGreedyMES(bench, n_initial_hf=INITIAL_HF, n_initial_lf=INITIAL_LF,
                      seed=seed, cost_budget=COST_BUDGET)
result = opt.run(bo_iterations=500)

rc, cc = result["regret_curve"], result["cost_curve"]
n_improved = incumbent_improved_count(rc)
checkpoint_regrets = {c: regret_at_cost(cc, rc, c) for c in CHECKPOINTS}

out = dict(result, incumbent_improved_count=n_improved,
           checkpoint_regrets=checkpoint_regrets, method="D_nodt+Greedy-MES", seed=seed)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"{tag} DONE | regret@50={checkpoint_regrets[50]:.4f} | "
      f"regret@150={checkpoint_regrets[150]:.4f} | "
      f"regret@500={checkpoint_regrets[500]:.4f} | "
      f"lf_fraction={result['lf_fraction']:.3f} | "
      f"incumbent_improved_count={n_improved}", flush=True)
