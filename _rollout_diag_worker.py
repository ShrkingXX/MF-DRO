"""
Rollout-exploration diagnostic: does simulate_mf_trajectory's rollout data
(what the DT is actually trained on) explore new regions of input space, or
collapse near the incumbent immediately after initialization? Run on oracle
init, Hartmann_6D, seed=42, num_epochs=10, exactly 10 BO iterations
(bo_iterations=10, cost_budget=9999 so iteration count -- not cost -- is
the only stop condition, matching this session's established diagnostic
convention).

Reuses _oracle_init_worker.py's oracle_init/monkeypatch pattern so the
initial data is seeded near x* exactly like the oracle-init test that
motivated this diagnostic.
"""
import sys
import os
import types

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization
from src.utils.init_design import lhs_design

BENCHMARK = "Hartmann_6D"
SEED = 42
N_ITERS = 10
EXP_NAME = "mfdro_rollout_diag"

torch.set_default_dtype(torch.float64)
hf_spec = get_benchmark(BENCHMARK + "_HF")
lf_spec = get_benchmark(BENCHMARK + "_LF")
f_hf = hf_spec["make_objective"]()
f_lf = lf_spec["make_objective"]()
bounds = torch.tensor([hf_spec["domain_min"], hf_spec["domain_max"]], dtype=torch.float64)

X_STAR = torch.tensor(
    [0.2017, 0.1500, 0.4769, 0.2753, 0.3116, 0.6573], dtype=torch.float64
)
RADIUS = 0.15
N_NEAR = 5


def oracle_init(bounds, d, n_hf, n_lf, seed, radius=RADIUS, n_near=N_NEAR):
    torch.manual_seed(seed)
    near = X_STAR + radius * (torch.rand(n_near, d, dtype=torch.float64) - 0.5)
    near = near.clamp(0.0, 1.0)
    n_near_hf = n_near // 2
    n_near_lf = n_near - n_near_hf
    lhs_hf = lhs_design(bounds, d, n_hf - n_near_hf, seed, seed_offset=0)
    lhs_lf = lhs_design(bounds, d, n_lf - n_near_lf, seed, seed_offset=1)
    near_scaled = bounds[0] + (bounds[1] - bounds[0]) * near
    X_hf = torch.cat([near_scaled[:n_near_hf], lhs_hf])
    X_lf = torch.cat([near_scaled[n_near_hf:], lhs_lf])
    return X_hf, X_lf


def _oracle_sample_initial_points(self):
    n_hf = self.config.initial_hf
    n_lf = self.config.initial_lf
    X_hf_init, X_lf_init = oracle_init(self.bounds, self.d, n_hf, n_lf, self.config.seed)
    for x in X_hf_init:
        y = self.f_hf(x.unsqueeze(0)).reshape(-1)[0].item()
        self.data_hf_x.append(x)
        self.data_hf_y.append(y)
        self.cumulative_cost += self.c_H
    self.initial_hf_values = list(self.data_hf_y)
    for x in X_lf_init:
        y = self.f_lf(x.unsqueeze(0)).reshape(-1)[0].item()
        self.data_lf_x.append(x)
        self.data_lf_y.append(y)
        self.cumulative_cost += self.c_L


torch.manual_seed(SEED)
np.random.seed(SEED)
config = _build_mf_dro_config(
    EXP_NAME, BENCHMARK, "ORACLE-diag", SEED,
    bo_iterations=N_ITERS, num_epochs=10,
    minimum_hf_fraction=0.25, real_hf_warmup=2,
    cost_budget=9999, initial_hf=30, initial_lf=30,
    dkl_threshold=9999, bes_delta=0.0,
)
config.seed = SEED
mf = DirectMFRegretOptimization(config, f_hf, f_lf, bounds)
mf._sample_initial_points = types.MethodType(_oracle_sample_initial_points, mf)
mf._diag_xstar = X_STAR

result = mf.run()

rc = result["hf_regret_curve"]
print(f"\n[SUMMARY] best_hf={max(mf.data_hf_y):.4f} final_regret={rc[-1]:.4f} "
      f"n_iters={len(rc)}", flush=True)
