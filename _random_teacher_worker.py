"""
Random teacher test: does breaking rollout-teacher state->action
determinism (compute_joint_mf_mes always picks the SAME x_tau/ell_tau
given a GP state, regardless of what RTG the rollout eventually earns)
let RTG discriminate lucky/unlucky rollouts, and does that let the DT
learn a working RTG-conditioned policy? Hartmann_6D only, seeds 42/43/44,
cost_budget=400, num_epochs=10, oracle init (5 HF near x* + 25 LHS HF,
5 LF near x* + 25 LHS LF -- distinct near-x* draws per fidelity, not a
shared 5-point pool split between them), rollout_policy="random" (new
teacher branch in simulate_mf_trajectory: uniform-random candidate,
independent of GP state, ell_tau Bernoulli(0.25) -- see mf_dro.py).

Reports, per seed: incumbent_improved_count, final_regret, n_hf_total
(real HF queries during the BO loop, initial-sample points excluded).
Per iteration: corr(rollout_rtg, true_f_hf) via the existing
ROLLOUT-DIAG/RTG-DIAG hook (gated behind _diag_xstar, unchanged from the
earlier MES-teacher run). At iterations 10/25/39: RTG sensitivity probe
(5 RTG multipliers at a fixed real state) via the _diag_probe_iters hook.
"""
import sys
import os
import json

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

seed = int(sys.argv[1])

BENCHMARK = "Hartmann_6D"
COST_BUDGET = 400
NUM_EPOCHS = 10
INITIAL_HF = 30
INITIAL_LF = 30
RADIUS = 0.15
N_NEAR_EACH = 5
PROBE_ITERS = {10, 25, 39}
EXP_NAME = "mfdro_random_teacher_test"

OUT_DIR = os.path.join("results", EXP_NAME, "checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"{BENCHMARK}__seed{seed}.json")
tag = f"[random-teacher {BENCHMARK} seed{seed}]"

if os.path.exists(out_path):
    print(f"{tag} SKIPPED (already exists)", flush=True)
    sys.exit(0)

from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization
from src.utils.init_design import lhs_design

torch.set_default_dtype(torch.float64)
hf_spec = get_benchmark(BENCHMARK + "_HF")
lf_spec = get_benchmark(BENCHMARK + "_LF")
f_hf = hf_spec["make_objective"]()
f_lf = lf_spec["make_objective"]()
bounds = torch.tensor([hf_spec["domain_min"], hf_spec["domain_max"]], dtype=torch.float64)

X_STAR = torch.tensor(
    [0.2017, 0.1500, 0.4769, 0.2753, 0.3116, 0.6573], dtype=torch.float64
)


def oracle_init(bounds, d, n_hf, n_lf, seed, radius=RADIUS, n_near_each=N_NEAR_EACH):
    torch.manual_seed(seed)
    near_hf = (X_STAR + radius * (torch.rand(n_near_each, d, dtype=torch.float64) - 0.5)).clamp(0, 1)
    near_lf = (X_STAR + radius * (torch.rand(n_near_each, d, dtype=torch.float64) - 0.5)).clamp(0, 1)
    lhs_hf = lhs_design(bounds, d, n_hf - n_near_each, seed, seed_offset=0)
    lhs_lf = lhs_design(bounds, d, n_lf - n_near_each, seed, seed_offset=1)
    near_hf_scaled = bounds[0] + (bounds[1] - bounds[0]) * near_hf
    near_lf_scaled = bounds[0] + (bounds[1] - bounds[0]) * near_lf
    X_hf = torch.cat([near_hf_scaled, lhs_hf])
    X_lf = torch.cat([near_lf_scaled, lhs_lf])
    assert X_hf.shape[0] == n_hf and X_lf.shape[0] == n_lf
    return X_hf, X_lf


import types


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


torch.manual_seed(seed)
np.random.seed(seed)
config = _build_mf_dro_config(
    EXP_NAME, BENCHMARK, "random-teacher", seed,
    bo_iterations=500, num_epochs=NUM_EPOCHS,
    minimum_hf_fraction=0.25, real_hf_warmup=2,
    cost_budget=COST_BUDGET, initial_hf=INITIAL_HF, initial_lf=INITIAL_LF,
    dkl_threshold=9999, bes_delta=0.0, rollout_length=8,
    rollout_policy="random",
)
config.seed = seed
mf = DirectMFRegretOptimization(config, f_hf, f_lf, bounds)
mf._sample_initial_points = types.MethodType(_oracle_sample_initial_points, mf)
mf._diag_xstar = X_STAR
mf._diag_probe_iters = PROBE_ITERS

result = mf.run()

rc = result["hf_regret_curve"]
n_improved = sum(1 for i in range(1, len(rc)) if rc[i] < rc[i - 1] - 1e-12)
n_hf_total = len(mf.data_hf_x) - INITIAL_HF

out = dict(result, seed=seed, final_regret=rc[-1],
           incumbent_improved_count=n_improved, n_hf_total=n_hf_total,
           x_star=X_STAR.tolist())
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n{tag} DONE incumbent_improved_count={n_improved} "
      f"final_regret={rc[-1]:.4f} n_hf_total={n_hf_total} "
      f"n_iters={len(rc)}", flush=True)
