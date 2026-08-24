"""
Oracle Initialization Test: does the incumbent freeze come from bad luck in
random LHS initialization, or from something deeper in the DT/rollout
pipeline? RANDOM (current pipeline, unmodified) vs ORACLE (n_near points
seeded near the TRUE known optimum x*, rest standard LHS) on Hartmann_6D,
same Stage-2-adjacent config as the earlier epochs/candidate-scoring pilots
(flat initial_hf=initial_lf=30, cost_budget=240, num_epochs=10,
dkl_threshold=9999, bes_delta=0.0). x* is oracle information a real BO run
never has -- this is diagnostic-only, never a real init strategy, so it is
NOT wired into mf_dro.py's config-driven pipeline; it monkeypatches the
instance's _sample_initial_points instead of touching the real pipeline.

oracle_init below fixes a point-count bug in the given pseudocode: splitting
n_near total near-x* points in half (near[:n_near//2] to HF, the rest to
LF) while independently subtracting the FULL n_near from BOTH n_hf's and
n_lf's LHS counts undercounts each fidelity's total by n_near - n_near//2
(HF) / n_near//2 (LF) -- e.g. n_hf=n_lf=30, n_near=5 lands on 27 HF + 28 LF
instead of 30/30. Fixed by subtracting only each fidelity's OWN share of
n_near from its LHS count.
"""
import sys
import os
import json
import types

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

variant = sys.argv[1]   # RANDOM | ORACLE
seed = int(sys.argv[2])

BENCHMARK = "Hartmann_6D"
COST_BUDGET = 240
NUM_EPOCHS = 10
INITIAL_HF = 30
INITIAL_LF = 30
RADIUS = 0.15
N_NEAR = 5
EXP_NAME = "mfdro_oracle_init_test"

OUT_DIR = os.path.join("results", EXP_NAME, "checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"{variant}__{BENCHMARK}__seed{seed}.json")
tag = f"[{variant} {BENCHMARK} seed{seed}]"

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

from botorch.test_functions.synthetic import Hartmann
X_STAR = Hartmann(dim=6, negate=True).optimizers[0].to(torch.float64)
print(f"{tag} x_star={X_STAR.tolist()}", flush=True)


def oracle_init(bounds, d, n_hf, n_lf, seed, radius=RADIUS, n_near=N_NEAR):
    torch.manual_seed(seed)
    near = X_STAR + radius * (torch.rand(n_near, d, dtype=torch.float64) - 0.5)
    near = near.clamp(0.0, 1.0)

    n_near_hf = n_near // 2
    n_near_lf = n_near - n_near_hf

    lhs_hf = lhs_design(bounds, d, n_hf - n_near_hf, seed, seed_offset=0)
    lhs_lf = lhs_design(bounds, d, n_lf - n_near_lf, seed, seed_offset=1)

    # near[] is in [0,1]^d; rescale to the actual domain (no-op for
    # Hartmann_6D, whose domain already is [0,1]^6, but keeps this correct
    # if ever reused on a non-unit-cube benchmark).
    near_scaled = bounds[0] + (bounds[1] - bounds[0]) * near

    X_hf = torch.cat([near_scaled[:n_near_hf], lhs_hf])
    X_lf = torch.cat([near_scaled[n_near_hf:], lhs_lf])
    assert X_hf.shape[0] == n_hf and X_lf.shape[0] == n_lf
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


torch.manual_seed(seed)
np.random.seed(seed)
config = _build_mf_dro_config(
    EXP_NAME, BENCHMARK, variant, seed,
    bo_iterations=500, num_epochs=NUM_EPOCHS,
    minimum_hf_fraction=0.25, real_hf_warmup=2,
    cost_budget=COST_BUDGET, initial_hf=INITIAL_HF, initial_lf=INITIAL_LF,
    dkl_threshold=9999, bes_delta=0.0,
)
config.seed = seed
mf = DirectMFRegretOptimization(config, f_hf, f_lf, bounds)

if variant == "ORACLE":
    mf._sample_initial_points = types.MethodType(_oracle_sample_initial_points, mf)

result = mf.run()

rc = result["hf_regret_curve"]
final_regret = rc[-1]
n_improved = sum(1 for i in range(1, len(rc)) if rc[i] < rc[i - 1] - 1e-12)
distinct = len(set(f"{r:.6f}" for r in rc))
lf_frac = result["lf_fraction"]

out = dict(result, variant=variant, seed=seed, final_regret=final_regret,
           incumbent_improved_count=n_improved, distinct_regret_values=distinct,
           x_star=X_STAR.tolist())
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"{tag} DONE final_regret={final_regret:.4f} n_iters={len(rc)} "
      f"n_improved={n_improved} distinct={distinct}/{len(rc)} "
      f"lf_fraction={lf_frac:.3f}", flush=True)
