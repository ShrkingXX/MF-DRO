"""
Controlled A/B: num_epochs=10 vs num_epochs=100, everything else identical
(same seed, same benchmark, same fixed 50-iteration horizon -- NOT
cost-budget-terminated, to keep both arms running the exact same number of
BO iterations so the comparison isn't confounded by cost-accumulation
differences). Replicates _diag_gp_calibration.py's exact successful config
(bo_iterations=50, cost_budget=9999, dkl_threshold=9999, seed=42) for the
num_epochs=10 arm, and changes ONLY num_epochs for the other arm.
"""
import sys
import os
import json

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

num_epochs = int(sys.argv[1])   # 10 or 100
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

BENCHMARK = "Hartmann_6D"
N_ITERS = 50
EXP_NAME = "mfdro_num_epochs_ablation"

OUT_DIR = os.path.join("results", EXP_NAME, "checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"epochs{num_epochs}__{BENCHMARK}__seed{seed}.json")
tag = f"[num_epochs={num_epochs} seed{seed}]"

from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization
from src.models.ko_gp import DeepKernel

torch.set_default_dtype(torch.float64)
hf_spec = get_benchmark(BENCHMARK + "_HF")
lf_spec = get_benchmark(BENCHMARK + "_LF")
f_hf = hf_spec["make_objective"]()
f_lf = lf_spec["make_objective"]()
bounds = torch.tensor([hf_spec["domain_min"], hf_spec["domain_max"]], dtype=torch.float64)

torch.manual_seed(seed)
np.random.seed(seed)
config = _build_mf_dro_config(
    EXP_NAME, BENCHMARK, f"epochs{num_epochs}", seed,
    bo_iterations=N_ITERS, num_epochs=num_epochs,
    minimum_hf_fraction=0.25, real_hf_warmup=2,
    cost_budget=9999,  # fixed N_ITERS, not cost-terminated -- matches the
    # original successful diagnostic's exact termination convention, so
    # BOTH arms run exactly 50 BO iterations regardless of fidelity mix.
    initial_hf=30, initial_lf=30,
    dkl_threshold=9999,
)
config.seed = seed
mf = DirectMFRegretOptimization(config, f_hf, f_lf, bounds)
result = mf.run()

def _mean_lengthscale(covar_module):
    scale_kernel = covar_module.base_kernel if isinstance(covar_module, DeepKernel) else covar_module
    return scale_kernel.base_kernel.lengthscale.detach().mean().item()

ko0 = mf.ko_ensemble[0]
mean_ls_lf = _mean_lengthscale(ko0.gp_lf.covar_module)
mean_ls_delta = _mean_lengthscale(ko0.gp_delta.covar_module)

rc = result['hf_regret_curve']
best_hf = max(mf.data_hf_y)
lf_frac = result['lf_fraction']
n_improved = sum(1 for i in range(1, len(rc)) if rc[i] < rc[i - 1] - 1e-12)
distinct_regrets = len(set(f"{r:.6f}" for r in rc))

out = dict(result, mean_ls_lf=mean_ls_lf, mean_ls_delta=mean_ls_delta,
           best_hf=best_hf, incumbent_improved_count=n_improved,
           distinct_regret_values=distinct_regrets, num_epochs=num_epochs)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"{tag} DONE | best_hf={best_hf:.4f} | final_regret={rc[-1]:.4f} | "
      f"lf_frac={lf_frac:.3f} | mean_ls_lf={mean_ls_lf:.4f} | "
      f"mean_ls_delta={mean_ls_delta:.4f} | incumbent_improved_count={n_improved} | "
      f"distinct_regret_values={distinct_regrets}/{len(rc)}", flush=True)
