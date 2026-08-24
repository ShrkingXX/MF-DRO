"""
Broader num_epochs ablation: {10, 20, 30, 100} x seeds {42,43,44} x
benchmarks {Hartmann_6D, Currin_2D, Borehole_8D}. Same fixed 50-iteration
horizon (cost_budget=9999, not cost-terminated) and same config convention
as _num_epochs_ablation_worker.py's original single-benchmark seed42 run,
generalized to take benchmark as an argv param and skip-if-done for
resumability under parallel launch.
"""
import sys
import os
import json

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BENCHMARK = sys.argv[1]
num_epochs = int(sys.argv[2])
seed = int(sys.argv[3])

N_ITERS = 50
EXP_NAME = "mfdro_num_epochs_ablation_v2"

OUT_DIR = os.path.join("results", EXP_NAME, "checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"{BENCHMARK}__epochs{num_epochs}__seed{seed}.json")
tag = f"[{BENCHMARK} num_epochs={num_epochs} seed{seed}]"

if os.path.exists(out_path):
    print(f"{tag} SKIPPED (already exists)", flush=True)
    sys.exit(0)

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
    cost_budget=9999,
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
           distinct_regret_values=distinct_regrets, num_epochs=num_epochs,
           benchmark=BENCHMARK, seed=seed)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"{tag} DONE | best_hf={best_hf:.4f} | final_regret={rc[-1]:.4f} | "
      f"lf_frac={lf_frac:.3f} | mean_ls_lf={mean_ls_lf:.4f} | "
      f"mean_ls_delta={mean_ls_delta:.4f} | incumbent_improved_count={n_improved} | "
      f"distinct_regret_values={distinct_regrets}/{len(rc)}", flush=True)
