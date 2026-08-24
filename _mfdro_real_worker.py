"""
MF-DRO-Real vs MF-DRO-Simulated: does building rollout training data from
REAL function evaluations (instead of sampling y from the GP's own
posterior) fix the Hartmann_6D incumbent freeze? Same Stage 2 v3 config for
both (initial_hf=18, initial_lf=30, cost_budget=800, num_epochs=10,
dkl_threshold=9999, bes_delta=0.0, rollout_length=8) -- only
use_real_rollout_queries toggles. MF-DRO-Real is a diagnostic-only oracle
ablation (unlimited free real-function access for rollout training data);
never a legitimate production config -- see simulate_mf_trajectory's
docstring in src/policy/mf_dro.py.
"""
import sys
import os
import json

variant = sys.argv[1]   # Simulated | Real
seed = int(sys.argv[2])

BENCHMARK = "Hartmann_6D"
COST_BUDGET = 800
INITIAL_HF = 18
INITIAL_LF = 30
NUM_EPOCHS = 10
EXP_NAME = "mfdro_real_vs_simulated"

OUT_DIR = os.path.join("results", EXP_NAME, "checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"{variant}__{BENCHMARK}__seed{seed}.json")
tag = f"[{variant} {BENCHMARK} seed{seed}]"

if os.path.exists(out_path):
    print(f"{tag} SKIPPED (already exists)", flush=True)
    sys.exit(0)

print(f"{tag} Starting (cost_budget={COST_BUDGET}, "
      f"initial_hf={INITIAL_HF}, initial_lf={INITIAL_LF})", flush=True)

import torch
import numpy as np
from benchmarks import get_benchmark
from dro_runner import run_mf_single_seed

torch.set_default_dtype(torch.float64)

torch.manual_seed(seed)
np.random.seed(seed)
result = run_mf_single_seed(
    EXP_NAME, BENCHMARK, variant, seed,
    bo_iterations=500,
    num_epochs=NUM_EPOCHS,
    minimum_hf_fraction=0.25,
    real_hf_warmup=2,
    cost_budget=COST_BUDGET,
    initial_hf=INITIAL_HF,
    initial_lf=INITIAL_LF,
    dkl_threshold=9999,
    bes_delta=0.0,
    rollout_length=8,
    use_real_rollout_queries=(variant == "Real"),
)

rc = result['hf_regret_curve']
final_regret = rc[-1]
n_improved = sum(1 for i in range(1, len(rc)) if rc[i] < rc[i - 1] - 1e-12)
distinct = len(set(f"{r:.6f}" for r in rc))
lf_frac = result['lf_fraction']

out = dict(result, variant=variant, seed=seed, final_regret=final_regret,
           incumbent_improved_count=n_improved, distinct_regret_values=distinct)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"{tag} DONE | final_regret={final_regret:.4f} | n_iters={len(rc)} | "
      f"incumbent_improved_count={n_improved} | distinct={distinct}/{len(rc)} | "
      f"lf_fraction={lf_frac:.3f}", flush=True)
