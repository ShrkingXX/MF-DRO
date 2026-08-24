"""
Quick 2-variant test: does raising initial_hf (cold-start HF data quantity)
fix the freeze, or is num_epochs/RNG-driven collapse independent of how much
initial HF data is available? Hartmann_6D, seeds 42/43/44, cost_budget=240,
num_epochs=10 (per the num_epochs ablation's best-case setting), everything
else matching Stage 2 v3 (lognormal prior + noise_lb=1e-2 are ko_gp.py's own
defaults now, diverse init / raw KO state are mf_dro.py's own defaults,
dkl_threshold=9999, rollout_length=8 default, bes_delta=0.0 default).

VARIANT A: initial_hf=30, initial_lf=30 (current/reference)
VARIANT B: initial_hf=60, initial_lf=30
"""
import sys
import os
import json

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

variant = sys.argv[1]   # A | B
seed = int(sys.argv[2])

BENCHMARK = "Hartmann_6D"
COST_BUDGET = 240
NUM_EPOCHS = 10
INITIAL_HF = 30 if variant == "A" else 60
INITIAL_LF = 30
EXP_NAME = "mfdro_init_hf_test"

OUT_DIR = os.path.join("results", EXP_NAME, "checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"variant{variant}__{BENCHMARK}__seed{seed}.json")
tag = f"[VARIANT {variant} seed{seed}]"

if os.path.exists(out_path):
    print(f"{tag} SKIPPED (already exists)", flush=True)
    sys.exit(0)

from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization

torch.set_default_dtype(torch.float64)
hf_spec = get_benchmark(BENCHMARK + "_HF")
lf_spec = get_benchmark(BENCHMARK + "_LF")
f_hf = hf_spec["make_objective"]()
f_lf = lf_spec["make_objective"]()
bounds = torch.tensor([hf_spec["domain_min"], hf_spec["domain_max"]], dtype=torch.float64)

torch.manual_seed(seed)
np.random.seed(seed)
config = _build_mf_dro_config(
    EXP_NAME, BENCHMARK, f"variant{variant}", seed,
    bo_iterations=500, num_epochs=NUM_EPOCHS,
    minimum_hf_fraction=0.25, real_hf_warmup=2,
    cost_budget=COST_BUDGET, initial_hf=INITIAL_HF, initial_lf=INITIAL_LF,
    dkl_threshold=9999, bes_delta=0.0,
)
config.seed = seed
mf = DirectMFRegretOptimization(config, f_hf, f_lf, bounds)

result = mf.run()

max_hf_at_init = max(mf.initial_hf_values)

rc = result['hf_regret_curve']
final_regret = rc[-1]
n_improved = sum(1 for i in range(1, len(rc)) if rc[i] < rc[i - 1] - 1e-12)
lf_frac = result['lf_fraction']

out = dict(result, max_hf_at_init=max_hf_at_init, incumbent_improved_count=n_improved,
           final_regret=final_regret, variant=variant, seed=seed,
           initial_hf=INITIAL_HF, initial_lf=INITIAL_LF)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"{tag} DONE | max_hf_at_init={max_hf_at_init:.4f} | "
      f"incumbent_improved_count={n_improved} | final_regret={final_regret:.4f} | "
      f"lf_fraction={lf_frac:.3f}", flush=True)
