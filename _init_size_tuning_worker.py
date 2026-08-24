"""
initial_hf/initial_lf tuning: does the current Song-2019 3d/5d asymmetric
sizing (multiplier=1.0) actually give the best incumbent-improvement
behavior, or does a smaller/larger initial dataset help? Tests 0.5x/1x/2x
the current per-benchmark hf/lf counts, all 3 benchmarks, 3 seeds, fixed
30-iteration horizon (cost_budget=9999, not real-budget-terminated -- a
fast, comparable read, matching this session's established "mini
diagnostic" convention) before committing to the expensive full Stage 2
sweep. Everything else matches the confirmed Stage 2 v3 settings
(num_epochs=10, dkl_threshold=9999, bes_delta=0.0, rollout_length=8).
"""
import sys
import os
import json

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_CONFIG = {
    "Currin_2D":   dict(hf=6,  lf=10),
    "Hartmann_6D": dict(hf=18, lf=30),
    "Borehole_8D": dict(hf=24, lf=40),
}

BENCHMARK = sys.argv[1]
multiplier = float(sys.argv[2])   # 0.5 | 1.0 | 2.0
seed = int(sys.argv[3])

N_ITERS = 30
EXP_NAME = "mfdro_init_size_tuning"

base = BASE_CONFIG[BENCHMARK]
initial_hf = max(2, round(base["hf"] * multiplier))
initial_lf = max(2, round(base["lf"] * multiplier))

OUT_DIR = os.path.join("results", EXP_NAME, "checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"{BENCHMARK}__mult{multiplier}__seed{seed}.json")
tag = f"[{BENCHMARK} mult={multiplier} seed{seed}]"

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

print(f"{tag} Starting (initial_hf={initial_hf}, initial_lf={initial_lf}, "
      f"n_iters={N_ITERS})", flush=True)

torch.manual_seed(seed)
np.random.seed(seed)
config = _build_mf_dro_config(
    EXP_NAME, BENCHMARK, f"mult{multiplier}", seed,
    bo_iterations=N_ITERS, num_epochs=10,
    minimum_hf_fraction=0.25, real_hf_warmup=2,
    cost_budget=9999, initial_hf=initial_hf, initial_lf=initial_lf,
    dkl_threshold=9999, bes_delta=0.0, rollout_length=8,
)
config.seed = seed
mf = DirectMFRegretOptimization(config, f_hf, f_lf, bounds)
result = mf.run()

rc = result["hf_regret_curve"]
final_regret = rc[-1]
n_improved = sum(1 for i in range(1, len(rc)) if rc[i] < rc[i - 1] - 1e-12)
distinct = len(set(f"{r:.6f}" for r in rc))
lf_frac = result["lf_fraction"]

out = dict(result, multiplier=multiplier, initial_hf=initial_hf, initial_lf=initial_lf,
           seed=seed, final_regret=final_regret, incumbent_improved_count=n_improved,
           distinct_regret_values=distinct)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"{tag} DONE | final_regret={final_regret:.4f} | n_improved={n_improved} | "
      f"distinct={distinct}/{len(rc)} | lf_fraction={lf_frac:.3f}", flush=True)
