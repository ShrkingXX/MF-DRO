"""
Does lowering Ackley_10D's HF:LF cost ratio (currently 10:1) shift the
cost-normalized MES acquisition toward more real HF selection, and does
that improve the freeze pattern? Tests c_H in {5, 2} (c_H=10 is the
existing Stage 2 v3 baseline, already have 5 seeds of data for it) -- c_L
stays fixed at 1.0, so this isolates cost ratio as the one variable.
Everything else matches the exact Stage 2 v3 MF-DRO protocol (2x init
sizing hf=60/lf=100, num_epochs=10, dkl_threshold=9999, bes_delta=0.0,
rollout_length=8, fixed 100 iterations, cost tracked not budget-gated).

c_H is overridden on the config object AFTER _build_mf_dro_config builds it
(config.c_H = ...) rather than editing benchmarks.py's registry value --
keeps the shared "Ackley_10D" benchmark definition used elsewhere in the
repo (including the already-completed c_H=10 Stage 2 v3 results) untouched.
"""
import sys
import os
import json

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

c_h_override = float(sys.argv[1])   # 5 or 2
seed = int(sys.argv[2])

BENCHMARK = "Ackley_10D"
N_ITERS = 100
EXP_NAME = "mfdro_ackley_cost_ratio"
X_STAR = [0.5] * 10

OUT_DIR = os.path.join("results", EXP_NAME, "checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"cH{c_h_override}__seed{seed}.json")
tag = f"[Ackley c_H={c_h_override} seed{seed}]"

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

print(f"{tag} Starting (real registry c_H={hf_spec['cost']}, overriding to {c_h_override})", flush=True)

torch.manual_seed(seed)
np.random.seed(seed)
config = _build_mf_dro_config(
    EXP_NAME, BENCHMARK, f"cH{c_h_override}", seed,
    bo_iterations=N_ITERS, num_epochs=10,
    minimum_hf_fraction=0.25, real_hf_warmup=2,
    cost_budget=1e9, initial_hf=60, initial_lf=100,
    dkl_threshold=9999, bes_delta=0.0, rollout_length=8,
    known_optimal_x=X_STAR,
)
config.seed = seed
config.c_H = c_h_override  # override AFTER building -- registry untouched
mf = DirectMFRegretOptimization(config, f_hf, f_lf, bounds)
# c_H is read from config.c_H inside __init__ (self.c_H = config.c_H), so
# the override above is already picked up correctly -- no further patching
# needed.

result = mf.run()

rc = result["hf_regret_curve"]
n_improved = sum(1 for i in range(1, len(rc)) if rc[i] < rc[i - 1] - 1e-12)
distinct = len(set(f"{r:.6f}" for r in rc))
out = dict(result, c_H=c_h_override, seed=seed, incumbent_improved_count=n_improved,
           distinct_regret_values=distinct, final_regret=rc[-1])
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"{tag} DONE final_regret={rc[-1]:.4f} n_iters={len(rc)} n_improved={n_improved} "
      f"distinct={distinct}/{len(rc)} lf_fraction={result['lf_fraction']:.3f}", flush=True)
