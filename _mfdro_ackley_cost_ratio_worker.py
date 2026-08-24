"""
MF-DRO on Ackley_10D, sweeping the HF:LF cost ratio (c_L fixed at 1.0, c_H
varied) to test whether a steeper ratio forces the MES acquisition to use
LF more -- the original Ackley test (c_H=5, c_L=1) showed lf_fraction=0.000
on all 3 seeds despite being the best-performing method tested. Does NOT
touch benchmarks.py's registry (which stays at the registered 5:1 for every
other experiment) -- config.c_H is overridden directly on the SimpleNamespace
after _build_mf_dro_config returns, isolated to this one-off script.

Everything else matches the established Ackley_10D MF-DRO config exactly:
initial_hf=30, initial_lf=30 (flat), num_epochs=10, dkl_threshold=9999,
bes_delta=0.0, rollout_length=8, cost_budget=500 (fixed across ratios, so a
steeper ratio genuinely limits how many HF-only queries fit in the same
budget -- the mechanism this test is checking).
"""
import sys
import os
import json

ratio = float(sys.argv[1])   # c_H / c_L, e.g. 5, 10, 20, 40
seed = int(sys.argv[2])

BENCHMARK = "Ackley_10D"
COST_BUDGET = 500
INITIAL_HF = 30
INITIAL_LF = 30
NUM_EPOCHS = 10
C_L = 1.0
C_H = ratio * C_L
EXP_NAME = "mfdro_ackley_cost_ratio"

OUT_DIR = os.path.join("results", EXP_NAME, "checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"ratio{ratio:g}__{BENCHMARK}__seed{seed}.json")
tag = f"[ratio={ratio:g} {BENCHMARK} seed{seed}]"

if os.path.exists(out_path):
    print(f"{tag} SKIPPED (already exists)", flush=True)
    sys.exit(0)

print(f"{tag} Starting (c_H={C_H}, c_L={C_L}, cost_budget={COST_BUDGET})", flush=True)

import torch
import numpy as np
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
    EXP_NAME, BENCHMARK, f"ratio{ratio:g}", seed,
    bo_iterations=2000, num_epochs=NUM_EPOCHS,
    minimum_hf_fraction=0.25, real_hf_warmup=2,
    cost_budget=COST_BUDGET, initial_hf=INITIAL_HF, initial_lf=INITIAL_LF,
    dkl_threshold=9999, bes_delta=0.0, rollout_length=8,
)
config.seed = seed
config.c_H = C_H
config.c_L = C_L
mf = DirectMFRegretOptimization(config, f_hf, f_lf, bounds)
result = mf.run()

rc = result['hf_regret_curve']
final_regret = rc[-1]
n_improved = sum(1 for i in range(1, len(rc)) if rc[i] < rc[i - 1] - 1e-12)
distinct = len(set(f"{r:.6f}" for r in rc))
lf_frac = result['lf_fraction']

out = dict(result, ratio=ratio, seed=seed, final_regret=final_regret,
           incumbent_improved_count=n_improved, distinct_regret_values=distinct,
           c_H=C_H, c_L=C_L)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"{tag} DONE | final_regret={final_regret:.4f} | n_iters={len(rc)} | "
      f"incumbent_improved_count={n_improved} | lf_fraction={lf_frac:.3f}", flush=True)
