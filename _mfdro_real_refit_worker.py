"""
3-way rollout-mechanism comparison on Hartmann_6D, all at a REDUCED but
consistent scale (full refit is ~0.29s/rollout-step, so Stage-2 scale --
M=10, rollouts_per_model=7, rollout_length=8 -- would take hours per run):

  Simulated:  ko.sample_fantasy + frozen-hyperparameter conditioning
              (make_fantasy_ko) -- the original/default rollout mechanism.
  Real:       TRUE function values + frozen-hyperparameter conditioning --
              isolates "does the sampled-vs-real VALUE matter."
  Real-Refit: TRUE function values + full MLL refit at every rollout step
              (KennedyOHaganGP.fit on the augmented real dataset) --
              additionally isolates "does the GP's BELIEF about kernel
              shape get a chance to correct itself mid-rollout."

Reduced scale (all three variants, for a fair comparison): M=5,
rollouts_per_model=3, rollout_length=4, cost_budget=80 (vs Stage 2 v3's
M=10/7/8/800) -- matches this session's established "mini" pilot-scale
convention (see _stage2_mini_worker.py). initial_hf=18, initial_lf=30,
num_epochs=10, dkl_threshold=9999, bes_delta=0.0 unchanged from Stage 2 v3.
"""
import sys
import os
import json

variant = sys.argv[1]   # Simulated | Real | Real-Refit
seed = int(sys.argv[2])

BENCHMARK = "Hartmann_6D"
COST_BUDGET = 80
INITIAL_HF = 18
INITIAL_LF = 30
NUM_EPOCHS = 10
M = 5
ROLLOUTS_PER_MODEL = 3
ROLLOUT_LENGTH = 4
EXP_NAME = "mfdro_real_refit_comparison"

OUT_DIR = os.path.join("results", EXP_NAME, "checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"{variant}__{BENCHMARK}__seed{seed}.json")
tag = f"[{variant} {BENCHMARK} seed{seed}]"

if os.path.exists(out_path):
    print(f"{tag} SKIPPED (already exists)", flush=True)
    sys.exit(0)

print(f"{tag} Starting (cost_budget={COST_BUDGET}, M={M}, "
      f"rollouts_per_model={ROLLOUTS_PER_MODEL}, rollout_length={ROLLOUT_LENGTH})", flush=True)

import torch
import numpy as np
from dro_runner import run_mf_single_seed

torch.set_default_dtype(torch.float64)

use_real = variant in ("Real", "Real-Refit")
use_refit = variant == "Real-Refit"

torch.manual_seed(seed)
np.random.seed(seed)
result = run_mf_single_seed(
    EXP_NAME, BENCHMARK, variant, seed,
    bo_iterations=500,
    num_epochs=NUM_EPOCHS,
    M=M, rollouts_per_model=ROLLOUTS_PER_MODEL, rollout_length=ROLLOUT_LENGTH,
    minimum_hf_fraction=0.25,
    real_hf_warmup=2,
    cost_budget=COST_BUDGET,
    initial_hf=INITIAL_HF,
    initial_lf=INITIAL_LF,
    dkl_threshold=9999,
    bes_delta=0.0,
    use_real_rollout_queries=use_real,
    refit_hyperparams_in_rollout=use_refit,
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
