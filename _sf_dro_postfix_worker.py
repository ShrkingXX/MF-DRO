"""
SF-DRO on Hartmann_6D with both fixes applied: causal mask always-on in
get_action_hidden_states (src/model/decisionTransformer.py) and
LogNormalPrior lengthscale regularization + noise_constraint 1e-4->1e-2 in
_construct_gp_model (src/policy/dro.py). Same config as Stage 2 v3's
SF-DRO-rotate-MES variant (_stage2_v3_worker.py): use_mes_reward=True,
rtg_schema="floored", alpha_floor=0.5, rollout_acq_function="rotate",
gp_num_models=5, rollouts_per_iter=75, rollout_length=4, initial_points=18,
cost_budget=800 (-> bo_iterations=100 at c_H=8).
"""
import sys
import os
import json

seed = int(sys.argv[1])

BENCHMARK = "Hartmann_6D"
COST_BUDGET = 800
C_H = 8.0
BO_ITERS = int(COST_BUDGET / C_H)
INITIAL_POINTS = 18
EXP_NAME = "sf_dro_postfix_test"

OUT_DIR = os.path.join("results", EXP_NAME, "checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"SF-DRO-postfix__{BENCHMARK}__seed{seed}.json")
tag = f"[SF-DRO-postfix {BENCHMARK} seed{seed}]"

if os.path.exists(out_path):
    print(f"{tag} SKIPPED (already exists)", flush=True)
    sys.exit(0)

print(f"{tag} Starting (bo_iterations={BO_ITERS}, initial_points={INITIAL_POINTS})", flush=True)

import torch
import numpy as np
from checkpoint import setup_dirs

setup_dirs(EXP_NAME)
from dro_runner import run_single_seed

torch.set_default_dtype(torch.float64)
torch.manual_seed(seed)
np.random.seed(seed)

result = run_single_seed(
    EXP_NAME, BENCHMARK, "SF-DRO-postfix", seed,
    use_mes_reward=True,
    rtg_schema="floored",
    alpha_floor=0.5,
    rollout_acq_function="rotate",
    gp_num_models=5,
    rollouts_per_iter=75,
    rollout_length=4,
    bo_iterations=BO_ITERS,
    initial_points=INITIAL_POINTS,
)

rc = result["regret_curve"]
final_regret = rc[-1]
n_improved = sum(1 for i in range(1, len(rc)) if rc[i] < rc[i - 1] - 1e-12)
distinct = len(set(f"{r:.6f}" for r in rc))

out = dict(result, seed=seed, final_regret=final_regret,
           incumbent_improved_count=n_improved, distinct_regret_values=distinct)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"{tag} DONE | final_regret={final_regret:.4f} | n_iters={len(rc)} | "
      f"incumbent_improved_count={n_improved} | distinct={distinct}/{len(rc)}", flush=True)
