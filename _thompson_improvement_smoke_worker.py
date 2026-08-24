"""
Smoke test for the new rollout_policy="thompson" / rollout_reward=
"improvement" combination. Hartmann_6D, seeds 42/43/44, cost_budget=240,
everything else matching Stage 2 v3 (lognormal prior + noise_lb=1e-2 are
ko_gp.py's own defaults now, diverse init / raw KO state are mf_dro.py's
own defaults, dkl_threshold=9999, num_epochs=10 per the num_epochs
ablation's best-case setting, bes_delta=0.0 -- also a no-op under
rollout_policy="thompson" regardless, since scores=None there).

Monkeypatches DirectMFRegretOptimization._generate_rollout_batch (instance-
level, not a library change) purely to ACCUMULATE each rollout trajectory's
zero_reward_frac / step count across the whole run for the aggregate stat
the smoke test wants -- simulate_mf_trajectory itself already exposes
zero_reward_frac per-trajectory (see mf_dro.py), this just weighted-averages
it over every batch generated during training.
"""
import sys
import os
import json

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

seed = int(sys.argv[1])

BENCHMARK = "Hartmann_6D"
COST_BUDGET = 240
NUM_EPOCHS = 10
INITIAL_HF = 30
INITIAL_LF = 30
EXP_NAME = "mfdro_thompson_improvement_smoke"

OUT_DIR = os.path.join("results", EXP_NAME, "checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"{BENCHMARK}__seed{seed}.json")
tag = f"[thompson/improvement seed{seed}]"

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
    EXP_NAME, BENCHMARK, "thompson_improvement", seed,
    bo_iterations=500, num_epochs=NUM_EPOCHS,
    minimum_hf_fraction=0.25, real_hf_warmup=2,
    cost_budget=COST_BUDGET, initial_hf=INITIAL_HF, initial_lf=INITIAL_LF,
    dkl_threshold=9999, bes_delta=0.0,
    rollout_policy="thompson", rollout_reward="improvement",
)
config.seed = seed
mf = DirectMFRegretOptimization(config, f_hf, f_lf, bounds)

reward_weighted_sum = 0.0
reward_step_count = 0
orig_generate = mf._generate_rollout_batch
def _instrumented_generate():
    global reward_weighted_sum, reward_step_count
    batch = orig_generate()
    for t in batch:
        if 'zero_reward_frac' in t:
            n_steps = t['states'].shape[0]
            reward_weighted_sum += t['zero_reward_frac'] * n_steps
            reward_step_count += n_steps
    return batch
mf._generate_rollout_batch = _instrumented_generate

result = mf.run()

zero_reward_frac = (reward_weighted_sum / reward_step_count
                     if reward_step_count > 0 else float('nan'))

rc = result['hf_regret_curve']
final_regret = rc[-1]
n_improved = sum(1 for i in range(1, len(rc)) if rc[i] < rc[i - 1] - 1e-12)
lf_frac = result['lf_fraction']

out = dict(result, zero_reward_frac=zero_reward_frac,
           incumbent_improved_count=n_improved, final_regret=final_regret,
           seed=seed)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"{tag} DONE | zero_reward_frac={zero_reward_frac:.4f} | "
      f"incumbent_improved_count={n_improved} | final_regret={final_regret:.4f} | "
      f"lf_fraction={lf_frac:.3f}", flush=True)
