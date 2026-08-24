"""
Tiny functional smoke test for the new rollout_policy/rollout_reward flags
in simulate_mf_trajectory (mf_dro.py) -- NOT the full requested Hartmann_6D
experiment, just a fast pre-flight check (bo_iterations=3) across all 4
(policy, reward) combinations to catch crashes/shape errors before spending
real time on the cost_budget=240 run.
"""
import os
import sys

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch.set_default_dtype(torch.float64)
from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization

BENCHMARK = "Hartmann_6D"
hf_spec = get_benchmark(BENCHMARK + "_HF")
lf_spec = get_benchmark(BENCHMARK + "_LF")
f_hf = hf_spec["make_objective"]()
f_lf = lf_spec["make_objective"]()
bounds = torch.tensor([hf_spec["domain_min"], hf_spec["domain_max"]], dtype=torch.float64)

for policy in ("mes", "thompson"):
    for reward in ("mes_entropy", "improvement"):
        torch.manual_seed(42)
        np.random.seed(42)
        config = _build_mf_dro_config(
            "smoke_fix12", BENCHMARK, f"{policy}_{reward}", 42,
            bo_iterations=3, num_epochs=2,
            minimum_hf_fraction=0.25, real_hf_warmup=2,
            cost_budget=9999, initial_hf=10, initial_lf=10,
            dkl_threshold=9999,
            rollout_policy=policy, rollout_reward=reward,
        )
        config.seed = 42
        try:
            mf = DirectMFRegretOptimization(config, f_hf, f_lf, bounds)
            result = mf.run()
            print(f"[{policy}/{reward}] OK | final_regret={result['hf_regret_curve'][-1]:.4f} | "
                  f"lf_frac={result['lf_fraction']:.3f}")
        except Exception as e:
            print(f"[{policy}/{reward}] FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

print("ALL COMBOS ATTEMPTED")
