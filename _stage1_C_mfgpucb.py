import json
import os

import torch

from src.baselines.mf_baselines import MultiFidelityBenchmark, MFGPUCBOptimizer

EXP_NAME = "mfdro_stage1"
BENCHMARKS = ["Currin_2D", "Hartmann_6D"]
SEEDS = [42, 43, 44]

out_dir = os.path.join("results", EXP_NAME, "baselines")
os.makedirs(out_dir, exist_ok=True)

for benchmark in BENCHMARKS:
    bench = MultiFidelityBenchmark(benchmark)
    for seed in SEEDS:
        path = os.path.join(out_dir, f"{benchmark}__MF-GP-UCB__seed{seed}.json")
        if os.path.exists(path):
            print(f"[MF-GP-UCB] SKIPPED {benchmark} seed{seed} (already exists)", flush=True)
            continue
        print(f"[MF-GP-UCB] Starting {benchmark} seed{seed}", flush=True)
        torch.manual_seed(seed)
        opt = MFGPUCBOptimizer(bench, n_initial=5, delta=0.1)
        result = opt.run(bo_iterations=30)
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[MF-GP-UCB] DONE {benchmark} seed{seed} "
              f"final_regret={result['regret_curve'][-1]:.4f} "
              f"lf_fraction={result['fidelities'].count('L')/len(result['fidelities']):.3f}", flush=True)

print("[MF-GP-UCB] ALL RUNS COMPLETE", flush=True)
