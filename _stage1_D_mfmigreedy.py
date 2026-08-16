import json
import os

import torch

from src.baselines.mf_baselines import MultiFidelityBenchmark, MFMIGreedyOptimizer

EXP_NAME = "mfdro_stage1"
BENCHMARKS = ["Currin_2D", "Hartmann_6D"]
SEEDS = [42, 43, 44]

out_dir = os.path.join("results", EXP_NAME, "baselines")
os.makedirs(out_dir, exist_ok=True)

for benchmark in BENCHMARKS:
    bench = MultiFidelityBenchmark(benchmark)
    for seed in SEEDS:
        path = os.path.join(out_dir, f"{benchmark}__MF-MI-Greedy__seed{seed}.json")
        if os.path.exists(path):
            print(f"[MF-MI-Greedy] SKIPPED {benchmark} seed{seed} (already exists)", flush=True)
            continue
        print(f"[MF-MI-Greedy] Starting {benchmark} seed{seed}", flush=True)
        torch.manual_seed(seed)
        opt = MFMIGreedyOptimizer(bench, n_initial=5)
        result = opt.run(bo_iterations=30)
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[MF-MI-Greedy] DONE {benchmark} seed{seed} "
              f"final_regret={result['regret_curve'][-1]:.4f}", flush=True)

print("[MF-MI-Greedy] ALL RUNS COMPLETE", flush=True)
