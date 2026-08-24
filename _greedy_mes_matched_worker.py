"""
GreedyMFMESOptimizer ("Greedy-MES") at the EXACT scale used for the
Real-Refit comparison (cost_budget=80, initial_hf=18, initial_lf=30,
Hartmann_6D) -- this class already IS "MF-DRO with the DT deleted, same
compute_joint_mf_mes-derived switching condition, full GP refit every real
iteration" (see its own docstring in src/baselines/mf_baselines.py). No new
mechanism needed; this just runs it at a scale directly comparable to
_mfdro_real_refit_worker.py's Simulated/Real/Real-Refit results.
"""
import sys
import os
import json

seed = int(sys.argv[1])

BENCHMARK = "Hartmann_6D"
COST_BUDGET = 80
INITIAL_HF = 18
INITIAL_LF = 30
EXP_NAME = "mfdro_real_refit_comparison"  # same experiment dir as the other 9 jobs

OUT_DIR = os.path.join("results", EXP_NAME, "checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"Greedy-MES__{BENCHMARK}__seed{seed}.json")
tag = f"[Greedy-MES {BENCHMARK} seed{seed}]"

if os.path.exists(out_path):
    print(f"{tag} SKIPPED (already exists)", flush=True)
    sys.exit(0)

print(f"{tag} Starting (cost_budget={COST_BUDGET}, initial_hf={INITIAL_HF}, initial_lf={INITIAL_LF})", flush=True)

from src.baselines.mf_baselines import MultiFidelityBenchmark, GreedyMFMESOptimizer

bench = MultiFidelityBenchmark(BENCHMARK)
opt = GreedyMFMESOptimizer(
    bench, n_initial_hf=INITIAL_HF, n_initial_lf=INITIAL_LF,
    seed=seed, cost_budget=COST_BUDGET, use_dkl=False,
)
result = opt.run(bo_iterations=500)

rc = result['regret_curve']
final_regret = rc[-1]
n_improved = sum(1 for i in range(1, len(rc)) if rc[i] < rc[i - 1] - 1e-12)
distinct = len(set(f"{r:.6f}" for r in rc))
lf_frac = result['lf_fraction']

out = dict(result, variant="Greedy-MES", seed=seed, final_regret=final_regret,
           incumbent_improved_count=n_improved, distinct_regret_values=distinct)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"{tag} DONE | final_regret={final_regret:.4f} | n_iters={len(rc)} | "
      f"incumbent_improved_count={n_improved} | distinct={distinct}/{len(rc)} | "
      f"lf_fraction={lf_frac:.3f}", flush=True)
