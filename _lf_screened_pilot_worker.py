"""
Pilot: does LF-screened HF init reduce the incumbent freeze? Same config as
Stage 2 v3 (per-benchmark initial_hf/lf, real cost_budget, num_epochs=10,
dkl_threshold=9999, bes_delta=0.0, rollout_length=8) -- only
use_lf_screened_init toggles. Hartmann_6D/Borehole_8D only (the two
benchmarks that showed severe freezing); Currin_2D skipped since its
"frozen" cases were mostly legitimate near-optimal convergence, not the
pathology this is meant to fix.
"""
import sys
import os
import json

BENCHMARK_CONFIG = {
    "Hartmann_6D": dict(d=6, initial_hf=18, initial_lf=30, cost_budget=800),
    "Borehole_8D": dict(d=8, initial_hf=24, initial_lf=40, cost_budget=200),
}

EXP_NAME = "mfdro_lf_screened_pilot"

variant = sys.argv[1]      # off | on
benchmark = sys.argv[2]
seed = int(sys.argv[3])

cfg = BENCHMARK_CONFIG[benchmark]
EXP_DIR = os.path.join("results", EXP_NAME, "checkpoints")
os.makedirs(EXP_DIR, exist_ok=True)
out_path = os.path.join(EXP_DIR, f"{variant}__{benchmark}__seed{seed}.json")
tag = f"[{variant} {benchmark} seed{seed}]"

if os.path.exists(out_path):
    print(f"{tag} SKIPPED (already exists)", flush=True)
    sys.exit(0)

print(f"{tag} Starting (cost_budget={cfg['cost_budget']}, "
      f"initial_hf={cfg['initial_hf']}, initial_lf={cfg['initial_lf']})", flush=True)

from dro_runner import run_mf_single_seed
result = run_mf_single_seed(
    EXP_NAME, benchmark, variant, seed,
    bo_iterations=500,
    num_epochs=10,
    minimum_hf_fraction=0.25,
    real_hf_warmup=2,
    cost_budget=cfg["cost_budget"],
    initial_hf=cfg["initial_hf"],
    initial_lf=cfg["initial_lf"],
    dkl_threshold=9999,
    bes_delta=0.0,
    rollout_length=8,
    use_lf_screened_init=(variant == "on"),
)
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)

rc = result["hf_regret_curve"]
n_improved = sum(1 for i in range(1, len(rc)) if rc[i] < rc[i - 1] - 1e-12)
distinct = len(set(f"{r:.6f}" for r in rc))
print(f"{tag} DONE final_regret={rc[-1]:.4f} n_iters={len(rc)} "
      f"n_improved={n_improved} distinct={distinct}/{len(rc)} "
      f"lf_fraction={result['lf_fraction']:.3f} final_cost={result['cost_curve'][-1]:.1f}",
      flush=True)
