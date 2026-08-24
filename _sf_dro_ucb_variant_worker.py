"""
SF-DRO UCB refinement variant comparison (B/C/D), extending the confirmed
Variant A result (regret 0.03-0.27 across seeds 42/43/44):
  VARIANT B (ensemble):  multi-ensemble UCB, mean over all M members.
  VARIANT C (twostage):  beta=5.0/20 steps then beta=0.5/20 steps, single GP.
  VARIANT D (restarts):  N=5 multi-start (DT proposal + 4 random), keep
                          the best-final-UCB result, single GP.

Same base config as the Variant A run (Hartmann_6D, bo_iterations=50,
num_epochs=10, improvement reward, initial_points=36) -- only
gp_refinement_variant differs.
"""
import sys
import os
import json

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

variant = sys.argv[1]   # B | C | D
seed = int(sys.argv[2])

VARIANT_MAP = {"B": "ensemble", "C": "twostage", "D": "restarts"}
gp_refinement_variant = VARIANT_MAP[variant]

BENCHMARK = "Hartmann_6D"
N_ITERS = 50
INITIAL_POINTS = 36
EXP_NAME = "sfdro_ucb_refinement_test"

X_STAR = torch.tensor(
    [0.2017, 0.1500, 0.4769, 0.2753, 0.3116, 0.6573], dtype=torch.float64
)

OUT_DIR = os.path.join("results", EXP_NAME, "checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"SF_REFINED_{variant}__{BENCHMARK}__seed{seed}.json")
tag = f"[SF_REFINED_{variant} {BENCHMARK} seed{seed}]"

if os.path.exists(out_path):
    print(f"{tag} SKIPPED (already exists)", flush=True)
    sys.exit(0)

print(f"{tag} Starting n_iters={N_ITERS} initial_points={INITIAL_POINTS} "
      f"gp_refinement_variant={gp_refinement_variant}", flush=True)

from checkpoint import setup_dirs
from benchmarks import get_benchmark
from dro_runner import _build_dro_config
from src.policy.dro import DirectRegretOptimization

setup_dirs(EXP_NAME)
benchmark_spec = get_benchmark(BENCHMARK)
objective_function = benchmark_spec["make_objective"]()

cfg = _build_dro_config(
    EXP_NAME, BENCHMARK, f"SF_REFINED_{variant}", seed,
    use_mes_reward=False,
    rtg_schema="fixed", alpha_floor=0.5, alpha_inference=None,
    lambda_rtg=1.0, rtg_warmup=3, benchmark_spec=benchmark_spec,
    gp_num_models=5, rollouts_per_iter=75, rollout_length=4,
    bo_iterations=N_ITERS, initial_points=INITIAL_POINTS,
    dt_hidden=128, dt_layers=4, dt_heads=4, dt_lr=1e-4,
    gp_kernel="rbf", gp_ard=True, verbose=False,
    rollout_acq_function="ei", use_awr=False, rollout_teacher="argmax",
    num_epochs=10,
    use_gp_refinement=True,
    gp_refinement_steps=30, gp_refinement_lr=0.05, gp_refinement_beta=2.0,
    gp_refinement_variant=gp_refinement_variant,
)

dro = DirectRegretOptimization(cfg, objective_function)
dro._diag_xstar = X_STAR  # activates the x_DT/x_refined/UCB diagnostic block

dro.run_optimization()

history = dro.iteration_log_history
regret = [d["regret"] for d in history]
refine_log = getattr(dro, '_diag_refine_log', [])

n_improved = 0
improved_per_iter = []
for i, r in enumerate(regret):
    if i > 0 and r < regret[i - 1] - 1e-12:
        n_improved += 1
    improved_per_iter.append(n_improved)

mean_dist_xrefined = (
    sum(r['x_refined_dist'] for r in refine_log) / len(refine_log)
    if refine_log else float('nan')
)
last20 = refine_log[-20:]
frac_closer_last20 = (sum(1 for r in last20 if r['closer']) / len(last20)) if last20 else float('nan')

out = {
    "variant": f"SF_REFINED_{variant}", "gp_refinement_variant": gp_refinement_variant,
    "benchmark": BENCHMARK, "seed": seed,
    "regret_curve": regret,
    "incumbent_improved_count_per_iter": improved_per_iter,
    "refine_log": refine_log,
    "final_regret": regret[-1], "final_incumbent_improved_count": n_improved,
    "mean_dist_xrefined": mean_dist_xrefined,
    "frac_closer_last20": frac_closer_last20,
}
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"\n{tag} DONE final_regret={regret[-1]:.4f} "
      f"incumbent_improved_count={n_improved} mean_dist_xrefined={mean_dist_xrefined:.4f} "
      f"n_iters={len(regret)}", flush=True)
