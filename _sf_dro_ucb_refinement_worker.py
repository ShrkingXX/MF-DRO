"""
SF-DRO UCB-refinement test: SF-DRO always queries HF (single-fidelity), so
this isolates whether LOCATION quality is the bottleneck when fidelity is
not a factor at all. Two variants:
  SF_BASELINE:    the DT's raw proposal, unchanged.
  SF_REFINED_UCB: warm-start UCB gradient ascent (Adam, 30 steps) from the
                   DT's proposal, using gp_ensemble[0] directly (see
                   DirectRegretOptimization._refine_proposal_ucb in dro.py).

exp_name="sfdro_ucb_refinement_test", Hartmann_6D, seeds 42/43/44,
bo_iterations=50, num_epochs=10 (per this experiment's explicit spec --
note SF-DRO's num_epochs was previously hardcoded to 100 with no override;
added num_epochs as a real _build_dro_config param for this run, default
still 100 so every other existing caller is unaffected).
"""
import sys
import os
import json

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

variant = sys.argv[1]   # SF_BASELINE | SF_REFINED_UCB
seed = int(sys.argv[2])

BENCHMARK = "Hartmann_6D"
N_ITERS = 50
INITIAL_POINTS = 36
EXP_NAME = "sfdro_ucb_refinement_test"

X_STAR = torch.tensor(
    [0.2017, 0.1500, 0.4769, 0.2753, 0.3116, 0.6573], dtype=torch.float64
)

OUT_DIR = os.path.join("results", EXP_NAME, "checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"{variant}__{BENCHMARK}__seed{seed}.json")
tag = f"[{variant} {BENCHMARK} seed{seed}]"

if os.path.exists(out_path):
    print(f"{tag} SKIPPED (already exists)", flush=True)
    sys.exit(0)

use_refinement = (variant == "SF_REFINED_UCB")
print(f"{tag} Starting n_iters={N_ITERS} initial_points={INITIAL_POINTS} "
      f"use_gp_refinement={use_refinement}", flush=True)

from checkpoint import setup_dirs
from benchmarks import get_benchmark
from dro_runner import _build_dro_config
from src.policy.dro import DirectRegretOptimization

setup_dirs(EXP_NAME)
benchmark_spec = get_benchmark(BENCHMARK)
objective_function = benchmark_spec["make_objective"]()

cfg = _build_dro_config(
    EXP_NAME, BENCHMARK, variant, seed,
    use_mes_reward=False,  # improvement reward, per this run's "current best" spec
    rtg_schema="fixed", alpha_floor=0.5, alpha_inference=None,
    lambda_rtg=1.0, rtg_warmup=3, benchmark_spec=benchmark_spec,
    gp_num_models=5, rollouts_per_iter=75, rollout_length=4,
    bo_iterations=N_ITERS, initial_points=INITIAL_POINTS,
    dt_hidden=128, dt_layers=4, dt_heads=4, dt_lr=1e-4,
    gp_kernel="rbf", gp_ard=True, verbose=False,
    rollout_acq_function="ei", use_awr=False, rollout_teacher="argmax",
    num_epochs=10,
    use_gp_refinement=use_refinement,
    gp_refinement_steps=30, gp_refinement_lr=0.05, gp_refinement_beta=2.0,
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

last20 = refine_log[-20:]
frac_closer_last20 = (sum(1 for r in last20 if r['closer']) / len(last20)) if last20 else float('nan')

out = {
    "variant": variant, "benchmark": BENCHMARK, "seed": seed,
    "regret_curve": regret,
    "incumbent_improved_count_per_iter": improved_per_iter,
    "refine_log": refine_log,
    "final_regret": regret[-1], "final_incumbent_improved_count": n_improved,
    "frac_closer_last20": frac_closer_last20,
}
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"\n{tag} DONE final_regret={regret[-1]:.4f} "
      f"incumbent_improved_count={n_improved} frac_closer_last20={frac_closer_last20:.3f} "
      f"n_iters={len(regret)}", flush=True)
print(f"{tag} per-iter table:", flush=True)
for i in range(len(regret)):
    rl = refine_log[i] if i < len(refine_log) else {}
    print(f"  iter={i:<4} x_DT_dist={rl.get('x_dt_dist', float('nan')):<8.4f} "
          f"x_refined_dist={rl.get('x_refined_dist', float('nan')):<8.4f} "
          f"ucb_dt={rl.get('ucb_at_dt', float('nan')):<8.4f} "
          f"ucb_refined={rl.get('ucb_at_refined', float('nan')):<8.4f} "
          f"regret={regret[i]:<10.4f} cum_improved={improved_per_iter[i]}", flush=True)
