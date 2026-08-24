"""
Two diagnostics to determine whether DRO's DT contributes anything to
Variant D's (5-restart UCB) result, or whether the whole improvement is
just "multi-start UCB + lognormal-prior GP" -- achievable without any
DT/rollout training at all.

DIAGNOSTIC 1 (D_nodt): 5 random-restart UCB, NO DT proposal seeding any
start (all 5 restarts uniform-random). Same beta/steps as Variant D.
Compare final_regret/n_improved against Variant D (0.0439/8, 0.0231/12,
0.0050/7 on seeds 42/43/44).

DIAGNOSTIC 2 (NaiveBO_lognormal): standard NaiveBO -- single-shot
_optimize_acquisition (broad random search + local gradient refinement,
no restarts, no DT) on the SAME lognormal-prior-calibrated GP
(_construct_gp_model already has this fix; nothing new needed).

Both bypass rollout simulation + DT training entirely via
propose_mode="multistart_ucb_nodt"/"naivebo_lognormal" in dro.py's
_propose_next_candidate.
"""
import sys
import os
import json

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

diagnostic = sys.argv[1]   # D_nodt | NaiveBO_lognormal
seed = int(sys.argv[2])

PROPOSE_MODE = {"D_nodt": "multistart_ucb_nodt", "NaiveBO_lognormal": "naivebo_lognormal"}[diagnostic]

BENCHMARK = "Hartmann_6D"
N_ITERS = 50
INITIAL_POINTS = 36
EXP_NAME = "sfdro_ucb_refinement_test"

X_STAR = torch.tensor(
    [0.2017, 0.1500, 0.4769, 0.2753, 0.3116, 0.6573], dtype=torch.float64
)

OUT_DIR = os.path.join("results", EXP_NAME, "checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"{diagnostic}__{BENCHMARK}__seed{seed}.json")
tag = f"[{diagnostic} {BENCHMARK} seed{seed}]"

if os.path.exists(out_path):
    print(f"{tag} SKIPPED (already exists)", flush=True)
    sys.exit(0)

print(f"{tag} Starting n_iters={N_ITERS} initial_points={INITIAL_POINTS} "
      f"propose_mode={PROPOSE_MODE}", flush=True)

from checkpoint import setup_dirs
from benchmarks import get_benchmark
from dro_runner import _build_dro_config
from src.policy.dro import DirectRegretOptimization

setup_dirs(EXP_NAME)
benchmark_spec = get_benchmark(BENCHMARK)
objective_function = benchmark_spec["make_objective"]()

cfg = _build_dro_config(
    EXP_NAME, BENCHMARK, diagnostic, seed,
    use_mes_reward=False,
    rtg_schema="fixed", alpha_floor=0.5, alpha_inference=None,
    lambda_rtg=1.0, rtg_warmup=3, benchmark_spec=benchmark_spec,
    gp_num_models=5, rollouts_per_iter=75, rollout_length=4,
    bo_iterations=N_ITERS, initial_points=INITIAL_POINTS,
    dt_hidden=128, dt_layers=4, dt_heads=4, dt_lr=1e-4,
    gp_kernel="rbf", gp_ard=True, verbose=False,
    rollout_acq_function="ei", use_awr=False, rollout_teacher="argmax",
    num_epochs=10,
    gp_refinement_steps=30, gp_refinement_lr=0.05, gp_refinement_beta=2.0,
    propose_mode=PROPOSE_MODE,
)

dro = DirectRegretOptimization(cfg, objective_function)
dro._diag_xstar = X_STAR

dro.run_optimization()

history = dro.iteration_log_history
regret = [d["regret"] for d in history]

n_improved = 0
improved_per_iter = []
for i, r in enumerate(regret):
    if i > 0 and r < regret[i - 1] - 1e-12:
        n_improved += 1
    improved_per_iter.append(n_improved)

out = {
    "diagnostic": diagnostic, "propose_mode": PROPOSE_MODE,
    "benchmark": BENCHMARK, "seed": seed,
    "regret_curve": regret,
    "incumbent_improved_count_per_iter": improved_per_iter,
    "final_regret": regret[-1], "final_incumbent_improved_count": n_improved,
}
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"\n{tag} DONE final_regret={regret[-1]:.4f} "
      f"incumbent_improved_count={n_improved} n_iters={len(regret)}", flush=True)
