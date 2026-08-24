"""
Gradient-ascent rollout generation test: does training the DT on
trajectories whose x_tau came from UCB gradient ascent (the SAME mechanism
validated at real inference) -- instead of the old acquisition-argmax
teacher -- let the DT learn a genuinely useful warm start? DT architecture
and inference-time UCB refinement mechanism are unchanged; only
rollout_teacher="gradient_ascent" is new (see
DirectRegretOptimization._select_x_tau_gradient_ascent /
_simulate_trajectory in dro.py).

METHOD A: D_nodt (5 random restarts, no DT) -- already have results,
not rerun here (0.0483/13, 0.0240/14, 0.0306/11 on seeds 42/43/44).
METHOD B: DT (trained on gradient-ascent rollouts) + Variant A-style
single UCB refinement warm-started from the DT's own proposal.
METHOD C: DT (trained on gradient-ascent rollouts) + Variant D-style
5-restart refinement (DT proposal + 4 random), keep best-UCB result.
"""
import sys
import os
import json

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

variant = sys.argv[1]   # B | C
seed = int(sys.argv[2])

REFINEMENT_VARIANT = {"B": "single", "C": "restarts"}[variant]

BENCHMARK = "Hartmann_6D"
N_ITERS = 50
INITIAL_POINTS = 36
EXP_NAME = "sfdro_ucb_refinement_test"

X_STAR = torch.tensor(
    [0.2017, 0.1500, 0.4769, 0.2753, 0.3116, 0.6573], dtype=torch.float64
)

OUT_DIR = os.path.join("results", EXP_NAME, "checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"GA_METHOD_{variant}__{BENCHMARK}__seed{seed}.json")
tag = f"[GA_METHOD_{variant} {BENCHMARK} seed{seed}]"

if os.path.exists(out_path):
    print(f"{tag} SKIPPED (already exists)", flush=True)
    sys.exit(0)

print(f"{tag} Starting n_iters={N_ITERS} initial_points={INITIAL_POINTS} "
      f"rollout_teacher=gradient_ascent gp_refinement_variant={REFINEMENT_VARIANT}", flush=True)

from checkpoint import setup_dirs
from benchmarks import get_benchmark
from dro_runner import _build_dro_config
from src.policy.dro import DirectRegretOptimization

setup_dirs(EXP_NAME)
benchmark_spec = get_benchmark(BENCHMARK)
objective_function = benchmark_spec["make_objective"]()

cfg = _build_dro_config(
    EXP_NAME, BENCHMARK, f"GA_METHOD_{variant}", seed,
    use_mes_reward=False,
    rtg_schema="fixed", alpha_floor=0.5, alpha_inference=None,
    lambda_rtg=1.0, rtg_warmup=3, benchmark_spec=benchmark_spec,
    gp_num_models=5, rollouts_per_iter=75, rollout_length=4,
    bo_iterations=N_ITERS, initial_points=INITIAL_POINTS,
    dt_hidden=128, dt_layers=4, dt_heads=4, dt_lr=1e-4,
    gp_kernel="rbf", gp_ard=True, verbose=False,
    rollout_acq_function="ei", use_awr=False,
    rollout_teacher="gradient_ascent", rollout_ga_steps=10,
    num_epochs=10,
    use_gp_refinement=True,
    gp_refinement_steps=30, gp_refinement_lr=0.05, gp_refinement_beta=2.0,
    gp_refinement_variant=REFINEMENT_VARIANT,
)

dro = DirectRegretOptimization(cfg, objective_function)
dro._diag_xstar = X_STAR

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

out = {
    "variant": f"GA_METHOD_{variant}", "gp_refinement_variant": REFINEMENT_VARIANT,
    "rollout_teacher": "gradient_ascent",
    "benchmark": BENCHMARK, "seed": seed,
    "regret_curve": regret,
    "incumbent_improved_count_per_iter": improved_per_iter,
    "final_regret": regret[-1], "final_incumbent_improved_count": n_improved,
    "mean_dist_xrefined": mean_dist_xrefined,
}
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"\n{tag} DONE final_regret={regret[-1]:.4f} "
      f"incumbent_improved_count={n_improved} mean_dist_xrefined={mean_dist_xrefined:.4f} "
      f"n_iters={len(regret)}", flush=True)
