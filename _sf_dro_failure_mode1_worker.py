"""
Failure-mode-1 diagnostic: does multimodal action distribution from the
diverse GP ensemble (distinct np.linspace lengthscale init per member)
cause the DT to "mode-average" across basins and thus produce bad
proposals -- versus the freeze being unrelated to ensemble diversity?

THREE variants, Hartmann_6D, seeds 42/43/44:
  MULTI: uniform_ensemble=False (diverse init, standard) + DT +
         5-restart UCB refinement at inference (Variant D).
  UNI:   uniform_ensemble=True (identical median-lengthscale init for all
         members) + DT + 5-restart UCB refinement at inference. Same as
         MULTI except ensemble diversity is removed.
  NODT:  uniform_ensemble=False (diverse, standard GP -- fidelity of the
         GP itself is irrelevant here since NODT never trains/queries a
         DT), propose_mode="multistart_ucb_nodt" (D_nodt baseline: 5
         random-restart UCB, no DT at all).

Per real BO iteration, DirectRegretOptimization._measure_ucb_peak_spread()
is called (gated behind dro._diag_ucb_spread=True) and prints
"iter t: ucb_peak_spread = ... (mode=...)" live; results also land in
dro._diag_ucb_spread_log for post-hoc aggregation here.
"""
import sys
import os
import json

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

variant = sys.argv[1]   # MULTI | UNI | NODT
seed = int(sys.argv[2])

VARIANT_CFG = {
    "MULTI": dict(uniform_ensemble=False, propose_mode="dt", use_gp_refinement=True,
                  gp_refinement_variant="restarts"),
    "UNI":   dict(uniform_ensemble=True, propose_mode="dt", use_gp_refinement=True,
                  gp_refinement_variant="restarts"),
    "NODT":  dict(uniform_ensemble=False, propose_mode="multistart_ucb_nodt",
                  use_gp_refinement=False, gp_refinement_variant="restarts"),
}[variant]

BENCHMARK = "Hartmann_6D"
N_ITERS = 50
INITIAL_POINTS = 36
EXP_NAME = "failure_mode1_sfdro"

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

print(f"{tag} Starting n_iters={N_ITERS} initial_points={INITIAL_POINTS} cfg={VARIANT_CFG}", flush=True)

from checkpoint import setup_dirs
from benchmarks import get_benchmark
from dro_runner import _build_dro_config
from src.policy.dro import DirectRegretOptimization

setup_dirs(EXP_NAME)
benchmark_spec = get_benchmark(BENCHMARK)
objective_function = benchmark_spec["make_objective"]()

cfg = _build_dro_config(
    EXP_NAME, BENCHMARK, variant, seed,
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
    **VARIANT_CFG,
)

dro = DirectRegretOptimization(cfg, objective_function)
dro._diag_xstar = X_STAR
dro._diag_ucb_spread = True

dro.run_optimization()

history = dro.iteration_log_history
regret = [d["regret"] for d in history]
spread_log = getattr(dro, '_diag_ucb_spread_log', [])

n_improved = 0
improved_per_iter = []
for i, r in enumerate(regret):
    if i > 0 and r < regret[i - 1] - 1e-12:
        n_improved += 1
    improved_per_iter.append(n_improved)

mean_ucb_peak_spread = sum(spread_log) / len(spread_log) if spread_log else float('nan')

out = {
    "variant": variant, "cfg": VARIANT_CFG,
    "benchmark": BENCHMARK, "seed": seed,
    "regret_curve": regret,
    "ucb_peak_spread_curve": spread_log,
    "incumbent_improved_count_per_iter": improved_per_iter,
    "final_regret": regret[-1], "final_incumbent_improved_count": n_improved,
    "mean_ucb_peak_spread": mean_ucb_peak_spread,
}
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"\n{tag} DONE final_regret={regret[-1]:.4f} "
      f"incumbent_improved_count={n_improved} "
      f"mean_ucb_peak_spread={mean_ucb_peak_spread:.4f} n_iters={len(regret)}", flush=True)
