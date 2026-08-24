"""
SF-DRO synthetic-expert-demonstration ablation (ported from
_synthetic_expert_worker.py's MF-DRO regime): does the DT's freeze/drift
pathology persist even when trained on hand-designed, unambiguously-correct
trajectories (linear interpolation toward the TRUE known optimum, TRUE
objective values, reward = normalized true-value improvement), or is it
specific to the GP+acquisition rollout generator / reward informativity?

Keeps the real pipeline's state extraction, _train_decision_transformer,
_propose_next_candidate's real-query DT-inference call, and real benchmark
evaluation entirely UNCHANGED. Replaces ONLY the trajectory-generation block
inside _propose_next_candidate (gated behind self._diag_synthetic_expert_xstar,
see DirectRegretOptimization._make_synthetic_expert_trajectory in dro.py).

If the DT trained on this data STILL proposes real-inference queries far
from x* (dist(x_t,x*) doesn't drop below ~0.4, matching every prior SF-DRO
variant this session), the problem is in the DT's own architecture/training
dynamics generalization, not the rollout generator or reward informativity.
If it doesn't -- i.e. dist shrinks and/or the incumbent starts improving --
the rollout generator/reward construction is implicated instead.
"""
import sys
import os
import json

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BENCHMARK = "Hartmann_6D"
SEED = 42
N_ITERS = 30
INITIAL_POINTS = 36
EXP_NAME = "sfdro_synthetic_expert"

X_STAR = torch.tensor(
    [0.2017, 0.1500, 0.4769, 0.2753, 0.3116, 0.6573], dtype=torch.float64
)

OUT_DIR = os.path.join("results", EXP_NAME, "checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"{BENCHMARK}__synthetic_expert__seed{SEED}.json")
tag = f"[synthetic-expert {BENCHMARK} seed{SEED}]"

if os.path.exists(out_path):
    print(f"{tag} SKIPPED (already exists)", flush=True)
    sys.exit(0)

print(f"{tag} Starting n_iters={N_ITERS} initial_points={INITIAL_POINTS}", flush=True)

from checkpoint import setup_dirs
from benchmarks import get_benchmark
from dro_runner import _build_dro_config
from src.policy.dro import DirectRegretOptimization

setup_dirs(EXP_NAME)
benchmark_spec = get_benchmark(BENCHMARK)
objective_function = benchmark_spec["make_objective"]()

cfg = _build_dro_config(
    EXP_NAME, BENCHMARK, "synthetic-expert", SEED,
    use_mes_reward=False,
    rtg_schema="fixed", alpha_floor=0.5, alpha_inference=None,
    lambda_rtg=1.0, rtg_warmup=3, benchmark_spec=benchmark_spec,
    gp_num_models=5, rollouts_per_iter=75, rollout_length=4,
    bo_iterations=N_ITERS, initial_points=INITIAL_POINTS,
    dt_hidden=128, dt_layers=4, dt_heads=4, dt_lr=1e-4,
    gp_kernel="rbf", gp_ard=True, verbose=False,
    rollout_acq_function="ei",
    use_awr=False,
    rollout_teacher="argmax",
)

dro = DirectRegretOptimization(cfg, objective_function)
dro._diag_coherence_check = True
dro._diag_checkpoint_iters = set(range(0, N_ITERS + 1))  # every iteration
dro._diag_xstar = X_STAR  # activates the dist(x_t, x*) print in _propose_next_candidate
dro._diag_synthetic_expert_xstar = X_STAR  # activates the synthetic-expert rollout override

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
    "benchmark": BENCHMARK, "seed": SEED,
    "regret_curve": regret,
    "incumbent_improved_count_per_iter": improved_per_iter,
    "final_regret": regret[-1], "final_incumbent_improved_count": n_improved,
}
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"\n{tag} DONE final_regret={regret[-1]:.4f} "
      f"incumbent_improved_count={n_improved} n_iters={len(regret)}", flush=True)
