"""
SF-DRO per-example gradient-coherence check, now over MULTIPLE checkpoint
iterations (0, 1, 2, 5, 10, 20) instead of just iteration 5: for each
training trajectory individually, compute the RTG-embedding gradient and
measure mean pairwise cosine similarity across all of them, plus an
RTG-sensitivity probe_spread, at the END of each checkpointed iteration's
training (so L_loc_final/mean_cos_sim/probe_spread all reflect the same
snapshot). Tests whether coherence is stable across training (a fixed
structural property) or changes as the DT converges (consistent with
convergence-to-a-state-only-minimum).

Same config as the grad-magnitude diagnostic (causal mask + lognormal prior
already applied): use_mes_reward=True, rtg_schema="floored", alpha_floor=0.5,
rollout_acq_function="rotate", gp_num_models=5, rollouts_per_iter=75,
rollout_length=4, num_epochs=10. Runs to iteration 20 (21 real iterations
total) to cover every requested checkpoint.
"""
import sys
import os

BENCHMARK = sys.argv[1]  # Hartmann_6D | Ackley_10D
seed = 42
N_ITERS = 21
NUM_EPOCHS = 10
CHECKPOINT_ITERS = {0, 1, 2, 5, 10, 20}

INITIAL_POINTS = {"Hartmann_6D": 18, "Ackley_10D": 30}[BENCHMARK]

print(f"=== SF-DRO grad coherence check: {BENCHMARK} seed={seed} "
      f"n_iters={N_ITERS} ===", flush=True)

import torch
import numpy as np
from checkpoint import setup_dirs
from benchmarks import get_benchmark
from dro_runner import _build_dro_config
from src.policy.dro import DirectRegretOptimization

setup_dirs("diag_sf_coherence")
torch.set_default_dtype(torch.float64)
torch.manual_seed(seed)
np.random.seed(seed)

benchmark_spec = get_benchmark(BENCHMARK)
objective_function = benchmark_spec["make_objective"]()

cfg = _build_dro_config(
    "diag_sf_coherence", BENCHMARK, "coherence-check", seed,
    use_mes_reward=True, rtg_schema="floored", alpha_floor=0.5,
    alpha_inference=None, lambda_rtg=1.0, rtg_warmup=3,
    benchmark_spec=benchmark_spec,
    gp_num_models=5, rollouts_per_iter=75, rollout_length=4,
    bo_iterations=N_ITERS, initial_points=INITIAL_POINTS,
    dt_hidden=128, dt_layers=4, dt_heads=4, dt_lr=1e-4,
    gp_kernel="rbf", gp_ard=True, verbose=False,
    rollout_acq_function="rotate",
)
cfg.transformer.num_epochs = NUM_EPOCHS

dro = DirectRegretOptimization(cfg, objective_function)
dro._diag_coherence_check = True
dro._diag_checkpoint_iters = CHECKPOINT_ITERS
dro.run_optimization()

print(f"=== DONE {BENCHMARK} seed={seed} ===", flush=True)
