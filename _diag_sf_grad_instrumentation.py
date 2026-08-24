"""
SF-DRO gradient-norm instrumentation: is Hartmann_6D's freeze explained by
the RTG embedding's gradient being structurally dwarfed by the state
embedding's, and is that imbalance benchmark-specific (Ackley differs) or a
fixed architectural property (same ratio on both, so imbalance alone can't
explain why Ackley works and Hartmann doesn't)?

Single seed (42), 20 real BO iterations, num_epochs=10 (overridden from the
config's default 100 -- cheap diagnostic, not the full Stage 2 run). All
other config matches the just-applied SF-DRO postfix (causal mask always-on
in get_action_hidden_states, LogNormalPrior lengthscale + noise_constraint
1e-2 in _construct_gp_model): use_mes_reward=True, rtg_schema="floored",
alpha_floor=0.5, rollout_acq_function="rotate", gp_num_models=5,
rollouts_per_iter=75, rollout_length=4.

Ackley_10D here resolves to the ORIGINAL single-fidelity registry entry
([-32.768,32.768]^10 bounds, benchmarks.py's pre-existing "Ackley_10D") --
NOT this session's new Ackley_10D_HF/LF pair (MF-only, requires the _HF/_LF
split SF-DRO doesn't have). This is the correct, already-established
SF-DRO-compatible Ackley benchmark.
"""
import sys
import os

BENCHMARK = sys.argv[1]  # Hartmann_6D | Ackley_10D
seed = 42
N_ITERS = 20
NUM_EPOCHS = 10

INITIAL_POINTS = {"Hartmann_6D": 18, "Ackley_10D": 30}[BENCHMARK]

print(f"=== SF-DRO grad instrumentation: {BENCHMARK} seed={seed} "
      f"n_iters={N_ITERS} num_epochs={NUM_EPOCHS} ===", flush=True)

import torch
import numpy as np
from checkpoint import setup_dirs
from benchmarks import get_benchmark
from dro_runner import _build_dro_config
from src.policy.dro import DirectRegretOptimization

setup_dirs("diag_sf_grad")
torch.set_default_dtype(torch.float64)
torch.manual_seed(seed)
np.random.seed(seed)

benchmark_spec = get_benchmark(BENCHMARK)
objective_function = benchmark_spec["make_objective"]()

cfg = _build_dro_config(
    "diag_sf_grad", BENCHMARK, "grad-instrumented", seed,
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
dro._diag_grad_instrumentation = True
dro.run_optimization()

print(f"=== DONE {BENCHMARK} seed={seed} ===", flush=True)
