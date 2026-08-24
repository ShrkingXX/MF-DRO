"""
DIAGNOSTIC 2: cos_sim trajectory under a stochastic "softmax teacher"
(rollout_teacher="softmax", softmax_temperature=0.5) instead of the current
deterministic argmax rollout action selection, on Hartmann_6D only, with the
ORIGINAL MES reward (use_mes_reward=True) unchanged -- isolates whether
stochastic exploration alone (independent of reward scheme) produces more
diverse/incoherent rollout gradients. See _optimize_acquisition_softmax's
docstring in src/policy/dro.py for the mechanism (same broad random search +
acquisition scoring as _optimize_acquisition, softmax-sampled instead of
argmax+refined). Everything else matches the established SF-DRO postfix
config: rtg_schema="floored", alpha_floor=0.5, rollout_acq_function="rotate"
(still cycles ei/ucb/pi/mes by step -- softmax teacher samples FROM whichever
acquisition's score landscape that step resolves to), gp_num_models=5,
rollouts_per_iter=75, rollout_length=4, num_epochs=10. Checkpoints:
0,1,2,5,10,20.
"""
import sys
import os

seed = 42
N_ITERS = 21
NUM_EPOCHS = 10
CHECKPOINT_ITERS = {0, 1, 2, 5, 10, 20}
BENCHMARK = "Hartmann_6D"
INITIAL_POINTS = 18

print(f"=== SF-DRO softmax-teacher diagnostic: {BENCHMARK} seed={seed} "
      f"n_iters={N_ITERS} rollout_teacher=softmax ===", flush=True)

import torch
import numpy as np
from checkpoint import setup_dirs
from benchmarks import get_benchmark
from dro_runner import _build_dro_config
from src.policy.dro import DirectRegretOptimization

setup_dirs("diag_sf_softmax")
torch.set_default_dtype(torch.float64)
torch.manual_seed(seed)
np.random.seed(seed)

benchmark_spec = get_benchmark(BENCHMARK)
objective_function = benchmark_spec["make_objective"]()

cfg = _build_dro_config(
    "diag_sf_softmax", BENCHMARK, "softmax-teacher-check", seed,
    use_mes_reward=True, rtg_schema="floored", alpha_floor=0.5,
    alpha_inference=None, lambda_rtg=1.0, rtg_warmup=3,
    benchmark_spec=benchmark_spec,
    gp_num_models=5, rollouts_per_iter=75, rollout_length=4,
    bo_iterations=N_ITERS, initial_points=INITIAL_POINTS,
    dt_hidden=128, dt_layers=4, dt_heads=4, dt_lr=1e-4,
    gp_kernel="rbf", gp_ard=True, verbose=False,
    rollout_acq_function="rotate",
    rollout_teacher="softmax", softmax_temperature=0.5,
)
cfg.transformer.num_epochs = NUM_EPOCHS

dro = DirectRegretOptimization(cfg, objective_function)
dro._diag_coherence_check = True
dro._diag_checkpoint_iters = CHECKPOINT_ITERS
dro.run_optimization()

print(f"=== DONE {BENCHMARK} seed={seed} ===", flush=True)
