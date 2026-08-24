"""
SF-DRO reward/teacher ablation (VARIANT A/B/C): does the incumbent freeze
on SF-DRO come from reward sparsity (density alone), or does it need BOTH
denser signal AND incoherent (non-collapsing) gradients to break?

VARIANT A -- current/base config: use_mes_reward=False, rollout_acq_function
default ("ei"), no AWR, rollout_teacher default ("argmax").
VARIANT B -- MES reward (denser), same teacher, no AWR: use_mes_reward=True.
VARIANT C -- improvement reward (incoherent) + softmax teacher (denser
basin-hit rate) + AWR loss weighting: use_mes_reward=False,
rollout_teacher="softmax", use_awr=True.

Bypasses dro_runner.run_single_seed (its **kwargs is never forwarded to
_build_dro_config, so use_awr/rollout_teacher would silently be dropped)
and constructs DirectRegretOptimization directly -- needed anyway to set
the _diag_coherence_check/_diag_checkpoint_iters diagnostic flags before
run_optimization(), which the existing (already-implemented, verified
working) coherence-check code in dro.py uses to print mean_cos_sim/
zero_reward_frac live, once per iteration in _diag_checkpoint_iters.

Benchmark/seed/iteration count are NOT specified in the request -- defaulting
to Hartmann_6D (this investigation's established primary failure case),
seed=42, bo_iterations=30 (kept moderate since _diag_coherence_check does
~rollouts_per_iter=75 extra backward passes EVERY iteration when checkpointed
every iteration, as requested -- 30 iterations keeps wall-clock reasonable
while still showing a clear trend).
"""
import sys
import os
import json

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

variant = sys.argv[1]   # A | B | C

BENCHMARK = "Hartmann_6D"
SEED = 42
N_ITERS = 30
INITIAL_POINTS = 36  # Stage2 v3's 2x-scaled Hartmann_6D initial_hf convention
EXP_NAME = "sfdro_reward_teacher_test"

OUT_DIR = os.path.join("results", EXP_NAME, "checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"VARIANT_{variant}__{BENCHMARK}__seed{SEED}.json")
tag = f"[VARIANT {variant} {BENCHMARK} seed{SEED}]"

if os.path.exists(out_path):
    print(f"{tag} SKIPPED (already exists)", flush=True)
    sys.exit(0)

VARIANT_CFG = {
    "A": dict(use_mes_reward=False, rollout_teacher="argmax", use_awr=False, rollout_acq_function="ei"),
    "B": dict(use_mes_reward=True, rollout_teacher="argmax", use_awr=False, rollout_acq_function="ei"),
    "C": dict(use_mes_reward=False, rollout_teacher="softmax", use_awr=True, rollout_acq_function="ei"),
    # D: paper's actual diversity mechanism (Zhang & Chen, arXiv 2507.06529) --
    # rotate the rollout's per-step query-selection acquisition through
    # [ei, ucb, pi, mes] (step % 4) instead of a single fixed acquisition.
    # Already-implemented, existing dro.py feature (rollout_acq_function=
    # "rotate", see _simulate_trajectory's step_acq_name) -- never actually
    # exercised in Variants A/B/C, which all pinned rollout_acq_function="ei".
    # Isolates rotation alone: otherwise identical to Variant A.
    "D": dict(use_mes_reward=False, rollout_teacher="argmax", use_awr=False, rollout_acq_function="rotate"),
    # E: action-reward informativity test -- reward = -dist(x_tau, x*)
    # (oracle, diagnostic-only), RTG = backward cumsum of that reward
    # (dro.py's own existing rtg_schema="fixed" mechanism, unchanged --
    # already literally "cumulative reward" by construction). Guarantees
    # a definite reward-quality relationship; if the freeze persists even
    # here, the problem isn't reward informativity. use_mes_reward is moot
    # (the oracle branch is checked first and short-circuits it) but kept
    # explicit for clarity.
    "E": dict(use_mes_reward=False, rollout_teacher="argmax", use_awr=False, rollout_acq_function="ei"),
    # F: state-informativity test -- use_roi_state=True appends
    # [mean(mu_roi), mean(sigma_roi)] per GP ensemble member (10 more dims)
    # to the state vector: the GP's posterior mean/std averaged over a
    # domain-spanning candidate set, the ONLY existing mechanism giving the
    # DT any location-resolved sense of the objective surface. Every prior
    # variant (A-E) ran with this off -- the state carried just one point
    # (best_position) and two global kernel scalars, nothing about
    # explored-vs-unexplored regions elsewhere in the domain. Otherwise
    # identical to Variant A.
    "F": dict(use_mes_reward=False, rollout_teacher="argmax", use_awr=False, rollout_acq_function="ei",
               use_roi_state=True),
}[variant]

print(f"{tag} Starting n_iters={N_ITERS} initial_points={INITIAL_POINTS} "
      f"config={VARIANT_CFG}", flush=True)

from checkpoint import setup_dirs
from benchmarks import get_benchmark
from dro_runner import _build_dro_config
from src.policy.dro import DirectRegretOptimization

setup_dirs(EXP_NAME)
benchmark_spec = get_benchmark(BENCHMARK)
objective_function = benchmark_spec["make_objective"]()

cfg = _build_dro_config(
    EXP_NAME, BENCHMARK, f"VARIANT-{variant}", SEED,
    use_mes_reward=VARIANT_CFG["use_mes_reward"],
    rtg_schema="fixed", alpha_floor=0.5, alpha_inference=None,
    lambda_rtg=1.0, rtg_warmup=3, benchmark_spec=benchmark_spec,
    gp_num_models=5, rollouts_per_iter=75, rollout_length=4,
    bo_iterations=N_ITERS, initial_points=INITIAL_POINTS,
    dt_hidden=128, dt_layers=4, dt_heads=4, dt_lr=1e-4,
    gp_kernel="rbf", gp_ard=True, verbose=False,
    rollout_acq_function=VARIANT_CFG["rollout_acq_function"],
    use_awr=VARIANT_CFG["use_awr"],
    rollout_teacher=VARIANT_CFG["rollout_teacher"],
    use_roi_state=VARIANT_CFG.get("use_roi_state", False),
)

X_STAR = torch.tensor(
    [0.2017, 0.1500, 0.4769, 0.2753, 0.3116, 0.6573], dtype=torch.float64
)

dro = DirectRegretOptimization(cfg, objective_function)
dro._diag_coherence_check = True
dro._diag_checkpoint_iters = set(range(0, N_ITERS + 1))  # every iteration
dro._diag_xstar = X_STAR  # activates the dist(x_t, x*) print in _propose_next_candidate
if variant == "E":
    dro._diag_oracle_reward_xstar = X_STAR  # activates the -dist(x_tau,x*) reward override

dro.run_optimization()

history = dro.iteration_log_history
regret = [d["regret"] for d in history]
zero_frac = [d["zero_frac"] for d in history]

n_improved = 0
improved_per_iter = []
for i, r in enumerate(regret):
    if i > 0 and r < regret[i - 1] - 1e-12:
        n_improved += 1
    improved_per_iter.append(n_improved)

out = {
    "variant": variant, "config": VARIANT_CFG, "benchmark": BENCHMARK, "seed": SEED,
    "regret_curve": regret, "zero_frac": zero_frac,
    "incumbent_improved_count_per_iter": improved_per_iter,
    "final_regret": regret[-1], "final_incumbent_improved_count": n_improved,
}
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"\n{tag} DONE final_regret={regret[-1]:.4f} "
      f"incumbent_improved_count={n_improved} n_iters={len(regret)}", flush=True)
print(f"{tag} per-iter table:", flush=True)
for i in range(len(regret)):
    print(f"  iter={i:<4} regret={regret[i]:<10.4f} zero_frac={zero_frac[i]:<8.4f} "
          f"cum_improved={improved_per_iter[i]}", flush=True)
