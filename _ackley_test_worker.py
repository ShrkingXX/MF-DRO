"""
MF-Ackley targeted test: is MF-DRO's incumbent-freeze pathology benchmark-
specific (narrow-basin Hartmann_6D) or architectural (persists on a smooth,
radially-symmetric landscape too)? Ackley_10D only, cost_budget=500
(=100*c_H), initial_hf=initial_lf=30 (flat, standard random LHS init --
deliberately NOT the LF-screened/oracle init from earlier diagnostics, to
isolate the benchmark-identity variable alone). MF-DRO uses the full
Stage 2 v3 config (num_epochs=10, dkl_threshold=9999, bes_delta=0.0,
rollout_length=8, lognormal-prior GP, diverse ensemble init -- all ko_gp.py/
mf_dro.py defaults at this point in the session).
"""
import sys
import os
import json

method = sys.argv[1]      # MF-GP-UCB | MF-MI-Greedy | Greedy-MES | MF-DRO
seed = int(sys.argv[2])

BENCHMARK = "Ackley_10D"
COST_BUDGET = 500
INITIAL_HF = 30
INITIAL_LF = 30
EXP_NAME = "mfdro_ackley_test"
CHECKPOINTS = [50, 150, 500]

EXP_DIR = os.path.join("results", EXP_NAME, "checkpoints")
os.makedirs(EXP_DIR, exist_ok=True)
out_path = os.path.join(EXP_DIR, f"{method}__{BENCHMARK}__seed{seed}.json")
tag = f"[{method} {BENCHMARK} seed{seed}]"

if os.path.exists(out_path):
    print(f"{tag} SKIPPED (already exists)", flush=True)
    sys.exit(0)

print(f"{tag} Starting (cost_budget={COST_BUDGET}, "
      f"initial_hf={INITIAL_HF}, initial_lf={INITIAL_LF})", flush=True)


def regret_at_cost(cost_curve, regret_curve, c_ref):
    idx = None
    for i, c in enumerate(cost_curve):
        if c <= c_ref:
            idx = i
        else:
            break
    return regret_curve[idx] if idx is not None else (regret_curve[0] if regret_curve else float('nan'))


def incumbent_improved_count(regret_curve):
    return sum(1 for i in range(1, len(regret_curve)) if regret_curve[i] < regret_curve[i - 1] - 1e-12)


if method == "MF-DRO":
    from dro_runner import run_mf_single_seed
    from src.models.ko_gp import DeepKernel

    result = run_mf_single_seed(
        EXP_NAME, BENCHMARK, "MF-DRO", seed,
        bo_iterations=500,
        num_epochs=10,
        minimum_hf_fraction=0.25,
        real_hf_warmup=2,
        cost_budget=COST_BUDGET,
        initial_hf=INITIAL_HF,
        initial_lf=INITIAL_LF,
        dkl_threshold=9999,
        bes_delta=0.0,
        rollout_length=8,
    )
    rc, cc = result["hf_regret_curve"], result["cost_curve"]
    best_hf = None  # not directly in result; derive from regret if needed
    n_improved = incumbent_improved_count(rc)
    checkpoint_regrets = {c: regret_at_cost(cc, rc, c) for c in CHECKPOINTS}
    neg_rtg_frac = result.get("neg_rtg_frac_per_iter", [])
    neg_rtg_frac_mean = (sum(neg_rtg_frac) / len(neg_rtg_frac)) if neg_rtg_frac else None

    out = dict(result, incumbent_improved_count=n_improved,
               checkpoint_regrets=checkpoint_regrets, method=method, seed=seed,
               neg_rtg_frac_mean=neg_rtg_frac_mean)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"{tag} DONE | regret@50={checkpoint_regrets[50]:.4f} | "
          f"regret@150={checkpoint_regrets[150]:.4f} | "
          f"regret@500={checkpoint_regrets[500]:.4f} | "
          f"lf_fraction={result['lf_fraction']:.3f} | "
          f"incumbent_improved_count={n_improved}", flush=True)

else:
    from src.baselines.mf_baselines import (
        MultiFidelityBenchmark, MFGPUCBOptimizer, MFMIGreedyOptimizer, GreedyMFMESOptimizer
    )
    bench = MultiFidelityBenchmark(BENCHMARK)
    common = dict(
        n_initial_hf=INITIAL_HF, n_initial_lf=INITIAL_LF,
        seed=seed, cost_budget=COST_BUDGET,
    )
    if method == "MF-GP-UCB":
        opt = MFGPUCBOptimizer(bench, **common)
    elif method == "MF-MI-Greedy":
        opt = MFMIGreedyOptimizer(bench, **common)
    elif method == "Greedy-MES":
        opt = GreedyMFMESOptimizer(bench, **common)
    else:
        raise ValueError(f"unknown method {method}")

    result = opt.run(bo_iterations=500)
    rc, cc = result["regret_curve"], result["cost_curve"]
    n_improved = incumbent_improved_count(rc)
    checkpoint_regrets = {c: regret_at_cost(cc, rc, c) for c in CHECKPOINTS}

    out = dict(result, incumbent_improved_count=n_improved,
               checkpoint_regrets=checkpoint_regrets, method=method, seed=seed)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"{tag} DONE | regret@50={checkpoint_regrets[50]:.4f} | "
          f"regret@150={checkpoint_regrets[150]:.4f} | "
          f"regret@500={checkpoint_regrets[500]:.4f} | "
          f"lf_fraction={result['lf_fraction']:.3f} | "
          f"incumbent_improved_count={n_improved}", flush=True)
