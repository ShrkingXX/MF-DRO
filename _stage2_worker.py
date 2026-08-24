import sys
import os
import json

BENCHMARK_CONFIG = {
    "Currin_2D":   dict(d=2, initial_hf=6,  initial_lf=10, cost_budget=300),
    "Hartmann_6D": dict(d=6, initial_hf=18, initial_lf=30, cost_budget=800),
    "Borehole_8D": dict(d=8, initial_hf=24, initial_lf=40, cost_budget=200),
}

method = sys.argv[1]      # SF-DRO | MF-GP-UCB | MF-MI-Greedy | Greedy-MES | MF-DRO
benchmark = sys.argv[2]
seed = int(sys.argv[3])

cfg = BENCHMARK_CONFIG[benchmark]
EXP_DIR = os.path.join("results", "mfdro_stage2", "checkpoints")
os.makedirs(EXP_DIR, exist_ok=True)
out_path = os.path.join(EXP_DIR, f"{method}__{benchmark}__seed{seed}.json")

tag = f"[{method} {benchmark} seed{seed}]"

if os.path.exists(out_path):
    print(f"{tag} SKIPPED (already exists)", flush=True)
    sys.exit(0)

print(f"{tag} Starting (cost_budget={cfg['cost_budget']}, "
      f"initial_hf={cfg['initial_hf']}, initial_lf={cfg['initial_lf']})", flush=True)

if method == "MF-DRO":
    from dro_runner import run_mf_single_seed
    result = run_mf_single_seed(
        "mfdro_stage2", benchmark, "MF-DRO", seed,
        bo_iterations=500,
        num_epochs=100,
        minimum_hf_fraction=0.25,
        real_hf_warmup=2,
        cost_budget=cfg["cost_budget"],
        initial_hf=cfg["initial_hf"],
        initial_lf=cfg["initial_lf"],
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    final_regret = result["hf_regret_curve"][-1]
    lf_frac = result["lf_fraction"]
    n_iters = len(result["fidelity_trace"])
    print(f"{tag} DONE final_regret={final_regret:.4f} lf_fraction={lf_frac:.3f} "
          f"n_iters={n_iters} final_cost={result['cost_curve'][-1]:.1f}", flush=True)

elif method == "SF-DRO":
    from checkpoint import setup_dirs
    setup_dirs("mfdro_stage2")
    from dro_runner import run_single_seed
    hf_spec_cost = {"Currin_2D": 3.0, "Hartmann_6D": 8.0, "Borehole_8D": 2.0}[benchmark]
    bo_iters = int(cfg["cost_budget"] / hf_spec_cost)  # SF-DRO is HF-only; = 100 for all 3
    result = run_single_seed(
        "mfdro_stage2", benchmark, "SF-DRO-rotate-MES", seed,
        use_mes_reward=True,
        rtg_schema="floored",
        alpha_floor=0.5,
        rollout_acq_function="rotate",
        gp_num_models=5,
        rollouts_per_iter=75,
        rollout_length=4,
        bo_iterations=bo_iters,
        initial_points=cfg["initial_hf"],
    )
    rc = result["regret_curve"]
    # SF-DRO has no native cost tracking (single-fidelity, every query
    # costs c_H) -- reconstruct POST-INIT cost_curve (bo.max_iterations is
    # already the post-init iteration count per dro.py's own convention:
    # num_iterations = initial_points + max_iterations).
    cost_curve = [hf_spec_cost * (i + 1) for i in range(len(rc))]
    out = {"regret_curve": rc, "cost_curve": cost_curve}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"{tag} DONE final_regret={rc[-1]:.4f} final_cost={cost_curve[-1]:.1f}",
          flush=True)

else:
    from src.baselines.mf_baselines import (
        MultiFidelityBenchmark, MFGPUCBOptimizer, MFMIGreedyOptimizer, GreedyMFMESOptimizer
    )
    bench = MultiFidelityBenchmark(benchmark)
    common = dict(
        n_initial_hf=cfg["initial_hf"], n_initial_lf=cfg["initial_lf"],
        seed=seed, cost_budget=cfg["cost_budget"],
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
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    rc, cc = result["regret_curve"], result["cost_curve"]
    print(f"{tag} DONE final_regret={rc[-1]:.4f} n_iters={len(rc)} "
          f"final_cost={cc[-1]:.1f}", flush=True)
