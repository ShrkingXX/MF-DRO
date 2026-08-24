"""
Worker for the KO-MES paper experiment (exp_name "ko_mes_paper").

    .venv/bin/python3 _ko_mes_worker.py <METHOD> <BENCHMARK> <SEED>

Protocol, applied identically to every method:
  * Song 2019 asymmetric init: 3*d HF + 5*d LF points.
  * Sequential max-variance initial design (src/utils/init_design.py) --
    same design for every method at a given (benchmark, seed, fidelity), so
    the Additive-MES vs KO-MES gap is attributable to the surrogate.
  * Post-init cost budget of 100 * c_H.
  * DKL disabled everywhere (Experiment B is gated on a separate DKL
    diagnostic; the KO GP's own default would otherwise switch the
    surrogate to a deep kernel partway through every run).

BO_ITERATIONS IS NOT A FREE PARAMETER. Every iteration costs at least c_L,
so a run needs up to cost_budget/c_L iterations if it selects LF every time.
Stage 2 used a flat bo_iterations=500, which silently truncated Hartmann_6D
(budget 800, c_L=1): 4 of 5 Greedy-MES seeds and all 5 MF-GP-UCB seeds
stopped at exactly 500 iterations having spent ~500 of their 800 budget, so
the @800 (100x c_H) checkpoint was never actually reached on that benchmark.
The cap is derived from the budget below so the budget is always the binding
constraint.
"""
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

from benchmarks import get_benchmark

EXP_NAME = "ko_mes_paper"
SEEDS = (42, 43, 44, 45, 46)
BENCHMARKS = ("Currin_2D", "Hartmann_6D", "Borehole_8D")
METHODS = ("KO-MES", "Additive-MES", "Additive-MES-Song", "SF-MES",
           "MF-GP-UCB", "MF-MI-Greedy")


def benchmark_config(name):
    """Song 2019 init counts and the 100*c_H budget, derived from the registry."""
    hf = get_benchmark(f"{name}_HF")
    lf = get_benchmark(f"{name}_LF")
    d = hf["dim"]
    cost_budget = 100.0 * hf["cost"]
    return dict(
        d=d,
        initial_hf=3 * d,
        initial_lf=5 * d,
        c_H=hf["cost"],
        c_L=lf["cost"],
        cost_budget=cost_budget,
        # +1 so the budget check, not the safety cap, always terminates.
        bo_iterations=int(cost_budget / lf["cost"]) + 1,
    )


def build_optimizer(method, bench, cfg, seed):
    from src.baselines.mf_baselines import (
        MFGPUCBOptimizer, MFMIGreedyOptimizer, GreedyMFMESOptimizer,
    )
    from src.baselines.additive_mes import AdditiveMESOptimizer, SFMESOptimizer

    common = dict(
        n_initial_hf=cfg["initial_hf"], n_initial_lf=cfg["initial_lf"],
        seed=seed, cost_budget=cfg["cost_budget"], use_sequential_init=True,
    )
    if method == "KO-MES":
        return GreedyMFMESOptimizer(bench, use_dkl=False, **common)
    if method == "Additive-MES":
        return AdditiveMESOptimizer(bench, variant="rho1", **common)
    if method == "Additive-MES-Song":
        return AdditiveMESOptimizer(bench, variant="song", **common)
    if method == "SF-MES":
        return SFMESOptimizer(bench, n_initial_hf=cfg["initial_hf"], seed=seed,
                              cost_budget=cfg["cost_budget"],
                              use_sequential_init=True)
    if method == "MF-GP-UCB":
        return MFGPUCBOptimizer(bench, **common)
    if method == "MF-MI-Greedy":
        return MFMIGreedyOptimizer(bench, **common)
    raise ValueError(f"unknown method {method!r}, expected one of {METHODS}")


def main():
    method, benchmark, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    cfg = benchmark_config(benchmark)

    out_dir = os.path.join("results", EXP_NAME, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{method}__{benchmark}__seed{seed}.json")
    tag = f"[{method} {benchmark} seed{seed}]"

    if os.path.exists(out_path):
        print(f"{tag} SKIPPED (already exists)", flush=True)
        return

    print(f"{tag} START budget={cfg['cost_budget']:.0f} "
          f"init={cfg['initial_hf']}HF/{cfg['initial_lf']}LF "
          f"max_iters={cfg['bo_iterations']}", flush=True)

    from src.baselines.mf_baselines import MultiFidelityBenchmark
    bench = MultiFidelityBenchmark(benchmark)
    opt = build_optimizer(method, bench, cfg, seed)

    t0 = time.time()
    result = opt.run(bo_iterations=cfg["bo_iterations"])
    elapsed = time.time() - t0

    result["method"] = method
    result["benchmark"] = benchmark
    result["seed"] = seed
    result["config"] = cfg
    result["elapsed_sec"] = elapsed

    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(result, f)
    os.replace(tmp_path, out_path)  # atomic: no partial file if killed mid-write

    rc, cc = result["regret_curve"], result["cost_curve"]
    print(f"{tag} DONE final_regret={rc[-1]:.5f} iters={len(rc)} "
          f"final_cost={cc[-1]:.1f} lf_frac={result.get('lf_fraction')} "
          f"rho={result.get('final_rho')} t={elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
