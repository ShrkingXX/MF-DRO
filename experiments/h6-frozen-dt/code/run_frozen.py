"""
H1 leak-fix validation under the frozen evaluation. See ../protocol.md.

Grid: 3 methods x 10 seeds = 30 jobs, cost_budget=200 post-init, identical
initial design (initial_hf=36, initial_lf=60) for every method and seed.

Threading discipline (copied from run_experiment_parallel.py, which is the
correct reference): each task runs in a FRESHLY SPAWNED worker that sets
OMP/MKL/VECLIB/NUMEXPR thread caps BEFORE importing numpy/torch, then calls
torch.set_num_threads(). torch.set_num_threads() warns and no-ops once any
torch op has run, so the ordering is load-bearing, not cosmetic.

num_workers x threads_per_worker = 15 x 1 = 15 <= 15 logical cores. OK.
"""
import argparse
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

BENCHMARK = "Hartmann_6D"
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
METHODS = ["MF-DRO"]
COST_BUDGET = 200.0
ITER_CAP = 250          # runaway guard ONLY -- analysis flags any run it binds
INITIAL_HF = 36
INITIAL_LF = 60
X_STAR = [0.2017, 0.1500, 0.4769, 0.2753, 0.3116, 0.6573]
X2 = [0.405, 0.882, 0.846, 0.574, 0.139, 0.038]


def _worker(task):
    method, seed, threads = task

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(threads)

    import sys
    sys.path.insert(0, REPO)
    import torch
    torch.set_num_threads(threads)
    torch.set_default_dtype(torch.float64)

    out_path = os.path.join(RESULTS, f"{method}__seed{seed}.json")
    if os.path.exists(out_path):
        return (method, seed, "SKIPPED", None, 0.0)

    t0 = time.perf_counter()
    try:
        import numpy as np
        from benchmarks import get_benchmark

        hf_spec = get_benchmark(BENCHMARK + "_HF")
        lf_spec = get_benchmark(BENCHMARK + "_LF")
        bounds = torch.tensor([hf_spec["domain_min"], hf_spec["domain_max"]],
                              dtype=torch.float64)

        torch.manual_seed(seed)
        np.random.seed(seed)

        if method == "MF-DRO":
            from dro_runner import _build_mf_dro_config
            from src.policy.mf_dro import DirectMFRegretOptimization
            cfg = _build_mf_dro_config(
                "h1_leak_fix_validation", BENCHMARK, method, seed,
                bo_iterations=ITER_CAP, num_epochs=10,
                minimum_hf_fraction=0.25, real_hf_warmup=2,
                cost_budget=COST_BUDGET,
                initial_hf=INITIAL_HF, initial_lf=INITIAL_LF,
                dkl_threshold=9999, bes_delta=0.0, rollout_length=8,
                known_optimal_x=X_STAR, known_secondary_x=X2,
            )
            cfg.seed = seed
            cfg.freeze_dt_after = 5   # H6: fixed in advance, not tuned
            mf = DirectMFRegretOptimization(
                cfg, hf_spec["make_objective"](), lf_spec["make_objective"](), bounds)
            result = mf.run()
            regret_curve = result["hf_regret_curve"]
            cost_curve = result["cost_curve"]
            fid_trace = result["fidelity_trace"]
        else:
            from src.baselines.mf_baselines import (
                MultiFidelityBenchmark, MFMIGreedyOptimizer, MFGPUCBOptimizer)
            bench = MultiFidelityBenchmark(BENCHMARK)
            common = dict(n_initial_hf=INITIAL_HF, n_initial_lf=INITIAL_LF,
                          seed=seed, cost_budget=COST_BUDGET)
            opt = (MFMIGreedyOptimizer(bench, **common) if method == "MF-MI-Greedy"
                   else MFGPUCBOptimizer(bench, **common))
            result = opt.run(bo_iterations=ITER_CAP)
            regret_curve = result["regret_curve"]
            cost_curve = result["cost_curve"]
            fid_trace = result.get("fidelity_trace", [])

        n_iters = len(regret_curve)
        final_cost = cost_curve[-1] if cost_curve else 0.0
        out = dict(
            result,
            method=method, seed=seed, benchmark=BENCHMARK,
            cost_budget=COST_BUDGET, iter_cap=ITER_CAP,
            final_regret=regret_curve[-1] if regret_curve else None,
            n_iters=n_iters,
            final_cost=final_cost,
            n_hf_queries=int(sum(fid_trace)) if fid_trace else None,
            # True => the runaway guard stopped this run, NOT the cost budget,
            # so it is not cost-matched and the analysis must flag it.
            iter_cap_bound=bool(n_iters >= ITER_CAP and final_cost < COST_BUDGET),
            incumbent_improved_count=sum(
                1 for i in range(1, len(regret_curve))
                if regret_curve[i] < regret_curve[i - 1] - 1e-12),
            wall_time_s=time.perf_counter() - t0,
        )
        os.makedirs(RESULTS, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        return (method, seed, "OK", out["final_regret"], out["wall_time_s"])
    except Exception as e:
        import traceback
        return (method, seed, "FAILED", f"{e}\n{traceback.format_exc()[-800:]}",
                time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--threads-per-worker", type=int, default=1)
    args = ap.parse_args()

    product = args.workers * args.threads_per_worker
    print(f"[compute] num_workers={args.workers} x threads_per_worker="
          f"{args.threads_per_worker} = {product} (limit 15)", flush=True)
    assert product <= 15, f"oversubscribed: {product} > 15"

    # Interleave methods rather than grouping: only 5 of 15 cores are P-cores,
    # so grouping the slow method makes E-core workers set the makespan.
    tasks = [(m, s, args.threads_per_worker) for s in SEEDS for m in METHODS]
    os.makedirs(RESULTS, exist_ok=True)

    t0 = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn")) as ex:
        futs = {ex.submit(_worker, t): t for t in tasks}
        for fut in as_completed(futs):
            method, seed, status, info, wt = fut.result()
            done += 1
            print(f"[{done}/{len(tasks)}] {method} seed{seed}: {status} "
                  f"regret={info if status=='OK' else ''} wall={wt:.1f}s",
                  flush=True)
            if status == "FAILED":
                print(f"    ERROR: {info}", flush=True)
    print(f"[done] {len(tasks)} jobs in {time.perf_counter()-t0:.1f}s wall",
          flush=True)


if __name__ == "__main__":
    main()
