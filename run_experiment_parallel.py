"""
Parallel launcher for run_experiment's (benchmark, variant, seed) grid.

Why not just run run_experiment.py N times in the background? PyTorch (and
its BLAS backend) default to using *all* logical cores for intra-op
parallelism within a single process (confirmed on this machine:
torch.get_num_threads() == 15, the full core count). DRO's actual per-run
compute is dominated by many small matrix ops (GP ensembles fit on a
handful of points, a 128-hidden-unit/4-layer Decision Transformer on
batch_size=32) -- too small to benefit much from >2-4 threads of intra-op
parallelism. So naively launching several full-thread-count processes at
once causes severe oversubscription/contention and can end up *slower* than
running them one at a time.

This script instead shards the run grid across N worker processes, each
explicitly capped to a small thread count (via env vars, set before
numpy/torch are imported in that worker -- spawn start method guarantees a
fresh interpreter per worker, so this is safe), trading thread-level
parallelism within a run for process-level parallelism across runs, which is
the right axis for an embarrassingly-parallel grid like this one.

Usage:
    python run_experiment_parallel.py --experiment mes_reward \\
        --workers 8 --threads-per-worker 1

    # Validate on a small subset first:
    python run_experiment_parallel.py --experiment mes_reward \\
        --workers 4 --threads-per-worker 1 --seeds 42 43 44 --benchmarks Ackley_2D
"""
import argparse
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def _worker_run(task):
    """
    Runs in a freshly-spawned worker process. Sets thread-count env vars
    BEFORE importing numpy/torch/dro_runner, so the BLAS backend and PyTorch
    both pick up the cap at library-init time.
    """
    benchmark, variant_name, seed, exp_name, variant_kwargs, threads_per_worker = task

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(threads_per_worker)

    import torch
    torch.set_num_threads(threads_per_worker)

    from dro_runner import run_single_seed
    t0 = time.perf_counter()
    try:
        run_single_seed(
            exp_name=exp_name, benchmark_name=benchmark,
            variant_name=variant_name, seed=seed, **variant_kwargs,
        )
        return (benchmark, variant_name, seed, "OK", None, time.perf_counter() - t0)
    except Exception as e:
        return (benchmark, variant_name, seed, "FAILED", str(e), time.perf_counter() - t0)


def run_experiment_parallel(exp_name, benchmarks, variants, seeds, variant_configs,
                             num_workers, threads_per_worker):
    from checkpoint import setup_dirs, is_completed, log_global

    setup_dirs(exp_name)

    tasks = []
    for benchmark in benchmarks:
        for variant_name in variants:
            for seed in seeds:
                if is_completed(exp_name, benchmark, variant_name, seed):
                    continue
                tasks.append((benchmark, variant_name, seed, exp_name,
                              variant_configs[variant_name], threads_per_worker))

    total_grid = len(benchmarks) * len(variants) * len(seeds)
    total = len(tasks)
    logical_cores = os.cpu_count()
    requested = num_workers * threads_per_worker
    print(f"{total}/{total_grid} runs remaining (already-completed runs skipped).")
    print(f"Launching {num_workers} worker processes x {threads_per_worker} thread(s) each "
          f"= {requested} of {logical_cores} logical cores requested.")
    if requested > logical_cores:
        print(f"WARNING: requested {requested} > {logical_cores} available logical cores -- "
              f"you will oversubscribe. Consider lowering --workers or --threads-per-worker.")
    if total == 0:
        print("Nothing to do -- all requested runs are already checkpointed.")
        return

    log_global(exp_name, f"PARALLEL RUN STARTED total={total} workers={num_workers} threads_per_worker={threads_per_worker}")
    start = time.perf_counter()
    completed = failed = 0

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        futures = [executor.submit(_worker_run, t) for t in tasks]
        for fut in as_completed(futures):
            benchmark, variant_name, seed, status, err, run_time = fut.result()
            if status == "OK":
                completed += 1
            else:
                failed += 1
                log_global(exp_name, f"FAILED {benchmark} {variant_name} seed{seed} error={err}")

            done = completed + failed
            elapsed = time.perf_counter() - start
            rate = done / elapsed if elapsed > 0 else 1e-9
            eta_min = (total - done) / rate / 60 if rate > 0 else float('inf')
            tag = "OK" if status == "OK" else f"FAILED ({err})"
            print(f"[{done}/{total}] {tag} {benchmark} {variant_name} seed{seed} "
                  f"({run_time:.1f}s) | completed={completed} failed={failed} | ETA {eta_min:.1f} min")

    total_time = time.perf_counter() - start
    log_global(exp_name, f"PARALLEL RUN FINISHED total={total} completed={completed} failed={failed} time={total_time:.1f}s")
    print(f"\nDone in {total_time / 60:.1f} min. completed={completed} failed={failed}")


if __name__ == '__main__':
    from run_experiment import EXPERIMENTS, SEEDS

    parser = argparse.ArgumentParser(description="Parallel (thread-capped, process-sharded) DRO experiment runner.")
    parser.add_argument("--experiment", required=True, choices=list(EXPERIMENTS.keys()))
    parser.add_argument("--workers", type=int, required=True, help="Number of concurrent worker processes.")
    parser.add_argument("--threads-per-worker", type=int, default=1, help="Torch/BLAS thread cap per worker.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--benchmarks", type=str, nargs="+", default=None)
    parser.add_argument("--variants", type=str, nargs="+", default=None)
    args = parser.parse_args()

    spec = EXPERIMENTS[args.experiment]
    run_experiment_parallel(
        exp_name=args.experiment,
        benchmarks=args.benchmarks or spec["benchmarks"],
        variants=args.variants or spec["variants"],
        seeds=args.seeds or SEEDS,
        variant_configs=spec["variant_configs"],
        num_workers=args.workers,
        threads_per_worker=args.threads_per_worker,
    )
