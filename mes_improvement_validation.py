"""
Full-scale validation of the MES-improvement smoke test findings: does
DRO_MES_Quantile_K30 (quantile RTG schema + K=30 MC samples in the MES
reward) actually beat plain DRO_MES, at real 50-iteration/75-rollout
production scale, with more seeds? Compares against DRO_Improvement
(sparse reward reference) too.

3 benchmarks (Ackley_2D, Eggholder, Hartmann_6D) x 3 variants x 5 seeds = 45
full-scale runs. Writes to a throwaway exp_name, separate from the real
mes_reward/rtg_schema checkpoints.
"""
import argparse
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

EXP_NAME = "mes_improvement_validation"
SEEDS = [42, 43, 44, 45, 46]
BENCHMARKS = ["Ackley_2D", "Eggholder", "Hartmann_6D"]

FULL_HPARAMS = dict(
    gp_num_models=5, rollouts_per_iter=75, rollout_length=4,
    bo_iterations=50, initial_points=5,
    dt_hidden=128, dt_layers=4, dt_heads=4,
    gp_kernel="rbf", gp_ard=True, verbose=False,
)

VARIANTS = {
    "DRO_Improvement":       dict(use_mes_reward=False, rtg_schema="floored", alpha_floor=0.5),
    "DRO_MES":               dict(use_mes_reward=True,  rtg_schema="floored", alpha_floor=0.5, mes_k=10),
    "DRO_MES_Quantile_K30":  dict(use_mes_reward=True,  rtg_schema="quantile", alpha_inference=0.5,
                                   lambda_rtg=1.0, rtg_warmup=3, mes_k=30),
}


def _worker_run(task):
    benchmark, variant_name, seed, threads_per_worker = task
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(threads_per_worker)
    import torch
    torch.set_num_threads(threads_per_worker)

    t0 = time.perf_counter()
    try:
        from dro_runner import run_single_seed
        variant_kwargs = dict(VARIANTS[variant_name])
        run_single_seed(
            exp_name=EXP_NAME, benchmark_name=benchmark, variant_name=variant_name, seed=seed,
            **variant_kwargs, **FULL_HPARAMS,
        )
        return (benchmark, variant_name, seed, "OK", None, time.perf_counter() - t0)
    except Exception as e:
        import traceback
        return (benchmark, variant_name, seed, "FAILED", f"{e}\n{traceback.format_exc()}", time.perf_counter() - t0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    args = parser.parse_args()

    from checkpoint import setup_dirs, is_completed, log_global
    setup_dirs(EXP_NAME)

    all_tasks = [(b, v, s, args.threads_per_worker) for b in BENCHMARKS for v in VARIANTS for s in SEEDS]
    tasks = [t for t in all_tasks if not is_completed(EXP_NAME, t[0], t[1], t[2])]
    total_grid = len(all_tasks)
    total = len(tasks)
    print(f"{total}/{total_grid} runs remaining (already-completed runs skipped).", flush=True)
    print(f"Full hyperparameters: {FULL_HPARAMS}", flush=True)

    if total == 0:
        print("Nothing to do.", flush=True)
        return

    log_global(EXP_NAME, f"VALIDATION STARTED total={total}")
    start = time.perf_counter()
    completed = failed = 0

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as executor:
        futures = [executor.submit(_worker_run, t) for t in tasks]
        for fut in as_completed(futures):
            benchmark, variant_name, seed, status, err, run_time = fut.result()
            if status == "OK":
                completed += 1
            else:
                failed += 1
            done = completed + failed
            elapsed = time.perf_counter() - start
            rate = done / elapsed if elapsed > 0 else 1e-9
            eta_min = (total - done) / rate / 60 if rate > 0 else float('inf')
            print(f"[{done}/{total}] {status} {benchmark} {variant_name} seed{seed} ({run_time:.1f}s) "
                  f"| ETA {eta_min:.1f} min" + (f"\n  ERROR: {err}" if status == "FAILED" else ""), flush=True)

    total_time = time.perf_counter() - start
    log_global(EXP_NAME, f"VALIDATION FINISHED completed={completed} failed={failed} time={total_time:.1f}s")
    print(f"\nDone in {total_time/60:.1f} min. completed={completed} failed={failed}", flush=True)


if __name__ == '__main__':
    main()
