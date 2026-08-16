"""
Tests candidate fixes for MES reward's underperformance (see session discussion):
  1. RTG quantile schema combined with MES reward (untested combination --
     Experiment 2 held reward fixed at sparse improvement to isolate the RTG
     question, so use_mes_reward=True + rtg_schema="quantile" has never run).
  2. Increased K (Monte Carlo sample count in compute_mes_reward's Term2),
     10 -> 30, to reduce reward-estimation noise. Free-ish: no extra GP
     evaluations, since gamma/Phi/phi/H are already vectorized across K.
  3. Both combined.

Compares against the current DRO_MES (floored, K=10) and DRO_Improvement as
references, on 3 benchmarks chosen to span where MES currently ties/loses:
Ackley_2D (DRO wins), Eggholder (NaiveBO wins clearly), Hartmann_6D (NaiveBO
wins, 6D action space). Reduced iteration/rollout counts (same as the earlier
smoke test) so this finishes in ~10-15 min; full DT/GP architecture unchanged.

Writes to a throwaway exp_name so it never touches the real mes_reward/
rtg_schema checkpoints.
"""
import argparse
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from benchmarks import BENCHMARKS

EXP_NAME = "mes_improvement_test"
SEEDS = [42, 43]
TEST_BENCHMARKS = ["Ackley_2D", "Eggholder", "Hartmann_6D"]

SMOKE_HPARAMS = dict(
    gp_num_models=5, rollouts_per_iter=20, rollout_length=4,
    bo_iterations=8, initial_points=5,
    dt_hidden=128, dt_layers=4, dt_heads=4,
    gp_kernel="rbf", gp_ard=True, verbose=False,
)

VARIANTS = {
    "DRO_Improvement":         dict(use_mes_reward=False, rtg_schema="floored", alpha_floor=0.5),
    "DRO_MES":                 dict(use_mes_reward=True,  rtg_schema="floored", alpha_floor=0.5, mes_k=10),
    "DRO_MES_K30":             dict(use_mes_reward=True,  rtg_schema="floored", alpha_floor=0.5, mes_k=30),
    "DRO_MES_Quantile":        dict(use_mes_reward=True,  rtg_schema="quantile", alpha_inference=0.5,
                                     lambda_rtg=1.0, rtg_warmup=3, mes_k=10),
    "DRO_MES_Quantile_K30":    dict(use_mes_reward=True,  rtg_schema="quantile", alpha_inference=0.5,
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
            **variant_kwargs, **SMOKE_HPARAMS,
        )
        return (benchmark, variant_name, seed, "OK", None, time.perf_counter() - t0)
    except Exception as e:
        import traceback
        return (benchmark, variant_name, seed, "FAILED", f"{e}\n{traceback.format_exc()}", time.perf_counter() - t0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    args = parser.parse_args()

    from checkpoint import setup_dirs, log_global
    setup_dirs(EXP_NAME)

    tasks = [(b, v, s, args.threads_per_worker) for b in TEST_BENCHMARKS for v in VARIANTS for s in SEEDS]
    total = len(tasks)
    print(f"MES improvement test: {total} tasks ({len(TEST_BENCHMARKS)} benchmarks x {len(VARIANTS)} variants x {len(SEEDS)} seeds)")
    print(f"Reduced hyperparameters: {SMOKE_HPARAMS}")

    log_global(EXP_NAME, f"TEST STARTED total={total}")
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
            print(f"[{done}/{total}] {status} {benchmark} {variant_name} seed{seed} ({run_time:.1f}s) "
                  f"| elapsed={elapsed:.0f}s" + (f"\n  ERROR: {err}" if status == "FAILED" else ""))

    total_time = time.perf_counter() - start
    log_global(EXP_NAME, f"TEST FINISHED completed={completed} failed={failed} time={total_time:.1f}s")
    print(f"\nDone in {total_time/60:.1f} min. completed={completed} failed={failed}")


if __name__ == '__main__':
    main()
