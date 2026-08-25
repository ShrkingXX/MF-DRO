"""V6: MF-MES must beat SF-MES (highest fidelity only) on Hartmann 6D at
fixed total cost. If it does not, the implementation is broken.

Both arms share the identical initial design (36 HF / 60 LF, drawn by
mf_dro's own _sample_initial_points so it matches every other experiment in
this project) and the identical post-init cost budget of 200. Both use the
same KO-GP surrogate and the same acquisition code; the ONLY difference is
that SF-MES may not choose the low fidelity. That isolates the multi-fidelity
benefit rather than confounding it with a surrogate change.
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore", message=".*power of 2.*")
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
ARMS = ["MF-MES", "SF-MES"]
COST_BUDGET = 200.0
NUM_WORKERS = 3


def _worker(task):
    arm, seed = task
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    sys.path.insert(0, REPO)
    import torch
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64)

    out = os.path.join(RESULTS, f"v6__{arm}__seed{seed}.json")
    if os.path.exists(out):
        return (arm, seed, "SKIPPED", None, 0.0)
    t0 = time.perf_counter()
    try:
        import numpy as np
        from benchmarks import get_benchmark
        from dro_runner import _build_mf_dro_config
        from src.policy.mf_dro import DirectMFRegretOptimization
        from src.baselines.mf_mes_takeno import run_mf_mes

        hf = get_benchmark("Hartmann_6D_HF"); lf = get_benchmark("Hartmann_6D_LF")
        bounds = torch.tensor([hf["domain_min"], hf["domain_max"]], dtype=torch.float64)
        torch.manual_seed(seed); np.random.seed(seed)

        cfg = _build_mf_dro_config("h48_v6", "Hartmann_6D", arm, seed,
            bo_iterations=1, num_epochs=1, minimum_hf_fraction=0.25,
            real_hf_warmup=2, cost_budget=COST_BUDGET, initial_hf=36,
            initial_lf=60, dkl_threshold=9999, bes_delta=0.0, rollout_length=8)
        cfg.seed = seed
        # Use mf_dro ONLY to draw the shared initial design, then discard it.
        mf = DirectMFRegretOptimization(cfg, hf["make_objective"](),
                                        lf["make_objective"](), bounds)
        mf._sample_initial_points()
        X_lf = torch.stack(list(mf.data_lf_x)); Y_lf = torch.tensor(list(mf.data_lf_y))
        X_hf = torch.stack(list(mf.data_hf_x)); Y_hf = torch.tensor(list(mf.data_hf_y))
        true_opt = -float(cfg.true_opt)     # cfg stores the negated optimum

        res = run_mf_mes(
            f_hf=mf.f_hf, f_lf=mf.f_lf, bounds=bounds, c_H=mf.c_H, c_L=mf.c_L,
            cost_budget=COST_BUDGET, true_opt=true_opt,
            X_lf0=X_lf, Y_lf0=Y_lf, X_hf0=X_hf, Y_hf0=Y_hf,
            single_fidelity=(arm == "SF-MES"), seed=seed, verbose=False,
            # D1: hold the surrogate identical to every other experiment here.
            # Without this the KO-GP defaults to dkl_threshold=30 and switches
            # on deep kernel learning at n_hf=36, changing the surrogate and
            # confounding the acquisition comparison.
            ko_kwargs=dict(dkl_threshold=9999))

        rec = {k: v for k, v in res.items() if k not in ("acq_info", "X_hf", "Y_hf")}
        rec.update(arm=arm, seed=seed, benchmark="Hartmann_6D",
                   wall_min=(time.perf_counter() - t0) / 60.0,
                   n_lbfgs_improved=int(sum(a["n_improved"] for a in res["acq_info"])),
                   n_acq_calls=len(res["acq_info"]),
                   mean_pool_vs_refined=float(np.mean(
                       [a["pool_best"] for a in res["acq_info"]])) if res["acq_info"] else float("nan"))
        tmp = out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rec, f)
        os.replace(tmp, out)
        return (arm, seed, "OK", rec["final_regret"], rec["wall_min"])
    except Exception as e:
        import traceback
        with open(os.path.join(RESULTS, f"V6FAILED__{arm}__seed{seed}.txt"), "w") as f:
            f.write(traceback.format_exc())
        return (arm, seed, f"FAILED: {type(e).__name__}: {e}", None,
                (time.perf_counter() - t0) / 60.0)


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    todo = [(a, s) for a in ARMS for s in SEEDS
            if not os.path.exists(os.path.join(RESULTS, f"v6__{a}__seed{s}.json"))]
    print(f"V6: {len(todo)} jobs, {NUM_WORKERS} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futs = {ex.submit(_worker, t): t for t in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            arm, seed, st, val, wall = fu.result()
            v = f"{val:.4f}" if val is not None else "--"
            print(f"[{i}/{len(todo)}] {arm} seed{seed}: {st} regret={v} "
                  f"({wall:.1f} min, elapsed {(time.time()-t0)/60:.1f})", flush=True)
    print(f"V6 DONE in {(time.time()-t0)/60:.1f} min", flush=True)
