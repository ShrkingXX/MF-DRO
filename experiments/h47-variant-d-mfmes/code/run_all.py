"""
H47 variant D driver. See ../protocol.md.

Per-seed checkpointing: each seed writes its own JSON the moment it finishes,
so intermediate results are readable and a kill loses at most the in-flight
jobs. Re-running skips seeds that already have a result file.

Threading: each task runs in a freshly spawned worker that sets OMP/MKL/
VECLIB/NUMEXPR caps BEFORE importing numpy/torch (torch.set_num_threads()
no-ops once any torch op has run, so the ordering is load-bearing).
num_workers x threads_per_worker = 4 x 1 = 4; h45 is using 10 and h46 1,
so total <= 15 logical cores. OK.
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))

BENCHMARK = "Hartmann_6D"
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
COST_BUDGET = 200.0
ITER_CAP = 250
INITIAL_HF, INITIAL_LF = 36, 60
X_STAR = [0.2017, 0.1500, 0.4769, 0.2753, 0.3116, 0.6573]
X2 = [0.405, 0.882, 0.846, 0.574, 0.139, 0.038]
NUM_WORKERS = 4


def _worker(seed):
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    sys.path.insert(0, REPO)
    import torch
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)

    out = os.path.join(RESULTS, f"variantD__seed{seed}.json")
    if os.path.exists(out):
        return (seed, "SKIPPED", None, 0.0)

    t0 = time.perf_counter()
    try:
        import numpy as np
        from benchmarks import get_benchmark
        from dro_runner import _build_mf_dro_config
        from src.policy.mf_dro import DirectMFRegretOptimization
        from src.policy.mf_mes_optimized import optimize_mf_mes

        hf_spec = get_benchmark(BENCHMARK + "_HF")
        lf_spec = get_benchmark(BENCHMARK + "_LF")
        bounds = torch.tensor([hf_spec["domain_min"], hf_spec["domain_max"]],
                              dtype=torch.float64)
        torch.manual_seed(seed); np.random.seed(seed)

        cfg = _build_mf_dro_config(
            "h47_variant_d", BENCHMARK, "variantD", seed,
            bo_iterations=ITER_CAP, num_epochs=10,
            minimum_hf_fraction=0.25, real_hf_warmup=2,
            cost_budget=COST_BUDGET, initial_hf=INITIAL_HF, initial_lf=INITIAL_LF,
            dkl_threshold=9999, bes_delta=0.0, rollout_length=8,
            known_optimal_x=X_STAR, known_secondary_x=X2)
        cfg.seed = seed
        cfg.rollout_reward = "mes_entropy"

        mf = DirectMFRegretOptimization(
            cfg, hf_spec["make_objective"](), lf_spec["make_objective"](), bounds)

        # THE ONE VARIABLE: the proposal is a real maximization of the Takeno
        # acquisition over the continuous box, not an argmax over 200 draws.
        # candidate_features is ignored on purpose -- D does not use a pool.
        gen = torch.Generator().manual_seed(seed)
        fid_choices = []

        def _propose(state, rtg, btg, timestep=0, use_candidate_scoring=False,
                     candidate_features=None, fidelity_sampling=True, hist=None):
            lo, hi = mf.bounds[0], mf.bounds[1]
            x_raw, ell, _ = optimize_mf_mes(
                mf.ko_ensemble[0], mf.bounds, mf.c_H, mf.c_L, gen=gen)
            fid_choices.append(int(ell))
            # run loop logs dt.last_p_pred after every proposal; D has no
            # fidelity probability, so record the deterministic choice.
            mf.dt.last_p_pred = float(ell)
            x_norm = ((x_raw - lo) / (hi - lo)).clamp(0.0, 1.0).float()
            return x_norm, int(ell)

        mf.dt.propose_mf = _propose
        res = mf.run()

        rec = {
            "method": "variantD", "seed": seed, "benchmark": BENCHMARK,
            "hf_regret_curve": [float(v) for v in res["hf_regret_curve"]],
            "cost_curve": [float(v) for v in res["cost_curve"]],
            "final_hf_regret": float(res["hf_regret_curve"][-1]),
            "n_iters": len(res["hf_regret_curve"]),
            "n_HF": int(sum(fid_choices)), "n_LF": int(len(fid_choices) - sum(fid_choices)),
            "iter_cap_bound": len(res["hf_regret_curve"]) >= ITER_CAP,
            "wall_min": (time.perf_counter() - t0) / 60.0,
        }
        for k in ("inference_regret_curve", "hf_inference_regret_curve"):
            if k in res:
                rec[k] = [float(v) for v in res[k]]
        tmp = out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rec, f)
        os.replace(tmp, out)                      # atomic: no partial JSON
        return (seed, "OK", rec["final_hf_regret"], rec["wall_min"])
    except Exception as e:
        import traceback
        with open(os.path.join(RESULTS, f"FAILED__seed{seed}.txt"), "w") as f:
            f.write(traceback.format_exc())
        return (seed, f"FAILED: {type(e).__name__}: {e}", None, (time.perf_counter()-t0)/60.0)


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    todo = [s for s in SEEDS
            if not os.path.exists(os.path.join(RESULTS, f"variantD__seed{s}.json"))]
    print(f"H47 variant D: {len(todo)} seeds to run, {NUM_WORKERS} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futs = {ex.submit(_worker, s): s for s in todo}
        done = 0
        for fu in as_completed(futs):
            seed, status, val, wall = fu.result()
            done += 1
            v = f"{val:.4f}" if val is not None else "--"
            print(f"[{done}/{len(todo)}] seed {seed}: {status} regret={v} "
                  f"({wall:.1f} min, elapsed {(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"ALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)
