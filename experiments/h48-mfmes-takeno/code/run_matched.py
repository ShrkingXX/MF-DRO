"""MF-MES on the EXACT surrogate mf_dro's h31 teacher used (ensemble member 0),
rather than the KO-GP defaults. Decides whether V6's MF-MES advantage is the
acquisition or the surrogate settings.

Member 0 carries rho_init ~ U(0.3,0.95) and initial_lengthscale = 0.1839 (the
short end of a diversity grid that exists only to decorrelate rollouts for the
DT). We read those off the constructed ensemble rather than re-deriving them.
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore", message=".*power of 2.*")
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
COST_BUDGET = 200.0
NUM_WORKERS = 3


def _worker(seed):
    for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    sys.path.insert(0, REPO)
    import torch
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
    out = os.path.join(RESULTS, f"matched__seed{seed}.json")
    if os.path.exists(out):
        return (seed, "SKIPPED", None, 0.0)
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
        cfg = _build_mf_dro_config("h48_matched", "Hartmann_6D", "MF-MES-matched", seed,
            bo_iterations=1, num_epochs=1, minimum_hf_fraction=0.25, real_hf_warmup=2,
            cost_budget=COST_BUDGET, initial_hf=36, initial_lf=60,
            dkl_threshold=9999, bes_delta=0.0, rollout_length=8)
        cfg.seed = seed
        mf = DirectMFRegretOptimization(cfg, hf["make_objective"](), lf["make_objective"](), bounds)
        mf._sample_initial_points()
        m0 = mf.ko_ensemble[0]
        ko_kwargs = dict(dkl_threshold=9999, rho_init=float(m0.rho_init),
                         initial_lengthscale=(None if getattr(m0, "initial_lengthscale", None) is None
                                              else float(m0.initial_lengthscale)))
        X_lf = torch.stack(list(mf.data_lf_x)); Y_lf = torch.tensor(list(mf.data_lf_y))
        X_hf = torch.stack(list(mf.data_hf_x)); Y_hf = torch.tensor(list(mf.data_hf_y))
        res = run_mf_mes(f_hf=mf.f_hf, f_lf=mf.f_lf, bounds=bounds, c_H=mf.c_H, c_L=mf.c_L,
                         cost_budget=COST_BUDGET, true_opt=-float(cfg.true_opt),
                         X_lf0=X_lf, Y_lf0=Y_lf, X_hf0=X_hf, Y_hf0=Y_hf,
                         single_fidelity=False, seed=seed, verbose=False,
                         ko_kwargs=ko_kwargs)
        rec = {k: v for k, v in res.items() if k not in ("acq_info","X_hf","Y_hf")}
        rec.update(seed=seed, arm="MF-MES-matched", ko_kwargs=ko_kwargs,
                   wall_min=(time.perf_counter()-t0)/60.0)
        tmp = out + ".tmp"
        with open(tmp,"w") as f: json.dump(rec,f)
        os.replace(tmp,out)
        return (seed,"OK",rec["final_regret"],rec["wall_min"])
    except Exception as e:
        import traceback
        with open(os.path.join(RESULTS,f"MFAILED__seed{seed}.txt"),"w") as f:
            f.write(traceback.format_exc())
        return (seed,f"FAILED: {type(e).__name__}: {e}",None,(time.perf_counter()-t0)/60.0)


if __name__ == "__main__":
    todo=[s for s in SEEDS if not os.path.exists(os.path.join(RESULTS,f"matched__seed{s}.json"))]
    print(f"matched-surrogate MF-MES: {len(todo)} seeds, {NUM_WORKERS} workers",flush=True)
    t0=time.time()
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futs={ex.submit(_worker,s):s for s in todo}
        for i,fu in enumerate(as_completed(futs),1):
            seed,st,val,wall=fu.result()
            v=f"{val:.4f}" if val is not None else "--"
            print(f"[{i}/{len(todo)}] seed{seed}: {st} regret={v} ({wall:.1f} min)",flush=True)
    print(f"DONE in {(time.time()-t0)/60:.1f} min",flush=True)
