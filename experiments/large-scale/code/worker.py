"""One (benchmark, method, seed) job. Thread caps set before numpy/torch."""
import os, sys, json, time
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
import config as C

def run(bench, method, seed):
    spec = C.BENCHMARKS[bench]
    hf = get_benchmark(f"{bench}_HF"); lf = get_benchmark(f"{bench}_LF")
    bounds = torch.tensor([hf["domain_min"], hf["domain_max"]], dtype=torch.float64)
    budget = spec["budget"] if C.BUDGET_MODE == "cost" else float("inf")
    iters  = C.ITER_CAP if C.BUDGET_MODE == "cost" else C.N_ITERS
    torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.time()

    if method in ("MF-DRO", "MF-MES-Greedy"):
        from src.policy.mf_dro import DirectMFRegretOptimization, compute_joint_mf_mes
        cfg = _build_mf_dro_config(
            "large_scale", bench, method, seed,
            bo_iterations=iters, num_epochs=10, minimum_hf_fraction=0.25,
            real_hf_warmup=2, cost_budget=budget,
            initial_hf=spec["n_hf"], initial_lf=spec["n_lf"],
            dkl_threshold=9999, bes_delta=0.0, rollout_length=8)
        cfg.seed = seed; cfg.rollout_reward = C.ROLLOUT_REWARD
        mf = DirectMFRegretOptimization(cfg, hf["make_objective"](), lf["make_objective"](), bounds)
        if method == "MF-MES-Greedy":
            # Teacher acquisition decides; the DT is still trained so the RNG
            # stream (and hence every candidate pool) matches the MF-DRO arm.
            def _teacher(state, rtg, btg, timestep=0, use_candidate_scoring=False,
                          candidate_features=None, fidelity_sampling=True, hist=None):
                cf = candidate_features.double(); lo, hi = mf.bounds[0], mf.bounds[1]
                X = lo + (hi - lo) * cf[:, :mf.d]
                x_raw, ell, _ = compute_joint_mf_mes(mf.ko_ensemble[0], X, mf.c_H, mf.c_L)
                mf.dt.last_p_pred = float(ell)
                return ((x_raw - lo) / (hi - lo)).clamp(0, 1).float(), int(ell)
            mf.dt.propose_mf = _teacher
        res = mf.run()
    else:
        from src.baselines.mf_baselines import (MultiFidelityBenchmark,
                                                 MFMIGreedyOptimizer, MFGPUCBOptimizer)
        b = MultiFidelityBenchmark(bench)
        common = dict(n_initial_hf=spec["n_hf"], n_initial_lf=spec["n_lf"],
                      seed=seed, cost_budget=budget)
        opt = (MFMIGreedyOptimizer(b, **common) if method == "MF-MI-Greedy"
               else MFGPUCBOptimizer(b, **common))
        res = opt.run(bo_iterations=iters)
        res.setdefault("hf_regret_curve", res.get("regret_curve"))
    res["_wall_s"] = round(time.time() - t0, 1)
    res["_meta"] = dict(bench=bench, method=method, seed=seed,
                        budget_mode=C.BUDGET_MODE, budget=spec["budget"],
                        n_hf=spec["n_hf"], n_lf=spec["n_lf"])
    return res

if __name__ == "__main__":
    bench, method, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"{bench}__{method}__seed{seed}.json")
    r = run(bench, method, seed)
    json.dump(r, open(out, "w"), default=float)
    c = r.get("cost_curve") or [0]
    print(f"[done] {bench} {method} seed{seed}  regret={r['hf_regret_curve'][-1]:.4f} "
          f"cost={c[-1]:.1f} iters={len(r['hf_regret_curve'])} wall={r['_wall_s']}s")
