"""H59 worker: SF-DRO vs SF-MES, single fidelity, cost budget matched to h57.

SF-DRO gotchas (first run of this class in the repo, all hit during smoke):
  - rtg_schema must be one of fixed/dynamic/floored/quantile/joint/entropy_joint
  - entry point is run_optimization(); results live in iteration_log_history
  - checkpoint.setup_dirs(exp_name) must run first or the first log write crashes
"""
import os, sys, json, time
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v]="1"
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO); os.chdir(REPO)
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from benchmarks import get_benchmark
RES=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results")
BUDGET=200.0
NHF={"Currin_2D":5,"Hartmann_6D":6,"Borehole_8D":10}      # matches h57's n_hf

def _code():
    import subprocess
    def g(*a):
        try: return subprocess.run(["git",*a],cwd=REPO,capture_output=True,text=True,timeout=20).stdout.strip()
        except Exception: return ""
    dirty=g("status","--porcelain","src","dro_runner.py","benchmarks.py")
    return dict(commit=g("rev-parse","HEAD"),dirty=bool(dirty))

def run(bench, arm, seed):
    t0=time.time(); spec=get_benchmark(f"{bench}_HF")
    c_H=float(spec["cost"]); n_iter=int(BUDGET//c_H)      # SF: every query costs c_H
    torch.manual_seed(seed); np.random.seed(seed)
    if arm=="SF-DRO":
        from checkpoint import setup_dirs
        from dro_runner import _build_dro_config
        from src.policy.dro import DirectRegretOptimization
        exp=f"h59_{bench}_{seed}"; setup_dirs(exp)
        cfg=_build_dro_config(exp,bench,"sfdro",seed,
            use_mes_reward=False, rtg_schema="fixed", alpha_floor=0.5,
            alpha_inference=None, lambda_rtg=1.0, rtg_warmup=3, benchmark_spec=spec,
            gp_num_models=10, rollouts_per_iter=200, rollout_length=8,
            bo_iterations=n_iter, initial_points=NHF[bench],
            dt_hidden=64, dt_layers=2, dt_heads=4, dt_lr=1e-3,
            gp_kernel="rbf", gp_ard=True, verbose=False)
        cfg.seed=seed
        dro=DirectRegretOptimization(cfg, spec["make_objective"]())
        dro.run_optimization()
        h=dro.iteration_log_history
        curve=[float(d["regret"]) for d in h]
        qx=[list(map(float,np.asarray(d["x"]).reshape(-1))) for d in h if "x" in d]
        qy=[float(d["y"]) for d in h if "y" in d]
    else:
        from src.baselines.mf_baselines import MultiFidelityBenchmark
        from src.baselines.additive_mes import SFMESOptimizer
        b=MultiFidelityBenchmark(bench)
        opt=SFMESOptimizer(b, n_initial_hf=NHF[bench], n_initial_lf=0,
                           seed=seed, cost_budget=BUDGET)
        qx,qy=[],[]
        _orig=opt._add_obs
        def _add(x,y,fid):
            qx.append(list(map(float,np.asarray(x if not torch.is_tensor(x) else x.cpu().numpy()).reshape(-1))))
            qy.append(float(y)); return _orig(x,y,fid)
        opt._add_obs=_add
        r=opt.run(bo_iterations=4000)
        curve=[float(v) for v in (r.get("hf_regret_curve") or r.get("regret_curve") or [])]
    return dict(bench=bench,arm=arm,seed=seed,n_iter_cap=n_iter,c_H=c_H,budget=BUDGET,
                regret_curve=curve, final_regret=(curve[-1] if curve else float("nan")),
                n_improvements=sum(1 for i in range(1,len(curve)) if curve[i]<curve[i-1]-1e-9),
                query_x=qx, query_y=qy, n_queries=len(qy),
                wall_s=round(time.time()-t0,1), code=_code())

if __name__=="__main__":
    bench,arm,seed=sys.argv[1],sys.argv[2],int(sys.argv[3])
    out=os.path.join(RES,f"{bench}__{arm}__seed{seed}.json")
    r=run(bench,arm,seed)
    tmp=out+".tmp"; json.dump(r,open(tmp,"w"),default=float); os.replace(tmp,out)
    print(f"[done] {bench} {arm} s{seed} regret={r['final_regret']:.4f} "
          f"iters={len(r['regret_curve'])} improv={r['n_improvements']} "
          f"wall={r['wall_s']/60:.1f}m",flush=True)
