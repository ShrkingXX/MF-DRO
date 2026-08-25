import os, sys, json, time
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v]="1"
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config

BUDGET=200.0
SPEC={"Currin_2D":dict(n_hf=5,n_lf=15),
      "Hartmann_6D":dict(n_hf=6,n_lf=45),
      "Borehole_8D":dict(n_hf=10,n_lf=20)}

def run(bench, method, seed):
    sp=SPEC[bench]; t0=time.time()
    hf=get_benchmark(f"{bench}_HF"); lf=get_benchmark(f"{bench}_LF")
    bounds=torch.tensor([hf["domain_min"],hf["domain_max"]],dtype=torch.float64)
    torch.manual_seed(seed); np.random.seed(seed)
    if method in ("MF-DRO","MF-MES"):
        from src.policy.mf_dro import DirectMFRegretOptimization
        cfg=_build_mf_dro_config("h57",bench,method,seed,bo_iterations=4000,num_epochs=10,
            minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=BUDGET,
            initial_hf=sp["n_hf"],initial_lf=sp["n_lf"],dkl_threshold=9999,
            bes_delta=0.0,rollout_length=8)
        cfg.seed=seed; cfg.rollout_reward="mes_entropy"
        cfg.use_candidate_scoring=False          # regression head (the new default)
        mf=DirectMFRegretOptimization(cfg,hf["make_objective"](),lf["make_objective"](),bounds)
        if method=="MF-DRO":
            r=mf.run()
        else:
            # h48 pattern: use mf_dro ONLY to draw the shared initial design,
            # then hand it to the standalone Takeno MF-MES and discard mf_dro.
            from src.baselines.mf_mes_takeno import run_mf_mes
            mf._sample_initial_points()
            res=run_mf_mes(f_hf=mf.f_hf,f_lf=mf.f_lf,bounds=bounds,c_H=mf.c_H,c_L=mf.c_L,
                cost_budget=BUDGET,true_opt=-float(cfg.true_opt),
                X_lf0=torch.stack(list(mf.data_lf_x)),Y_lf0=torch.tensor(list(mf.data_lf_y)),
                X_hf0=torch.stack(list(mf.data_hf_x)),Y_hf0=torch.tensor(list(mf.data_hf_y)),
                seed=seed,verbose=False,ko_kwargs=dict(dkl_threshold=9999))
            r={k:v for k,v in res.items() if k not in ("acq_info","X_hf","Y_hf")}
            r["hf_regret_curve"]=res["regret_curve"]
            r["fidelity_trace"]=res["fidelity_history"]
    else:
        from src.baselines.mf_baselines import (MultiFidelityBenchmark,
                                                MFMIGreedyOptimizer, MFGPUCBOptimizer)
        b=MultiFidelityBenchmark(bench)
        common=dict(n_initial_hf=sp["n_hf"],n_initial_lf=sp["n_lf"],seed=seed,
                    cost_budget=BUDGET)
        opt=(MFMIGreedyOptimizer(b,**common) if method=="MF-MI-Greedy"
             else MFGPUCBOptimizer(b,**common))
        r=opt.run(bo_iterations=4000)
        r.setdefault("hf_regret_curve",r.get("regret_curve"))
    c=r.get("hf_regret_curve") or []
    r["_meta"]=dict(bench=bench,method=method,seed=seed,budget=BUDGET,**sp)
    r["_wall_s"]=round(time.time()-t0,1)
    r["final_regret"]=float(c[-1]) if c else float("nan")
    r["n_improvements"]=sum(1 for i in range(1,len(c)) if c[i]<c[i-1]-1e-9)
    return r

if __name__=="__main__":
    bench,method,seed=sys.argv[1],sys.argv[2],int(sys.argv[3])
    out=os.path.join(os.path.dirname(__file__),"..","results",f"{bench}__{method}__seed{seed}.json")
    r=run(bench,method,seed)
    tmp=out+".tmp"; json.dump(r,open(tmp,"w"),default=float); os.replace(tmp,out)
    c=r["hf_regret_curve"]
    print(f"[done] {bench} {method} seed{seed} regret={r['final_regret']:.4f} "
          f"iters={len(c)} improv={r['n_improvements']} wall={r['_wall_s']/60:.1f}m",flush=True)
