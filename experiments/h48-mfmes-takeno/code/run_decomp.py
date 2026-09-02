"""Which knob costs the 0.134 -- rho_init, initial_lengthscale, or both?

The H48 'defaults vs matched' comparison changed BOTH at once, so it cannot
attribute the effect. Four arms, same seeds, same everything else:
  def   : rho_init=0.8            , initial_lengthscale=None     (KO-GP defaults)
  rho   : rho_init=member0's      , initial_lengthscale=None
  ls    : rho_init=0.8            , initial_lengthscale=member0's (0.1839)
  both  : rho_init=member0's      , initial_lengthscale=member0's (= h48 'matched')
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore", message=".*power of 2.*")
from concurrent.futures import ProcessPoolExecutor, as_completed
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
SEEDS = [42,43,44,45,46,47,48,49,50,51]
ARMS = ["def","rho","ls","both"]
NUM_WORKERS = 6

def _worker(task):
    arm, seed = task
    for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[v]="1"
    sys.path.insert(0, REPO)
    import torch
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
    out = os.path.join(RESULTS, f"decomp__{arm}__seed{seed}.json")
    if os.path.exists(out): return (arm,seed,"SKIPPED",None)
    try:
        import numpy as np
        from benchmarks import get_benchmark
        from dro_runner import _build_mf_dro_config
        from src.policy.mf_dro import DirectMFRegretOptimization
        from src.baselines.mf_mes_takeno import run_mf_mes
        hf=get_benchmark("Hartmann_6D_HF"); lf=get_benchmark("Hartmann_6D_LF")
        bounds=torch.tensor([hf["domain_min"],hf["domain_max"]],dtype=torch.float64)
        torch.manual_seed(seed); np.random.seed(seed)
        cfg=_build_mf_dro_config("h48_decomp","Hartmann_6D",arm,seed,bo_iterations=1,
            num_epochs=1,minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=200.0,
            initial_hf=36,initial_lf=60,dkl_threshold=9999,bes_delta=0.0,rollout_length=8)
        cfg.seed=seed
        mf=DirectMFRegretOptimization(cfg,hf["make_objective"](),lf["make_objective"](),bounds)
        mf._sample_initial_points()
        m0=mf.ko_ensemble[0]
        m0_rho=float(m0.rho_init)
        m0_ls=(None if getattr(m0,"initial_lengthscale",None) is None else float(m0.initial_lengthscale))
        kk=dict(dkl_threshold=9999)
        if arm in ("rho","both"): kk["rho_init"]=m0_rho
        if arm in ("ls","both"):  kk["initial_lengthscale"]=m0_ls
        X_lf=torch.stack(list(mf.data_lf_x)); Y_lf=torch.tensor(list(mf.data_lf_y))
        X_hf=torch.stack(list(mf.data_hf_x)); Y_hf=torch.tensor(list(mf.data_hf_y))
        res=run_mf_mes(f_hf=mf.f_hf,f_lf=mf.f_lf,bounds=bounds,c_H=mf.c_H,c_L=mf.c_L,
            cost_budget=200.0,true_opt=-float(cfg.true_opt),X_lf0=X_lf,Y_lf0=Y_lf,
            X_hf0=X_hf,Y_hf0=Y_hf,single_fidelity=False,seed=seed,verbose=False,ko_kwargs=kk)
        rec={k:v for k,v in res.items() if k not in ("acq_info","X_hf","Y_hf")}
        rec.update(arm=arm,seed=seed,ko_kwargs={k:v for k,v in kk.items()},
                   m0_rho=m0_rho,m0_ls=m0_ls)
        tmp=out+".tmp"; json.dump(rec,open(tmp,"w")); os.replace(tmp,out)
        return (arm,seed,"OK",rec["final_regret"])
    except Exception as e:
        import traceback
        open(os.path.join(RESULTS,f"DFAILED__{arm}__{seed}.txt"),"w").write(traceback.format_exc())
        return (arm,seed,f"FAILED: {e}",None)

if __name__=="__main__":
    todo=[(a,s) for a in ARMS for s in SEEDS
          if not os.path.exists(os.path.join(RESULTS,f"decomp__{a}__seed{s}.json"))]
    print(f"decomposition: {len(todo)} jobs, {NUM_WORKERS} workers",flush=True)
    t0=time.time()
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futs={ex.submit(_worker,t):t for t in todo}
        for i,fu in enumerate(as_completed(futs),1):
            a,s,st,v=fu.result()
            if i%8==0 or st!="OK":
                print(f"[{i}/{len(todo)}] {a} seed{s}: {st} {v if v is None else f'{v:.4f}'} "
                      f"({(time.time()-t0)/60:.1f} min)",flush=True)
    print(f"DONE in {(time.time()-t0)/60:.1f} min",flush=True)
