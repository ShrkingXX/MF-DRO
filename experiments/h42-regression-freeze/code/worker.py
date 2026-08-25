import os, sys, json, time
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v]="1"
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization

def run(seed):
    hf=get_benchmark("Hartmann_6D_HF"); lf=get_benchmark("Hartmann_6D_LF")
    bounds=torch.tensor([hf["domain_min"],hf["domain_max"]],dtype=torch.float64)
    torch.manual_seed(seed); np.random.seed(seed)
    cfg=_build_mf_dro_config("h42","Hartmann_6D","reg",seed,
        bo_iterations=50, num_epochs=10, minimum_hf_fraction=0.25, real_hf_warmup=2,
        cost_budget=1e12,                      # iteration-capped, not cost-capped
        initial_hf=6, initial_lf=45, dkl_threshold=9999, bes_delta=0.0, rollout_length=8)
    cfg.seed=seed; cfg.rollout_reward="mes_entropy"
    cfg.use_candidate_scoring=False            # THE VARIABLE: regression head
    mf=DirectMFRegretOptimization(cfg,hf["make_objective"](),lf["make_objective"](),bounds)
    t0=time.time(); r=mf.run()
    reg=np.array(r["hf_regret_curve"],dtype=float)
    ir=np.array(r.get("inference_regret_curve") or [np.nan]*len(reg),dtype=float)
    xs=np.array(r["x_t_trace"]); fid=np.array(r["fidelity_trace"])
    impr=int(sum(1 for i in range(1,len(reg)) if reg[i]<reg[i-1]-1e-12))
    d=[np.linalg.norm(xs[i]-xs[j]) for i in range(len(xs)) for j in range(i+1,len(xs))]
    return dict(seed=seed, use_candidate_scoring=False,
                n_iters=int(len(reg)), n_improvements=impr, frozen=bool(impr==0),
                distinct=int(len({tuple(np.round(x,6)) for x in xs})),
                x_spread=float(np.mean(d)) if d else 0.0,
                final_regret=float(reg[-1]), first_regret=float(reg[0]),
                final_inference_regret=float(ir[-1]) if np.isfinite(ir[-1]) else None,
                n_hf=int((fid==1).sum()), n_lf=int((fid==0).sum()),
                total_cost=float(r["cost_curve"][-1]), wall=round(time.time()-t0,1))

if __name__=="__main__":
    s=int(sys.argv[1]); out=run(s)
    d=os.path.join(os.path.dirname(__file__),"..","results")
    json.dump(out, open(os.path.join(d,f"reg__seed{s}.json"),"w"))
    print(json.dumps(out))
