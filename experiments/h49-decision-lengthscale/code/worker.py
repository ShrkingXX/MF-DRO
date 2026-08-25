"""Arm B: MF-DRO with the REGRESSION head -- x = action_head(h) directly.
No candidate pool, no argmax. Settings identical to h17/h31 so the three-way
comparison is valid."""
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
    cfg=_build_mf_dro_config("h49","Hartmann_6D","nat",seed,bo_iterations=2000,
        num_epochs=10,minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=200.0,
        initial_hf=36,initial_lf=60,dkl_threshold=9999,bes_delta=0.0,rollout_length=8)
    cfg.seed=seed; cfg.rollout_reward="mes_entropy"
    cfg.use_candidate_scoring=False
    cfg.natural_decision_lengthscale=True    # <-- the ONE variable vs h45
    mf=DirectMFRegretOptimization(cfg,hf["make_objective"](),lf["make_objective"](),bounds)
    t0=time.time(); r=mf.run()
    reg=np.array(r["hf_regret_curve"]); fid=np.array(r["fidelity_trace"])
    xs=np.array(r["x_t_trace"])
    n_impr=int(sum(1 for i in range(1,len(reg)) if reg[i]<reg[i-1]-1e-12))
    return dict(arm="regr_natural_ls", seed=seed, final_regret=float(reg[-1]),
                n_improvements=n_impr, frozen=bool(n_impr==0),
                n_iters=int(len(reg)), n_hf=int((fid==1).sum()),
                distinct=int(len({tuple(np.round(x,6)) for x in xs})),
                x_spread=float(np.mean([np.linalg.norm(xs[i]-xs[j])
                    for i in range(min(len(xs),40)) for j in range(i+1,min(len(xs),40))])),
                hf_regret_curve=[float(v) for v in reg],
                cost_curve=[float(v) for v in r["cost_curve"]],
                wall=round(time.time()-t0,1))

if __name__=="__main__":
    s=int(sys.argv[1]); o=run(s)
    d=os.path.join(os.path.dirname(__file__),"..","results")
    with open(os.path.join(d,f"nat__seed{s}.json"),"w") as f: json.dump(o,f)
    print(f"[done] seed{s} regret={o['final_regret']:.4f} impr={o['n_improvements']} "
          f"iters={o['n_iters']} wall={o['wall']}s", flush=True)
