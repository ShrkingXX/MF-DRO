"""Arm C: MF-MES teacher, no DT deciding, at H40's exact settings."""
import os, sys, json, time
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v]="1"
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization, compute_joint_mf_mes

def run(seed):
    hf=get_benchmark("Hartmann_6D_HF"); lf=get_benchmark("Hartmann_6D_LF")
    bounds=torch.tensor([hf["domain_min"],hf["domain_max"]],dtype=torch.float64)
    torch.manual_seed(seed); np.random.seed(seed)
    cfg=_build_mf_dro_config("h44","Hartmann_6D","teacher",seed,bo_iterations=2000,
        num_epochs=10,minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=200.0,
        initial_hf=6,initial_lf=45,dkl_threshold=9999,bes_delta=0.0,rollout_length=8)
    cfg.seed=seed; cfg.rollout_reward="mes_entropy"; cfg.use_candidate_scoring=True
    mf=DirectMFRegretOptimization(cfg,hf["make_objective"](),lf["make_objective"](),bounds)
    # DT still TRAINED (so the RNG stream and candidate pools match H40's arms
    # exactly); only the decision is replaced by the teacher's argmax.
    def teacher(state,rtg,btg,timestep=0,use_candidate_scoring=False,
                candidate_features=None,fidelity_sampling=True,hist=None):
        cf=candidate_features.double(); lo,hi=mf.bounds[0],mf.bounds[1]
        X=lo+(hi-lo)*cf[:,:mf.d]
        x_raw,ell,_=compute_joint_mf_mes(mf.ko_ensemble[0],X,mf.c_H,mf.c_L)
        mf.dt.last_p_pred=float(ell)
        return ((x_raw-lo)/(hi-lo)).clamp(0,1).float(), int(ell)
    mf.dt.propose_mf=teacher
    t0=time.time(); r=mf.run()
    reg=np.array(r["hf_regret_curve"]); fid=np.array(r["fidelity_trace"])
    n_impr=int(sum(1 for i in range(1,len(reg)) if reg[i]<reg[i-1]-1e-12))
    return dict(arm="teacher", seed=seed, n_improvements=n_impr,
                frozen=bool(n_impr==0), final_regret=float(reg[-1]),
                n_iters=int(len(reg)), n_hf=int((fid==1).sum()),
                wall=round(time.time()-t0,1))

if __name__=="__main__":
    s=int(sys.argv[1]); o=run(s)
    d=os.path.join(os.path.dirname(__file__),"..","results")
    with open(os.path.join(d,f"teacher__seed{s}.json"),"w") as f: json.dump(o,f)
    print(json.dumps(o))
