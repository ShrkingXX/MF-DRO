"""H29 worker: one (cost_ratio, K) cell -> chosen-sigma_H / mu_H percentiles."""
import os, sys, json
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v]="1"
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
import src.policy.mf_dro as M

def run(ratio, K):
    hf=get_benchmark("Hartmann_6D_HF"); lf=get_benchmark("Hartmann_6D_LF")
    bounds=torch.tensor([hf["domain_min"],hf["domain_max"]],dtype=torch.float64)
    torch.manual_seed(44); np.random.seed(44)
    cfg=_build_mf_dro_config("h29","Hartmann_6D","g",44,bo_iterations=1,num_epochs=1,
        minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=1e9,initial_hf=36,
        initial_lf=60,dkl_threshold=9999,bes_delta=0.0,rollout_length=8)
    cfg.seed=44; cfg.rollout_reward="mes_entropy"
    cfg.c_H=float(ratio); cfg.c_L=1.0
    # force the teacher's y* sample count
    _orig=M.compute_joint_mf_mes
    M.compute_joint_mf_mes=lambda ko,roi,cH,cL,K=K,_o=_orig: _o(ko,roi,cH,cL,K=K)
    try:
        mf=M.DirectMFRegretOptimization(cfg,hf["make_objective"](),lf["make_objective"](),bounds)
        mf.c_H=float(ratio); mf.c_L=1.0
        mf._sample_initial_points(); mf._update_ko_ensemble()
        batch=mf._generate_rollout_batch()
    finally:
        M.compute_joint_mf_mes=_orig
    d=mf.d; ps,pm=[],[]
    for t in batch:
        if "candidates" not in t: continue
        cnd=t["candidates"].double().numpy(); idx=t["chosen_idx"].long().numpy().reshape(-1)
        for j in range(min(len(idx),cnd.shape[0])):
            pool=cnd[j]; k=int(idx[j])
            if k<0 or k>=pool.shape[0]: continue
            sg=pool[:,d+1]; mu=pool[:,d]
            if sg.std()<1e-12 or mu.std()<1e-12: continue
            ps.append(float((sg<sg[k]).mean())); pm.append(float((mu<mu[k]).mean()))
    return {"ratio":ratio,"K":K,"n":len(ps),
            "pct_sigma":float(np.mean(ps)) if ps else None,
            "pct_mu":float(np.mean(pm)) if pm else None}

if __name__=="__main__":
    print(json.dumps(run(float(sys.argv[1]), int(sys.argv[2]))))
