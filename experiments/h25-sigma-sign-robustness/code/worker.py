"""H25 worker: one seed -> mean scoring coefficients of an independently trained model."""
import os, sys, json
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v]="1"
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization
FD=torch.float32

def run(seed):
    hf=get_benchmark("Hartmann_6D_HF"); lf=get_benchmark("Hartmann_6D_LF")
    bounds=torch.tensor([hf["domain_min"],hf["domain_max"]],dtype=torch.float64)
    torch.manual_seed(seed); np.random.seed(seed)
    cfg=_build_mf_dro_config("h25","Hartmann_6D","g",seed,bo_iterations=3,num_epochs=10,
        minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=1e9,initial_hf=36,
        initial_lf=60,dkl_threshold=9999,bes_delta=0.0,rollout_length=8)
    cfg.seed=seed
    mf=DirectMFRegretOptimization(cfg,hf["make_objective"](),lf["make_objective"](),bounds)
    mf._sample_initial_points(); mf._update_ko_ensemble()
    b=mf._generate_rollout_batch(); mf._train_dt(b)
    rpm=mf.config.rollouts_per_model
    idx=[k*rpm for k in range(len(b)//rpm)]
    H=mf.dt.hidden_size
    def hid(st):
        s=st.unsqueeze(0).unsqueeze(0).to(FD)
        r=torch.tensor([[[0.7]]],dtype=FD); b_=torch.tensor([[[22.0]]],dtype=FD)
        ax=torch.zeros(1,1,mf.d,dtype=FD); ae=torch.zeros(1,1,dtype=FD)
        e=[mf.dt.reward_ln(mf.dt.reward_embedding(r)),mf.dt.btg_ln(mf.dt.btg_embed(b_)),
           mf.dt.state_ln(mf.dt.state_embedding(s)),
           mf.dt.action_ln(mf.dt.action_embed_mf(torch.cat([ax,ae.unsqueeze(-1)],dim=-1)))]
        pos=mf.dt.position_embedding(torch.tensor([[0]],dtype=torch.long)).repeat_interleave(4,dim=1)
        seq=torch.stack(e,dim=2).reshape(1,4,H)+pos
        cm=torch.triu(torch.ones(4,4,dtype=torch.bool),diagonal=1)
        return mf.dt.transformer(seq,mask=cm)[0,2::4,:][0]
    mf.dt.eval()
    with torch.no_grad():
        W=np.stack([mf.dt.coef_head(hid(b[i]["states"][0].double())).double().numpy() for i in idx])
    w=W.mean(0); d=mf.d
    return {"seed":seed,"mu_H":float(w[d]),"sigma_H":float(w[d+1]),
            "mu_L":float(w[d+2]),"sigma_L":float(w[d+3])}

if __name__=="__main__":
    print(json.dumps(run(int(sys.argv[1]))))
