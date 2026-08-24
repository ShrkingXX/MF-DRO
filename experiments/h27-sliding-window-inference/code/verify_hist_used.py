import os,sys
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
sys.path.insert(0,"/Users/yurucui/Desktop/DRO-Code/DRO-aistats-submission")
import torch,numpy as np
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization, build_candidate_features, _y_star_for_model
hf=get_benchmark("Hartmann_6D_HF"); lf=get_benchmark("Hartmann_6D_LF")
bounds=torch.tensor([hf["domain_min"],hf["domain_max"]],dtype=torch.float64)
torch.manual_seed(44); np.random.seed(44)
cfg=_build_mf_dro_config("v","Hartmann_6D","v",44,bo_iterations=3,num_epochs=10,
    minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=1e9,initial_hf=36,
    initial_lf=60,dkl_threshold=9999,bes_delta=0.0,rollout_length=8)
cfg.seed=44; cfg.rollout_reward="mes_entropy"
mf=DirectMFRegretOptimization(cfg,hf["make_objective"](),lf["make_objective"](),bounds)
mf._sample_initial_points(); mf._update_ko_ensemble()
b=mf._generate_rollout_batch(); mf._train_dt(b)
rpm=mf.config.rollouts_per_model
states=[b[k*rpm]["states"][0].double().float() for k in range(len(b)//rpm)]
dt=mf.dt; dt.eval()
FD=torch.float32; H=dt.hidden_size
def h_of(state, hist):
    with torch.no_grad():
        if hist:
            st=[x['state'] for x in hist]+[state]; T=len(st)
            s=torch.stack(st).unsqueeze(0)
            r=torch.tensor([x['rtg'] for x in hist]+[0.7],dtype=FD).view(1,T,1)
            bb=torch.tensor([x['btg'] for x in hist]+[22.0],dtype=FD).view(1,T,1)
            ts=torch.arange(T,dtype=torch.long).unsqueeze(0)
        else:
            T=1; s=state.unsqueeze(0).unsqueeze(0)
            r=torch.tensor([[[0.7]]],dtype=FD); bb=torch.tensor([[[22.0]]],dtype=FD)
            ts=torch.tensor([[0]],dtype=torch.long)
        ax=torch.zeros(1,T,dt.action_dim,dtype=FD); ae=torch.zeros(1,T,dtype=FD)
        e=[dt.reward_ln(dt.reward_embedding(r)),dt.btg_ln(dt.btg_embed(bb)),
           dt.state_ln(dt.state_embedding(s)),
           dt.action_ln(dt.action_embed_mf(torch.cat([ax,ae.unsqueeze(-1)],dim=-1)))]
        pos=dt.position_embedding(ts).repeat_interleave(4,dim=1)
        seq=torch.stack(e,dim=2).reshape(1,4*T,H)+pos
        cm=torch.triu(torch.ones(4*T,4*T,dtype=torch.bool),diagonal=1)
        return dt.transformer(seq,mask=cm)[0,2::4,:][-1]
cur=states[0]
hist=[{'state':states[i],'rtg':0.5+0.05*i,'btg':22.0+3*i} for i in range(1,8)]
h1=h_of(cur,None); h8=h_of(cur,hist)
print(f"||h(T=1)||        = {h1.norm():.6f}")
print(f"||h(T=8)||        = {h8.norm():.6f}")
print(f"||h(T=8)-h(T=1)|| = {(h8-h1).norm():.8f}")
print(f"cosine            = {torch.nn.functional.cosine_similarity(h1.unsqueeze(0),h8.unsqueeze(0)).item():.8f}")
w1=dt.coef_head(h1).detach().numpy(); w8=dt.coef_head(h8).detach().numpy()
print(f"||w(T=8)-w(T=1)|| = {np.linalg.norm(w8-w1):.8f}   (||w||={np.linalg.norm(w1):.4f})")
print(f"relative change in w = {np.linalg.norm(w8-w1)/np.linalg.norm(w1):.6%}")
print()
print("VERDICT:", "hist IS used -- h genuinely changes; the argmax simply does not"
      if (h8-h1).norm()>1e-9 else "*** WIRING BUG: hist ignored, h identical ***")
