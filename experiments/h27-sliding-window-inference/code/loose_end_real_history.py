"""H27 loose end: with REAL history (not synthetic), how much does w move?
The synthetic probe (mixed ensemble blocks + fabricated rtg/btg) moved w by
19.9% yet real history changed no proposal. Measure the real case directly."""
import os,sys
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
sys.path.insert(0,"/Users/yurucui/Desktop/DRO-Code/DRO-aistats-submission")
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import (DirectMFRegretOptimization, build_candidate_features,
                                _y_star_for_model)
FD=torch.float32
hf=get_benchmark("Hartmann_6D_HF"); lf=get_benchmark("Hartmann_6D_LF")
bounds=torch.tensor([hf["domain_min"],hf["domain_max"]],dtype=torch.float64)
torch.manual_seed(44); np.random.seed(44)
cfg=_build_mf_dro_config("le","Hartmann_6D","le",44,bo_iterations=10,num_epochs=3,
    minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=1e9,initial_hf=36,
    initial_lf=60,dkl_threshold=9999,bes_delta=0.0,rollout_length=8)
cfg.seed=44; cfg.rollout_reward="mes_entropy"; cfg.inference_context_k=1
mf=DirectMFRegretOptimization(cfg,hf["make_objective"](),lf["make_objective"](),bounds)
mf.run()
H=mf._real_hist
print(f"real history captured: {len(H)} iterations")
S=torch.stack([h['state'].double() for h in H])
d=[np.linalg.norm((S[i]-S[j]).numpy()) for i in range(len(S)) for j in range(i+1,len(S))]
print(f"  mean pairwise L2 among REAL consecutive states: {np.mean(d):.4f}")
print(f"  rtg range {min(h['rtg'] for h in H):.4f}..{max(h['rtg'] for h in H):.4f}   "
      f"btg range {min(h['btg'] for h in H):.2f}..{max(h['btg'] for h in H):.2f}")
dt=mf.dt; dt.eval(); HID=dt.hidden_size
def h_of(cur, hist):
    with torch.no_grad():
        if hist:
            st=[x['state'].float() for x in hist]+[cur]; T=len(st)
            s=torch.stack(st).unsqueeze(0)
            r=torch.tensor([x['rtg'] for x in hist]+[float(H[-1]['rtg'])],dtype=FD).view(1,T,1)
            b=torch.tensor([x['btg'] for x in hist]+[float(H[-1]['btg'])],dtype=FD).view(1,T,1)
            ts=torch.arange(T,dtype=torch.long).unsqueeze(0)
        else:
            T=1; s=cur.unsqueeze(0).unsqueeze(0)
            r=torch.tensor([[[float(H[-1]['rtg'])]]],dtype=FD)
            b=torch.tensor([[[float(H[-1]['btg'])]]],dtype=FD)
            ts=torch.tensor([[0]],dtype=torch.long)
        ax=torch.zeros(1,T,dt.action_dim,dtype=FD); ae=torch.zeros(1,T,dtype=FD)
        e=[dt.reward_ln(dt.reward_embedding(r)),dt.btg_ln(dt.btg_embed(b)),
           dt.state_ln(dt.state_embedding(s)),
           dt.action_ln(dt.action_embed_mf(torch.cat([ax,ae.unsqueeze(-1)],dim=-1)))]
        pos=dt.position_embedding(ts).repeat_interleave(4,dim=1)
        seq=torch.stack(e,dim=2).reshape(1,4*T,HID)+pos
        cm=torch.triu(torch.ones(4*T,4*T,dtype=torch.bool),diagonal=1)
        return dt.transformer(seq,mask=cm)[0,2::4,:][-1]
cur=H[-1]['state'].float(); hist=[{'state':h['state'],'rtg':h['rtg'],'btg':h['btg']} for h in H[-8:-1]]
h1=h_of(cur,None); h8=h_of(cur,hist)
with torch.no_grad():
    w1=dt.coef_head(h1).double().numpy(); w8=dt.coef_head(h8).double().numpy()
print(f"\nREAL history, T=1 vs T={len(hist)+1}:")
print(f"  ||h8-h1|| = {(h8-h1).norm():.6f}   cosine = "
      f"{torch.nn.functional.cosine_similarity(h1.unsqueeze(0),h8.unsqueeze(0)).item():.6f}")
print(f"  ||w8-w1||/||w1|| = {np.linalg.norm(w8-w1)/np.linalg.norm(w1):.4%}   "
      f"(synthetic probe gave 19.88%)")
ysa=_y_star_for_model(mf.ko_ensemble[0],mf.y_star_pool,seed=44); moved=0
with torch.no_grad():
    for p in range(12):
        X=bounds[0]+(bounds[1]-bounds[0])*torch.rand(200,mf.d,dtype=torch.float64,
            generator=torch.Generator().manual_seed(300+p))
        cf=build_candidate_features(mf.ko_ensemble[0],X,bounds,mf.c_H,mf.c_L,
            torch.zeros(mf.d,dtype=torch.float64),y_star_arr=ysa)
        f=lambda w,h:((cf.to(FD)*torch.tensor(w,dtype=FD).unsqueeze(0)).sum(-1)+dt.bias_head(h)).argmax()
        if int(f(w1,h1))!=int(f(w8,h8)): moved+=1
print(f"  argmax moved: {moved}/12")
