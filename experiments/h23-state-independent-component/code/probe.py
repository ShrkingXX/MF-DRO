"""H23 (see ../protocol.md). Single process, thread-capped."""
import os, sys, json
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v]="1"
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import (DirectMFRegretOptimization,
                                build_candidate_features, _y_star_for_model)
SEED=44; FD=torch.float32; N_POOLS=12
hf=get_benchmark("Hartmann_6D_HF"); lf=get_benchmark("Hartmann_6D_LF")
bounds=torch.tensor([hf["domain_min"],hf["domain_max"]],dtype=torch.float64)
torch.manual_seed(SEED); np.random.seed(SEED)
cfg=_build_mf_dro_config("h23","Hartmann_6D","g",SEED,bo_iterations=3,num_epochs=10,
    minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=1e9,initial_hf=36,
    initial_lf=60,dkl_threshold=9999,bes_delta=0.0,rollout_length=8)
cfg.seed=SEED
mf=DirectMFRegretOptimization(cfg,hf["make_objective"](),lf["make_objective"](),bounds)
mf._sample_initial_points(); mf._update_ko_ensemble()
batch=mf._generate_rollout_batch(); mf._train_dt(batch)
rpm=mf.config.rollouts_per_model
idx=[k*rpm for k in range(len(batch)//rpm)]

def hid(st):
    H=mf.dt.hidden_size
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
    Hs=[hid(batch[i]["states"][0].double()) for i in idx]
    W=np.stack([mf.dt.coef_head(h).double().numpy() for h in Hs])
    B=np.array([float(mf.dt.bias_head(h)) for h in Hs])
    wbar=W.mean(0); D=W-wbar
    ysa=_y_star_for_model(mf.ko_ensemble[0],mf.y_star_pool,seed=SEED)
    agree=0; ratios=[]; bias_flips=0
    for p in range(N_POOLS):
        Xp=bounds[0]+(bounds[1]-bounds[0])*torch.rand(200,mf.d,dtype=torch.float64,
            generator=torch.Generator().manual_seed(300+p))
        cf=build_candidate_features(mf.ko_ensemble[0],Xp,bounds,mf.c_H,mf.c_L,
            torch.zeros(mf.d,dtype=torch.float64),y_star_arr=ysa).double().numpy()
        s_bar = cf @ wbar                       # state-INDEPENDENT scores
        a_bar = int(s_bar.argmax())
        # full model, per state
        fulls=[int((cf @ W[i] + B[i]).argmax()) for i in range(len(W))]
        if all(f==a_bar for f in fulls): agree+=1
        # margin vs what delta can contribute
        top2=np.sort(s_bar)[-2:]
        margin=float(top2[1]-top2[0])
        dcontrib=max(float((cf @ D[i]).max()-(cf @ D[i]).min()) for i in range(len(D)))
        ratios.append(margin/max(dcontrib,1e-12))
        # bias check: does adding B change the argmax? (it cannot -- constant)
        if int((cf @ W[0] + B[0]).argmax()) != int((cf @ W[0]).argmax()): bias_flips+=1
print(f"\nPRED 1  w-bar alone reproduces the full argmax on {agree}/{N_POOLS} pools "
      f"-> {'PASS' if agree>=11 else 'FAIL'}")
print(f"PRED 2  median margin / max |delta contribution| = {np.median(ratios):.2f} "
      f"-> {'PASS' if np.median(ratios)>=5 else 'FAIL'}")
print(f"        per-pool ratios: {[round(r,1) for r in ratios]}")
print(f"SECONDARY bias_head changed the argmax on {bias_flips}/{N_POOLS} pools "
      f"(must be 0 -- it adds a per-candidate constant) "
      f"-> {'CONFIRMED' if bias_flips==0 else 'UNEXPECTED'}")
print(f"\n        ||w-bar||={np.linalg.norm(wbar):.4f}  "
      f"mean||delta||={np.linalg.norm(D,axis=1).mean():.6f}  "
      f"ratio={np.linalg.norm(D,axis=1).mean()/np.linalg.norm(wbar):.5f}")
json.dump({"agree":agree,"median_ratio":float(np.median(ratios)),
           "ratios":[float(r) for r in ratios],"bias_flips":bias_flips,
           "norm_wbar":float(np.linalg.norm(wbar)),
           "mean_norm_delta":float(np.linalg.norm(D,axis=1).mean())},
          open(os.path.join(os.path.dirname(__file__),"..","results","h23.json"),"w"),indent=2)
