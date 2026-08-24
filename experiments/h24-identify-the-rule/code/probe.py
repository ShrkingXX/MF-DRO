"""H24 (see ../protocol.md). Single process, thread-capped."""
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
                                build_candidate_features, _y_star_for_model,
                                _gp_candidate_features)
SEED=44; FD=torch.float32; N_POOLS=12
hf=get_benchmark("Hartmann_6D_HF"); lf=get_benchmark("Hartmann_6D_LF")
bounds=torch.tensor([hf["domain_min"],hf["domain_max"]],dtype=torch.float64)
torch.manual_seed(SEED); np.random.seed(SEED)
cfg=_build_mf_dro_config("h24","Hartmann_6D","g",SEED,bo_iterations=3,num_epochs=10,
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

NAMES=[f"x[{i}]" for i in range(mf.d)]+["mu_H","sigma_H","mu_L","sigma_L","dist_inc"]
mf.dt.eval()
with torch.no_grad():
    W=np.stack([mf.dt.coef_head(hid(batch[i]["states"][0].double())).double().numpy() for i in idx])
wbar=W.mean(0)
print("SIGNED w-bar (the fixed acquisition rule):")
order=np.argsort(-np.abs(wbar))
for i in order:
    print(f"   {NAMES[i]:>9}  {wbar[i]:+.4f}")

ysa=_y_star_for_model(mf.ko_ensemble[0],mf.y_star_pool,seed=SEED)
rules={}
with torch.no_grad():
    for p in range(N_POOLS):
        Xp=bounds[0]+(bounds[1]-bounds[0])*torch.rand(200,mf.d,dtype=torch.float64,
            generator=torch.Generator().manual_seed(300+p))
        cf=build_candidate_features(mf.ko_ensemble[0],Xp,bounds,mf.c_H,mf.c_L,
            torch.zeros(mf.d,dtype=torch.float64),y_star_arr=ysa).double().numpy()
        learned=int((cf@wbar).argmax())
        d=mf.d
        muH,sgH,muL,sgL=cf[:,d],cf[:,d+1],cf[:,d+2],cf[:,d+3]
        cand={"mu_H alone":muH,"mu_L alone":muL,"mu_H + mu_L":muH+muL}
        for beta in (0.5,1,2,3,5):
            cand[f"mu_H + {beta}*sigma_H"]=muH+beta*sgH
        g=_gp_candidate_features(mf.ko_ensemble[0],Xp,mf.c_H,mf.c_L,ysa)
        try:
            cand["MF-MES teacher (HF/c_H)"]=np.asarray(g[4],dtype=float)
        except Exception:
            pass
        for k,v in cand.items():
            rules.setdefault(k,0)
            if int(np.asarray(v).argmax())==learned: rules[k]+=1
print(f"\nagreement of argmax(w-bar . cf) with standard rules ({N_POOLS} pools):")
best=None
for k,v in sorted(rules.items(),key=lambda kv:-kv[1]):
    print(f"   {k:<26} {v}/{N_POOLS} = {v/N_POOLS:5.1%}")
    if best is None: best=(k,v/N_POOLS)
print("\n"+"="*66)
print(f"PRED 1 (some rule >= 75%): best = {best[0]} at {best[1]:.1%} -> "
      f"{'PASS' if best[1]>=0.75 else 'FAIL'}")
if best[1] < 0.75:
    print("PRED 2 NULL: no standard rule matches; the learned rule is a linear")
    print("  combination without a simple name. The coefficient table IS the result.")
print("  (Chance agreement ~0.5%. Agreement is descriptive, not identity.)")
print("="*66)
json.dump({"wbar":{NAMES[i]:float(wbar[i]) for i in range(len(wbar))},
           "agreement":{k:v/N_POOLS for k,v in rules.items()}},
          open(os.path.join(os.path.dirname(__file__),"..","results","h24.json"),"w"),indent=2)
