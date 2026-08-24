"""H22 (see ../protocol.md). Single process, thread-capped."""
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
LAMBDAS=[1,2,3,5,8,12,20,35,60,100]

hf=get_benchmark("Hartmann_6D_HF"); lf=get_benchmark("Hartmann_6D_LF")
bounds=torch.tensor([hf["domain_min"],hf["domain_max"]],dtype=torch.float64)
torch.manual_seed(SEED); np.random.seed(SEED)
cfg=_build_mf_dro_config("h22","Hartmann_6D","g",SEED,bo_iterations=3,num_epochs=10,
    minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=1e9,initial_hf=36,
    initial_lf=60,dkl_threshold=9999,bes_delta=0.0,rollout_length=8)
cfg.seed=SEED
mf=DirectMFRegretOptimization(cfg,hf["make_objective"](),lf["make_objective"](),bounds)
mf._sample_initial_points(); mf._update_ko_ensemble()
batch=mf._generate_rollout_batch(); mf._train_dt(batch)
rpm=mf.config.rollouts_per_model
idx=[k*rpm for k in range(len(batch)//rpm)]
S=torch.stack([batch[i]["states"][0].double() for i in idx])
Sbar=S.mean(0, keepdim=True)

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

ysa=_y_star_for_model(mf.ko_ensemble[0],mf.y_star_pool,seed=SEED)
pools=[]
for p in range(N_POOLS):
    Xp=bounds[0]+(bounds[1]-bounds[0])*torch.rand(200,mf.d,dtype=torch.float64,
        generator=torch.Generator().manual_seed(300+p))
    pools.append(build_candidate_features(mf.ko_ensemble[0],Xp,bounds,mf.c_H,mf.c_L,
        torch.zeros(mf.d,dtype=torch.float64),y_star_arr=ysa))

mf.dt.eval(); res=[]
print(f"{'lambda':>7} {'argmax moved':>14} {'min cos(w)':>13} {'rel spread w':>13}")
with torch.no_grad():
    for lam in LAMBDAS:
        Sl = Sbar + lam*(S - Sbar)
        Hs=[hid(Sl[i]) for i in range(Sl.shape[0])]
        W=np.stack([mf.dt.coef_head(h).double().numpy() for h in Hs])
        n=np.linalg.norm(W,axis=1); Wn=W/n[:,None]; cos=Wn@Wn.T
        off=cos[~np.eye(len(W),dtype=bool)]
        d=[np.linalg.norm(W[i]-W[j]) for i in range(len(W)) for j in range(i+1,len(W))]
        relw=float(np.mean(d)/np.mean(n))
        moved=0
        for p,cf in enumerate(pools):
            def sc(h):
                return ((cf.to(FD)*mf.dt.coef_head(h).unsqueeze(0)).sum(-1)+mf.dt.bias_head(h))
            a0=int(sc(Hs[0]).argmax())
            if int(sc(Hs[1+(p%(len(Hs)-1))]).argmax())!=a0: moved+=1
        res.append({"lambda":lam,"moved":moved,"frac":moved/N_POOLS,
                    "min_cos":float(off.min()),"rel_w":relw})
        print(f"{lam:>7} {moved:>7}/{N_POOLS} = {moved/N_POOLS:>4.0%} "
              f"{off.min():>13.8f} {relw:>13.6f}")

star=next((r["lambda"] for r in res if r["frac"]>0.30), None)
print("\n"+"="*66)
if star is None:
    print("PRED 3 NULL: no lambda <= 100 moves the argmax on >30% of pools.")
    print("  The learned map is invariant to state DIRECTION over two orders of")
    print("  magnitude of gain -- stronger than 'far from threshold'.")
else:
    print(f"lambda* = {star}  (smallest gain with >30% movement)")
    print(f"PRED 1 (lambda* >= 10): {'PASS' if star>=10 else 'FAIL'}")
    if star < 5:
        print("PRED 2: lambda* < 5 -- closer to threshold than H21 suggested;")
        print("  the paper's 'far short' phrasing must be softened.")
print("\nCAVEAT (pre-registered): large lambda puts states OUT OF DISTRIBUTION.")
print("This measures the learned map's gain, NOT a usable intervention.")
print("="*66)
json.dump({"lambda_star":star,"sweep":res},
          open(os.path.join(os.path.dirname(__file__),"..","results","h22.json"),"w"),indent=2)
