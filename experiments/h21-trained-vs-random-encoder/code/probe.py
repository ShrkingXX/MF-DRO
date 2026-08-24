"""H21 (see ../protocol.md). Single process, thread-capped."""
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

def build(train):
    hf=get_benchmark("Hartmann_6D_HF"); lf=get_benchmark("Hartmann_6D_LF")
    bounds=torch.tensor([hf["domain_min"],hf["domain_max"]],dtype=torch.float64)
    torch.manual_seed(SEED); np.random.seed(SEED)
    cfg=_build_mf_dro_config("h21","Hartmann_6D",str(train),SEED,bo_iterations=3,
        num_epochs=10,minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=1e9,
        initial_hf=36,initial_lf=60,dkl_threshold=9999,bes_delta=0.0,rollout_length=8)
    cfg.seed=SEED
    mf=DirectMFRegretOptimization(cfg,hf["make_objective"](),lf["make_objective"](),bounds)
    mf._sample_initial_points(); mf._update_ko_ensemble()
    b=mf._generate_rollout_batch()
    if train: mf._train_dt(b)     # THE ONLY DIFFERENCE
    return mf,b,bounds

def hid(mf,st,rtg=0.7,btg=22.0):
    H=mf.dt.hidden_size
    s=st.unsqueeze(0).unsqueeze(0).to(FD)
    r=torch.tensor([[[rtg]]],dtype=FD); b_=torch.tensor([[[btg]]],dtype=FD)
    ax=torch.zeros(1,1,mf.d,dtype=FD); ae=torch.zeros(1,1,dtype=FD)
    e=[mf.dt.reward_ln(mf.dt.reward_embedding(r)),mf.dt.btg_ln(mf.dt.btg_embed(b_)),
       mf.dt.state_ln(mf.dt.state_embedding(s)),
       mf.dt.action_ln(mf.dt.action_embed_mf(torch.cat([ax,ae.unsqueeze(-1)],dim=-1)))]
    pos=mf.dt.position_embedding(torch.tensor([[0]],dtype=torch.long)).repeat_interleave(4,dim=1)
    seq=torch.stack(e,dim=2).reshape(1,4,H)+pos
    cm=torch.triu(torch.ones(4,4,dtype=torch.bool),diagonal=1)
    return mf.dt.transformer(seq,mask=cm)[0,2::4,:][0]

def relspread(M):
    n=np.linalg.norm(M,axis=1)
    d=[np.linalg.norm(M[i]-M[j]) for i in range(len(M)) for j in range(i+1,len(M))]
    return float(np.mean(d)/np.mean(n))

res={}
for train,name in [(True,"T trained"),(False,"R random init")]:
    mf,batch,bounds=build(train); rpm=mf.config.rollouts_per_model
    idx=[k*rpm for k in range(len(batch)//rpm)]
    S=np.stack([batch[i]["states"][0].double().numpy() for i in idx])
    mf.dt.eval()
    with torch.no_grad():
        Hs=np.stack([hid(mf,batch[i]["states"][0].double()).double().numpy() for i in idx])
        W=np.stack([mf.dt.coef_head(torch.tensor(h_,dtype=FD)).double().numpy() for h_ in Hs])
        ysa=_y_star_for_model(mf.ko_ensemble[0],mf.y_star_pool,seed=SEED)
        moved=0
        for p in range(N_POOLS):
            Xp=bounds[0]+(bounds[1]-bounds[0])*torch.rand(200,mf.d,dtype=torch.float64,
                generator=torch.Generator().manual_seed(300+p))
            cf=build_candidate_features(mf.ko_ensemble[0],Xp,bounds,mf.c_H,mf.c_L,
                torch.zeros(mf.d,dtype=torch.float64),y_star_arr=ysa)
            def sc(h):
                h=torch.tensor(h,dtype=FD)
                return ((cf.to(FD)*mf.dt.coef_head(h).unsqueeze(0)).sum(-1)+mf.dt.bias_head(h))
            a0=int(sc(Hs[0]).argmax())
            if int(sc(Hs[1+(p%(len(Hs)-1))]).argmax())!=a0: moved+=1
    ss,sh,sw=relspread(S),relspread(Hs),relspread(W)
    n=np.linalg.norm(W,axis=1); Wn=W/n[:,None]; cos=Wn@Wn.T
    off=cos[~np.eye(len(W),dtype=bool)]
    res[name]={"s":ss,"h":sh,"w":sw,"s_to_h":sh/ss,"h_to_w":sw/sh,
               "moved":moved,"frac":moved/N_POOLS,"min_cos":float(off.min())}
    print(f"[{name}]")
    print(f"  relative spread  s={ss:.6f}  h={sh:.6f}  w={sw:.6f}")
    print(f"  attenuation      s->h {sh/ss:.4f}x   h->w {sw/sh:.4f}x")
    print(f"  argmax moved on state swap: {moved}/{N_POOLS} = {moved/N_POOLS:.1%}")
    print(f"  min pairwise cosine(w)    : {off.min():.8f}\n")

T,R=res["T trained"],res["R random init"]
print("="*70)
p1 = R["s_to_h"] >= 1.5*T["s_to_h"]
print(f"PRED 1 (random s->h >= 1.5x trained's {T['s_to_h']:.4f} = {1.5*T['s_to_h']:.4f}): "
      f"random {R['s_to_h']:.4f} -> {'PASS -- TRAINING destroys the signal' if p1 else 'FAIL'}")
p2 = R["frac"] > 0.30
print(f"PRED 2 (random moves argmax >30%): {R['frac']:.1%} -> {'PASS' if p2 else 'FAIL'}")
if not (p1 or p2):
    print("\nPRED 3 NULL: the contraction is ARCHITECTURAL, not learned. The")
    print("  paper's stronger phrasing is defensible and the Limitations")
    print("  sentence about an untested encoder-side remedy should be revised.")
print("  (Arm R is a channel-capacity probe only -- a random net scoring")
print("   differently is NOT evidence that random is better.)")
print("="*70)
json.dump(res,open(os.path.join(os.path.dirname(__file__),"..","results","h21.json"),"w"),indent=2)
