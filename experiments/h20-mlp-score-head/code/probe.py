"""H20 (see ../protocol.md). Single process, thread-capped."""
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
IN_BAND=[0.50,0.60,0.70,0.80,0.90,1.00]; N_POOLS=12; SEED=44; FD=torch.float32

def build(linear):
    hf=get_benchmark("Hartmann_6D_HF"); lf=get_benchmark("Hartmann_6D_LF")
    bounds=torch.tensor([hf["domain_min"],hf["domain_max"]],dtype=torch.float64)
    torch.manual_seed(SEED); np.random.seed(SEED)
    cfg=_build_mf_dro_config("h20","Hartmann_6D",str(linear),SEED,bo_iterations=3,
        num_epochs=10,minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=1e9,
        initial_hf=36,initial_lf=60,dkl_threshold=9999,bes_delta=0.0,rollout_length=8)
    cfg.seed=SEED; cfg.use_linear_score_head=linear
    mf=DirectMFRegretOptimization(cfg,hf["make_objective"](),lf["make_objective"](),bounds)
    mf._sample_initial_points(); mf._update_ko_ensemble()
    b=mf._generate_rollout_batch(); mf._train_dt(b)
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

def score(mf,h,cf):
    if mf.dt.use_linear_score_head:
        return ((cf.to(FD)*mf.dt.coef_head(h).unsqueeze(0)).sum(-1)+mf.dt.bias_head(h)).detach()
    K=cf.shape[0]
    return mf.dt.score_head(torch.cat([h.unsqueeze(0).expand(K,-1),cf.to(FD)],dim=-1)).squeeze(-1).detach()

res={}
for linear,name in [(True,"L linear (default)"),(False,"M MLP [h;cf]")]:
    mf,batch,bounds=build(linear)
    rpm=mf.config.rollouts_per_model
    assert mf.dt.use_linear_score_head is linear, "flag did not take effect"
    ysa=_y_star_for_model(mf.ko_ensemble[0],mf.y_star_pool,seed=SEED)
    s0=batch[0]["states"][0].double()
    moved_state=0; moved_rtg=0; resid=[]; Hs=[]; SCs=[]
    mf.dt.eval()
    with torch.no_grad():
        for p in range(N_POOLS):
            Xp=bounds[0]+(bounds[1]-bounds[0])*torch.rand(200,mf.d,dtype=torch.float64,
                generator=torch.Generator().manual_seed(300+p))
            cf=build_candidate_features(mf.ko_ensemble[0],Xp,bounds,mf.c_H,mf.c_L,
                torch.zeros(mf.d,dtype=torch.float64),y_star_arr=ysa)
            h0=hid(mf,s0); sc0=score(mf,h0,cf); a0=int(sc0.argmax())
            # (1) genuine state swap: different ensemble-model block
            k=1+(p%9)
            hk=hid(mf,batch[k*rpm]["states"][0].double())
            if int(score(mf,hk,cf).argmax())!=a0: moved_state+=1
            # (2) RTG sweep in band
            ams=[int(score(mf,hid(mf,s0,rtg=t),cf).argmax()) for t in IN_BAND]
            if len(set(ams))>1: moved_rtg+=1
            # manipulation: is the score affine in cf?
            A=torch.cat([cf.to(FD),torch.ones(cf.shape[0],1,dtype=FD)],dim=1).double().numpy()
            y=sc0.double().numpy()
            w,*_=np.linalg.lstsq(A,y,rcond=None)
            resid.append(float(np.linalg.norm(y-A@w)/(np.linalg.norm(y-y.mean())+1e-12)))
            Hs.append(hid(mf,batch[k*rpm]["states"][0].double()).double().numpy()); SCs.append(sc0.double().numpy())
    res[name]={"state_moved":moved_state,"rtg_moved":moved_rtg,
               "affine_residual":float(np.mean(resid))}
    print(f"[{name}]")
    print(f"  argmax moved on genuine STATE swap : {moved_state}/{N_POOLS} = {moved_state/N_POOLS:.1%}")
    print(f"  argmax moved on in-band RTG sweep  : {moved_rtg}/{N_POOLS} = {moved_rtg/N_POOLS:.1%}")
    print(f"  relative residual of best affine fit score~cf: {np.mean(resid):.6f}")
print("="*70)
L=res["L linear (default)"]; M=res["M MLP [h;cf]"]
ok = M["affine_residual"] > 0.05
print(f"MANIPULATION (arm M not affine in cf, residual>0.05): "
      f"{M['affine_residual']:.4f} -> {'PASS' if ok else 'FAIL -- VOID'}")
if ok:
    p1 = M["state_moved"]/N_POOLS > 0.30
    print(f"PRED 1 (arm M state-swap movement >30%): "
          f"{M['state_moved']/N_POOLS:.1%} -> {'PASS' if p1 else 'FAIL'}")
    if not p1:
        print("\nPRED 2 NULL: the conditioning failure is ARCHITECTURE-INDEPENDENT")
        print("  within this family. The linear bottleneck is not the cause; the")
        print("  negative result is stronger and the empirical programme is done.")
print("="*70)
json.dump(res,open(os.path.join(os.path.dirname(__file__),"..","results","h20.json"),"w"),indent=2)
