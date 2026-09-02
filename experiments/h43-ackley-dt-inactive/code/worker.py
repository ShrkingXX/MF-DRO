import os, sys, json, time
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v]="1"
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import (DirectMFRegretOptimization, build_candidate_features,
                                _y_star_for_model)
FD=torch.float32; N_POOLS=12

def relspread(M):
    M=np.asarray(M,float); n=np.linalg.norm(M,axis=1)
    d=[np.linalg.norm(M[i]-M[j]) for i in range(len(M)) for j in range(i+1,len(M))]
    return float(np.mean(d)/np.mean(n)) if len(d) else 0.0

def probe(mf, batch, bounds, use_cs):
    """Same probes used on Hartmann, so numbers are directly comparable."""
    rpm=mf.config.rollouts_per_model
    idx=[k*rpm for k in range(len(batch)//rpm)]
    dt=mf.dt; dt.eval(); H=dt.hidden_size
    def hid(st):
        s=st.unsqueeze(0).unsqueeze(0).to(FD)
        r=torch.tensor([[[0.7]]],dtype=FD); b=torch.tensor([[[22.0]]],dtype=FD)
        ax=torch.zeros(1,1,mf.d,dtype=FD); ae=torch.zeros(1,1,dtype=FD)
        e=[dt.reward_ln(dt.reward_embedding(r)),dt.btg_ln(dt.btg_embed(b)),
           dt.state_ln(dt.state_embedding(s)),
           dt.action_ln(dt.action_embed_mf(torch.cat([ax,ae.unsqueeze(-1)],dim=-1)))]
        pos=dt.position_embedding(torch.tensor([[0]],dtype=torch.long)).repeat_interleave(4,dim=1)
        seq=torch.stack(e,dim=2).reshape(1,4,H)+pos
        cm=torch.triu(torch.ones(4,4,dtype=torch.bool),diagonal=1)
        return dt.transformer(seq,mask=cm)[0,2::4,:][0]
    out={}
    with torch.no_grad():
        S=np.stack([batch[i]["states"][0].double().numpy() for i in idx])
        Hs=np.stack([hid(batch[i]["states"][0].double()).double().numpy() for i in idx])
        out["spread_state"]=relspread(S); out["spread_h"]=relspread(Hs)
        FID=np.array([float(dt.fidelity_head(torch.tensor(h_,dtype=FD))) for h_ in Hs])
        out["fid_p_min"]=float(FID.min()); out["fid_p_max"]=float(FID.max())
        out["fid_p_sd"]=float(FID.std())
        if use_cs:
            W=np.stack([dt.coef_head(torch.tensor(h_,dtype=FD)).double().numpy() for h_ in Hs])
            wbar=W.mean(0); D=W-wbar
            out["spread_w"]=relspread(W)
            out["delta_over_wbar"]=float(np.linalg.norm(D,axis=1).mean()/np.linalg.norm(wbar))
            ysa=_y_star_for_model(mf.ko_ensemble[0],mf.y_star_pool,seed=mf.config.seed)
            moved=agree=0
            for p in range(N_POOLS):
                X=bounds[0]+(bounds[1]-bounds[0])*torch.rand(200,mf.d,dtype=torch.float64,
                    generator=torch.Generator().manual_seed(300+p))
                cf=build_candidate_features(mf.ko_ensemble[0],X,bounds,mf.c_H,mf.c_L,
                    torch.zeros(mf.d,dtype=torch.float64),y_star_arr=ysa).double().numpy()
                a0=int((cf@W[0]).argmax())
                if int((cf@W[1+(p%(len(W)-1))]).argmax())!=a0: moved+=1
                if int((cf@wbar).argmax())==a0: agree+=1
            out["state_swap_moved"]=f"{moved}/{N_POOLS}"
            out["wbar_reproduces_argmax"]=f"{agree}/{N_POOLS}"
        else:
            A=np.stack([dt.action_head(torch.tensor(h_,dtype=FD)).clamp(0,1).double().numpy()
                        for h_ in Hs])
            out["spread_action_head"]=relspread(A)
            out["action_head_sd_per_coord"]=float(np.mean(A.std(axis=0)))
    return out

def run(seed, use_cs):
    hf=get_benchmark("Ackley_10D_HF"); lf=get_benchmark("Ackley_10D_LF")
    bounds=torch.tensor([hf["domain_min"],hf["domain_max"]],dtype=torch.float64)
    torch.manual_seed(seed); np.random.seed(seed)
    cfg=_build_mf_dro_config("h43","Ackley_10D",str(use_cs),seed,bo_iterations=100,
        num_epochs=10,minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=1e9,
        initial_hf=8,initial_lf=40,dkl_threshold=9999,bes_delta=0.0,rollout_length=8)
    cfg.seed=seed; cfg.rollout_reward="mes_entropy"; cfg.use_candidate_scoring=use_cs
    mf=DirectMFRegretOptimization(cfg,hf["make_objective"](),lf["make_objective"](),bounds)
    t0=time.time(); r=mf.run()
    reg=np.array(r["hf_regret_curve"]); xs=np.array(r["x_t_trace"])
    n_impr=int(sum(1 for i in range(1,len(reg)) if reg[i]<reg[i-1]-1e-12))
    batch=mf._generate_rollout_batch()
    p=probe(mf,batch,bounds,use_cs)
    return dict(bench="Ackley_10D", use_cs=bool(use_cs), seed=seed,
                n_improvements=n_impr, frozen=bool(n_impr==0),
                final_regret=float(reg[-1]), n_iters=int(len(reg)),
                distinct=int(len({tuple(np.round(x,6)) for x in xs})),
                wall=round(time.time()-t0,1), **p)

if __name__=="__main__":
    seed=int(sys.argv[1]); d=os.path.join(os.path.dirname(__file__),"..","results")
    for use_cs in (True, False):
        o=run(seed,use_cs)
        tag="cs" if use_cs else "reg"
        with open(os.path.join(d,f"{tag}__seed{seed}.json"),"w") as f: json.dump(o,f)
        print(json.dumps(o), flush=True)
