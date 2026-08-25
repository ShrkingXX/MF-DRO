import os, sys, json
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v]="1"
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from scipy.stats import spearmanr
from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import (DirectMFRegretOptimization, build_candidate_features,
                                _y_star_for_model, compute_joint_mf_mes)
SEED=44; FD=torch.float32; N=12
hf=get_benchmark("Hartmann_6D_HF"); lf=get_benchmark("Hartmann_6D_LF")
bounds=torch.tensor([hf["domain_min"],hf["domain_max"]],dtype=torch.float64)
torch.manual_seed(SEED); np.random.seed(SEED)
cfg=_build_mf_dro_config("h32","Hartmann_6D","g",SEED,bo_iterations=3,num_epochs=10,
    minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=1e9,initial_hf=36,
    initial_lf=60,dkl_threshold=9999,bes_delta=0.0,rollout_length=8)
cfg.seed=SEED; cfg.rollout_reward="mes_entropy"
mf=DirectMFRegretOptimization(cfg,hf["make_objective"](),lf["make_objective"](),bounds)
mf._sample_initial_points(); mf._update_ko_ensemble()
batch=mf._generate_rollout_batch(); mf._train_dt(batch)
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
st=batch[0]["states"][0].double(); mf.dt.eval()
rho=[];agree=0;top10=[];rank=[]
ysa=_y_star_for_model(mf.ko_ensemble[0],mf.y_star_pool,seed=SEED)
with torch.no_grad():
    h=hid(st); w=mf.dt.coef_head(h); bb=mf.dt.bias_head(h)
    for p in range(N):
        X=bounds[0]+(bounds[1]-bounds[0])*torch.rand(200,mf.d,dtype=torch.float64,
            generator=torch.Generator().manual_seed(300+p))
        cf=build_candidate_features(mf.ko_ensemble[0],X,bounds,mf.c_H,mf.c_L,
            torch.zeros(mf.d,dtype=torch.float64),y_star_arr=ysa)
        stu=((cf.to(FD)*w.unsqueeze(0)).sum(-1)+bb).double().numpy()
        _,_,sc=compute_joint_mf_mes(mf.ko_ensemble[0],X,mf.c_H,mf.c_L)
        tea=sc.max(dim=1).values.double().numpy()   # best over fidelity, per candidate
        rho.append(spearmanr(stu,tea).correlation)
        a_s=int(stu.argmax()); a_t=int(tea.argmax())
        if a_s==a_t: agree+=1
        top10.append(len(set(np.argsort(-stu)[:10]) & set(np.argsort(-tea)[:10])))
        rank.append(int((tea>tea[a_s]).sum())+1)     # teacher's rank of student's pick
print(f"pools: {N}")
print(f"  Spearman(student score, teacher score) : {np.mean(rho):+.4f} +/- {np.std(rho)/np.sqrt(N):.4f}")
print(f"  argmax agreement                       : {agree}/{N} = {agree/N:.1%}")
print(f"  top-10 overlap (of 10)                 : {np.mean(top10):.2f}")
print(f"  teacher's RANK of student's pick /200  : median {int(np.median(rank))}, "
      f"range [{min(rank)}, {max(rank)}]")
print(f"    per-pool ranks: {sorted(rank)}")
med=float(np.median(rank))
print("\n"+"="*66)
print(f"PRED 1 (median teacher-rank of student's pick worse than 10): "
      f"{med:.0f} -> {'PASS -- distillation is lossy' if med>10 else 'FAIL -- student picks near-optimally'}")
print(f"CONSEQUENT EXPECTATION for H31: teacher-only regret "
      f"{'LOWER than 0.4007' if med>10 else 'about equal to 0.4007'}")
print("="*66)
json.dump({"rho":float(np.mean(rho)),"agree":agree/N,"top10":float(np.mean(top10)),
           "median_rank":med,"ranks":rank},
          open(os.path.join(os.path.dirname(__file__),"..","results","h32.json"),"w"),indent=2)
