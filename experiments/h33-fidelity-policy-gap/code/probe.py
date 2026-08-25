import os, sys, json
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v]="1"
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import (DirectMFRegretOptimization, build_candidate_features,
                                _y_star_for_model, compute_joint_mf_mes)
SEED=44; FD=torch.float32; N=24
hf=get_benchmark("Hartmann_6D_HF"); lf=get_benchmark("Hartmann_6D_LF")
bounds=torch.tensor([hf["domain_min"],hf["domain_max"]],dtype=torch.float64)
torch.manual_seed(SEED); np.random.seed(SEED)
cfg=_build_mf_dro_config("h33","Hartmann_6D","g",SEED,bo_iterations=3,num_epochs=10,
    minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=1e9,initial_hf=36,
    initial_lf=60,dkl_threshold=9999,bes_delta=0.0,rollout_length=8)
cfg.seed=SEED; cfg.rollout_reward="mes_entropy"
mf=DirectMFRegretOptimization(cfg,hf["make_objective"](),lf["make_objective"](),bounds)
mf._sample_initial_points(); mf._update_ko_ensemble()
batch=mf._generate_rollout_batch(); mf._train_dt(batch)
rpm=mf.config.rollouts_per_model
states=[batch[k*rpm]["states"][0].double() for k in range(len(batch)//rpm)]
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
mf.dt.eval(); ps=[]; ells=[]
with torch.no_grad():
    for p_i in range(N):
        st=states[p_i % len(states)]
        X=bounds[0]+(bounds[1]-bounds[0])*torch.rand(200,mf.d,dtype=torch.float64,
            generator=torch.Generator().manual_seed(900+p_i))
        _,ell,_=compute_joint_mf_mes(mf.ko_ensemble[0],X,mf.c_H,mf.c_L)
        ells.append(int(ell))
        ps.append(float(mf.dt.fidelity_head(hid(st))))
ps=np.array(ps); ells=np.array(ells)
print(f"decisions: {N}")
print(f"  teacher ell: HF on {ells.sum()}/{N} = {ells.mean():.1%}   (varies: {len(set(ells.tolist()))>1})")
print(f"  student p  : min {ps.min():.4f}  max {ps.max():.4f}  sd {ps.std():.6f}")
c = float(np.corrcoef(ps,ells)[0,1]) if ps.std()>1e-12 and ells.std()>0 else float('nan')
print(f"  corr(p, teacher ell) : {c:+.4f}")
print(f"  implied student HF rate (mean p) : {ps.mean():.1%}   vs teacher {ells.mean():.1%}")
print("\n"+"="*66)
if len(set(ells.tolist()))<2:
    print("PRED 2 FAILS: teacher's ell does not vary -> comparison VACUOUS, VOID.")
else:
    ok = abs(c)<0.2 if np.isfinite(c) else True
    print(f"PRED 1 (|corr(p, teacher ell)| < 0.2): {abs(c):.4f} -> "
          f"{'PASS -- student fidelity is uninformative about the teacher' if ok else 'FAIL'}")
    if not ok: print("PRED 3 NULL: fidelity channel tracks the teacher; H32 hypothesis wrong.")
print("="*66)
json.dump({"teacher_hf_rate":float(ells.mean()),"student_p_mean":float(ps.mean()),
           "student_p_sd":float(ps.std()),"corr":c,"n":N},
          open(os.path.join(os.path.dirname(__file__),"..","results","h33.json"),"w"),indent=2)
