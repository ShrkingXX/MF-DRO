import os, sys, json
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v]="1"
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization, compute_joint_mf_mes

ARM=sys.argv[1]; SEED=int(sys.argv[2])          # ARM in {S,R,T}
RES=os.path.join(os.path.dirname(__file__),"..","results")
OUT=os.path.join(RES,f"{ARM}__seed{SEED}.json")
NPOOL=200

hf=get_benchmark("Hartmann_6D_HF"); lf=get_benchmark("Hartmann_6D_LF")
bounds=torch.tensor([hf["domain_min"],hf["domain_max"]],dtype=torch.float64)
lo,hi=bounds[0],bounds[1]
torch.manual_seed(SEED); np.random.seed(SEED)
cfg=_build_mf_dro_config("h55","Hartmann_6D","mfdro",SEED,bo_iterations=50,num_epochs=10,
    minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=1e12,initial_hf=6,
    initial_lf=45,dkl_threshold=9999,bes_delta=0.0,rollout_length=8)
cfg.seed=SEED; cfg.rollout_reward="mes_entropy"
cfg.use_candidate_scoring=(ARM in ("S","T"))   # T builds the pool, then overrides the proposal
mf=DirectMFRegretOptimization(cfg,hf["make_objective"](),lf["make_objective"](),bounds)
d=mf.d

if ARM=="T":
    # Teacher-only: identical to h31's construction. Everything downstream of the
    # proposal (initial design, cost accounting, regret) is the same code path.
    def _teacher(state,rtg,btg,timestep=0,use_candidate_scoring=False,
                 candidate_features=None,fidelity_sampling=True,hist=None):
        cf=candidate_features.double()
        X_raw=lo+(hi-lo)*cf[:,:d]
        x_raw,ell,_=compute_joint_mf_mes(mf.ko_ensemble[0],X_raw,mf.c_H,mf.c_L)
        mf.dt.last_p_pred=float(ell)
        return ((x_raw-lo)/(hi-lo)).clamp(0.,1.).float(), int(ell)
    _base=_teacher
else:
    _base=mf.dt.propose_mf

diag=[]; seen=set()
def _hooked(state,rtg,btg,**kw):
    x_norm,ell=_base(state,rtg,btg,**kw)
    it=len(mf.iteration_log)
    if it in seen:                 # snapshot/probe re-calls in the same iteration
        return x_norm,ell
    seen.add(it)
    try:
        with torch.no_grad():
            pool=lo+(hi-lo)*torch.rand(NPOOL,d,dtype=torch.float64)
            x_raw=(lo+(hi-lo)*x_norm.double().reshape(-1)).reshape(1,d)
            _,_,sc=compute_joint_mf_mes(mf.ko_ensemble[0],torch.cat([pool,x_raw],0),
                                        mf.c_H,mf.c_L)
            t_flat=sc[:NPOOL].reshape(-1)
            t_best=float(t_flat.max()); ti=int(t_flat.argmax()//2); t_ell=int(t_flat.argmax()%2)
            dt_best=float(sc[NPOOL].max())
            diag.append(dict(iter=it,
                delta_acq=dt_best-t_best,
                teacher_best=t_best, dt_val=dt_best,
                dt_at_own_ell=float(sc[NPOOL,int(ell)]),
                dist_to_teacher=float(((x_raw[0]-pool[ti])/(hi-lo)).norm()),
                ell=int(ell), teacher_ell=t_ell, ell_agree=bool(int(ell)==t_ell)))
    except Exception as e:
        diag.append(dict(iter=it,error=repr(e)))
    return x_norm,ell

mf.dt.propose_mf=_hooked
r=mf.run()

xs=np.array(r["x_t_trace"]); span=(hi-lo).numpy()
xs_n=(xs-lo.numpy())/span
json.dump(dict(arm=ARM,seed=SEED,
    hf_regret_curve=r["hf_regret_curve"],
    inference_regret_curve=r["inference_regret_curve"],
    cumulative_cost_curve=r["cumulative_cost_curve"],
    fidelity_trace=r["fidelity_trace"],
    x_t_trace=xs.tolist(),
    distinct_queries=len({tuple(np.round(x,6)) for x in xs_n}),
    n_iters=len(xs), diag=diag), open(OUT,"w"))
print(f"[done] {ARM} seed{SEED}  final_regret={r['hf_regret_curve'][-1]:.4f}  "
      f"distinct={len({tuple(np.round(x,6)) for x in xs_n})}/{len(xs)}  "
      f"mean_delta_acq={np.mean([e.get('delta_acq',np.nan) for e in diag]):+.5f}",flush=True)
