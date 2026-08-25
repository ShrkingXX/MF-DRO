import os, sys, json, time
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v]="1"
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization

ARM=sys.argv[1]; SEED=int(sys.argv[2])            # ARM in {GLOBAL, ROI}
OUT=os.path.join(os.path.dirname(__file__),"..","results",f"{ARM}__seed{SEED}.json")
hf=get_benchmark("Hartmann_6D_HF"); lf=get_benchmark("Hartmann_6D_LF")
bounds=torch.tensor([hf["domain_min"],hf["domain_max"]],dtype=torch.float64)
torch.manual_seed(SEED); np.random.seed(SEED)
cfg=_build_mf_dro_config("h56","Hartmann_6D","mfdro",SEED,bo_iterations=2000,num_epochs=10,
    minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=100.0,initial_hf=6,
    initial_lf=45,dkl_threshold=9999,bes_delta=0.0,rollout_length=8)
cfg.seed=SEED; cfg.rollout_reward="mes_entropy"
# regression head is the default now; state it anyway so the record is explicit
cfg.use_candidate_scoring=False
cfg.use_roi=(ARM!="GLOBAL"); cfg.roi_raw_pool=2000
cfg.roi_mode=("mes" if ARM=="MESROI" else "ucb")
cfg.roi_beta_sqrt=0.5; cfg.roi_top_q=0.10
t0=time.time()
mf=DirectMFRegretOptimization(cfg,hf["make_objective"](),lf["make_objective"](),bounds)
r=mf.run()
xs=np.array(r["x_t_trace"]); span=(bounds[1]-bounds[0]).numpy()
xs_n=(xs-bounds[0].numpy())/span
st=mf.roi_stats or []
roi={}
if st:
    a=np.array([e['accept_frac'] for e in st])
    roi=dict(n_rollouts=len(st),accept_mean=float(a.mean()),accept_min=float(a.min()),
             accept_max=float(a.max()))
    if 'min_dist_to_xstar' in st[0]:
        d=np.array([e['min_dist_to_xstar'] for e in st])
        w=np.array([e['frac_within_0.2'] for e in st])
        roi.update(min_dist_mean=float(d.mean()),min_dist_min=float(d.min()),
                   frac_within_0p2_mean=float(w.mean()))
json.dump(dict(arm=ARM,seed=SEED,
    final_regret=float(r["hf_regret_curve"][-1]),
    hf_regret_curve=[float(v) for v in r["hf_regret_curve"]],
    inference_regret_curve=[float(v) for v in r["inference_regret_curve"]],
    cost_curve=[float(v) for v in r["cost_curve"]],
    fidelity_trace=[int(v) for v in r["fidelity_trace"]],
    n_iters=len(xs), n_hf=int(sum(r["fidelity_trace"])),
    distinct=len({tuple(np.round(x,6)) for x in xs_n}),
    wall=time.time()-t0, roi=roi), open(OUT,"w"))
print(f"[done] {ARM} seed{SEED} regret={r['hf_regret_curve'][-1]:.4f} iters={len(xs)} "
      f"n_hf={int(sum(r['fidelity_trace']))} accept={roi.get('accept_mean','n/a')} "
      f"({(time.time()-t0)/60:.1f} min)",flush=True)
