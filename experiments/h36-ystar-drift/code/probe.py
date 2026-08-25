import os, sys, json
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v]="1"
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
import src.policy.mf_dro as M

hf=get_benchmark("Hartmann_6D_HF"); lf=get_benchmark("Hartmann_6D_LF")
bounds=torch.tensor([hf["domain_min"],hf["domain_max"]],dtype=torch.float64)
torch.manual_seed(44); np.random.seed(44)
cfg=_build_mf_dro_config("h36","Hartmann_6D","g",44,bo_iterations=1,num_epochs=1,
    minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=1e9,initial_hf=36,
    initial_lf=60,dkl_threshold=9999,bes_delta=0.0,rollout_length=8)
cfg.seed=44; cfg.rollout_reward="mes_entropy"
tr=[]
_orig=M.compute_joint_mf_mes
def traced(ko,roi,cH,cL,K=10):
    x,ell,sc=_orig(ko,roi,cH,cL,K=K)
    proxy=M._build_hf_proxy_model(ko)
    ys=M.thompson_sample_y_star(proxy,roi,K=64)
    with torch.no_grad():
        muH,_=ko.hf_posterior(roi); muL,_=ko.lf_posterior(roi)
    ys=np.asarray(ys,dtype=float).reshape(-1)
    tr.append(dict(ys_mean=float(ys.mean()), ys_sd=float(ys.std()),
                   muH_max=float(muH.max()), muH_mean=float(muH.mean()),
                   muL_mean=float(muL.mean()),
                   gap=float(ys.mean()-float(muH.max())),
                   lf=float(sc[:,0].max()), hf=float(sc[:,1].max())))
    return x,ell,sc
M.compute_joint_mf_mes=traced
try:
    mf=M.DirectMFRegretOptimization(cfg,hf["make_objective"](),lf["make_objective"](),bounds)
    mf._sample_initial_points(); mf._update_ko_ensemble()
    n0=len(tr); batch=mf._generate_rollout_batch()
finally:
    M.compute_joint_mf_mes=_orig
T=cfg.rollout_length; roll=tr[n0:]
by={t:[] for t in range(T)}
for i,r in enumerate(roll): by[i%T].append(r)
print(f"{'tau':>4} {'mean y*':>9} {'sd y*':>8} {'max muH':>9} {'gap':>8} {'mean muL':>9} {'LF/c_L':>8}")
g0=g7=None
for t in range(T):
    r=by[t]; f=lambda k: np.mean([x[k] for x in r])
    if t==0: g0=f('gap')
    if t==T-1: g7=f('gap')
    print(f"{t:>4} {f('ys_mean'):>9.4f} {f('ys_sd'):>8.4f} {f('muH_max'):>9.4f} "
          f"{f('gap'):>8.4f} {f('muL_mean'):>9.4f} {f('lf'):>8.4f}")
X=bounds[0]+(bounds[1]-bounds[0])*torch.rand(200,mf.d,dtype=torch.float64,
    generator=torch.Generator().manual_seed(4242))
ko=mf.ko_ensemble[0]; proxy=M._build_hf_proxy_model(ko)
ys=np.asarray(M.thompson_sample_y_star(proxy,X,K=64),dtype=float).reshape(-1)
with torch.no_grad(): muH,_=ko.hf_posterior(X); muL,_=ko.lf_posterior(X)
_,_,scr=_orig(ko,X,mf.c_H,mf.c_L)
print(f"{'REAL':>4} {ys.mean():>9.4f} {ys.std():>8.4f} {float(muH.max()):>9.4f} "
      f"{ys.mean()-float(muH.max()):>8.4f} {float(muL.mean()):>9.4f} {float(scr[:,0].max()):>8.4f}")
print("\n"+"="*72)
rose = g7 > g0
print(f"PRED 1 (gap y* - max muH grows with tau): {g0:.4f} -> {g7:.4f} -> "
      f"{'PASS' if rose else 'FAIL'}")
if not rose:
    print("PRED 2 NULL: y* drift does not explain it either. Per the protocol's")
    print("  discipline note, STOP proposing mechanisms -- report the LF collapse")
    print("  as measured but unexplained.")
print("="*72)
json.dump({"gap0":g0,"gap7":g7},open(os.path.join(os.path.dirname(__file__),"..",
          "results","h36.json"),"w"),indent=2)
