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
cfg=_build_mf_dro_config("h35","Hartmann_6D","g",44,bo_iterations=1,num_epochs=1,
    minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=1e9,initial_hf=36,
    initial_lf=60,dkl_threshold=9999,bes_delta=0.0,rollout_length=8)
cfg.seed=44; cfg.rollout_reward="mes_entropy"

# Trace, per call, the winning HF-vs-LF cost-normalised scores and posterior spread.
trace=[]
_orig=M.compute_joint_mf_mes
def traced(ko,roi,cH,cL,K=10):
    x,ell,sc=_orig(ko,roi,cH,cL,K=K)
    with torch.no_grad():
        muH,vH=ko.hf_posterior(roi); muL,vL=ko.lf_posterior(roi)
    trace.append(dict(lf_best=float(sc[:,0].max()), hf_best=float(sc[:,1].max()),
                      ell=int(ell), sH=float(vH.clamp_min(0).sqrt().mean()),
                      sL=float(vL.clamp_min(0).sqrt().mean())))
    return x,ell,sc
M.compute_joint_mf_mes=traced
try:
    mf=M.DirectMFRegretOptimization(cfg,hf["make_objective"](),lf["make_objective"](),bounds)
    mf._sample_initial_points(); mf._update_ko_ensemble()
    n_before=len(trace)
    batch=mf._generate_rollout_batch()
finally:
    M.compute_joint_mf_mes=_orig

T=cfg.rollout_length
roll=trace[n_before:]
# rollout calls come in blocks of T per trajectory
by={t:[] for t in range(T)}
for i,r in enumerate(roll): by[i%T].append(r)
print(f"rollout teacher-calls traced: {len(roll)}  (T={T})")
print(f"\n{'tau':>4} {'HF/c_H':>10} {'LF/c_L':>10} {'ratio':>8} {'sigma_H':>9} {'sigma_L':>9} {'sH/sL':>7} {'HF rate':>8}")
ratios=[]
for t in range(T):
    r=by[t]
    hb=np.mean([x['hf_best'] for x in r]); lb=np.mean([x['lf_best'] for x in r])
    sH=np.mean([x['sH'] for x in r]); sL=np.mean([x['sL'] for x in r])
    rat=hb/max(lb,1e-12); ratios.append(rat)
    print(f"{t:>4} {hb:>10.4f} {lb:>10.4f} {rat:>8.3f} {sH:>9.4f} {sL:>9.4f} "
          f"{sH/max(sL,1e-12):>7.3f} {np.mean([x['ell'] for x in r]):>8.1%}")
# real-inference comparison: fresh pool on the REAL (non-fantasy) model
X=bounds[0]+(bounds[1]-bounds[0])*torch.rand(200,mf.d,dtype=torch.float64,
    generator=torch.Generator().manual_seed(4242))
_,ell_r,sc_r=_orig(mf.ko_ensemble[0],X,mf.c_H,mf.c_L)
with torch.no_grad():
    _,vH=mf.ko_ensemble[0].hf_posterior(X); _,vL=mf.ko_ensemble[0].lf_posterior(X)
hb=float(sc_r[:,1].max()); lb=float(sc_r[:,0].max())
sH=float(vH.clamp_min(0).sqrt().mean()); sL=float(vL.clamp_min(0).sqrt().mean())
print(f"{'REAL':>4} {hb:>10.4f} {lb:>10.4f} {hb/max(lb,1e-12):>8.3f} {sH:>9.4f} {sL:>9.4f} "
      f"{sH/max(sL,1e-12):>7.3f} {'ell='+str(ell_r):>8}")
print("\n"+"="*74)
rises = ratios[-1] > ratios[0]
print(f"PRED 1 (HF:LF ratio rises with tau): {ratios[0]:.3f} -> {ratios[-1]:.3f} -> "
      f"{'PASS' if rises else 'FAIL'}")
print(f"   and higher at tau=7 than at REAL: {ratios[-1]:.3f} vs {hb/max(lb,1e-12):.3f} -> "
      f"{'PASS' if ratios[-1] > hb/max(lb,1e-12) else 'FAIL'}")
if not rises:
    print("PRED 3 NULL: ratio flat in tau -- fantasy accumulation is NOT the driver.")
print("="*74)
json.dump({"ratios":ratios,"real_ratio":hb/max(lb,1e-12)},
          open(os.path.join(os.path.dirname(__file__),"..","results","h35.json"),"w"),indent=2)
