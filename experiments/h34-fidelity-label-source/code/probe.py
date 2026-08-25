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
cfg=_build_mf_dro_config("h34","Hartmann_6D","g",44,bo_iterations=1,num_epochs=1,
    minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=1e9,initial_hf=36,
    initial_lf=60,dkl_threshold=9999,bes_delta=0.0,rollout_length=8)
cfg.seed=44; cfg.rollout_reward="mes_entropy"

# Trace what joint MES chose BEFORE any floor override.
raw=[]
_orig=M.compute_joint_mf_mes
def traced(ko,roi,cH,cL,K=10):
    x,ell,sc=_orig(ko,roi,cH,cL,K=K); raw.append(int(ell)); return x,ell,sc
M.compute_joint_mf_mes=traced
try:
    mf=M.DirectMFRegretOptimization(cfg,hf["make_objective"](),lf["make_objective"](),bounds)
    mf._sample_initial_points(); mf._update_ko_ensemble()
    batch=mf._generate_rollout_batch()
finally:
    M.compute_joint_mf_mes=_orig

lab=[]; bypos={}
for t in batch:
    e=t["actions_ell"].flatten().tolist()
    m=t.get("valid_mask")
    if m is not None:
        mm=m.flatten().bool().tolist(); e=[v for v,k in zip(e,mm) if k]
    lab+= [int(v) for v in e]
    for i,v in enumerate(e): bypos.setdefault(i,[]).append(int(v))
lab=np.array(lab); rawa=np.array(raw)
print(f"training fidelity LABELS      : {lab.mean():.1%} HF   (n={len(lab)})")
print(f"teacher's raw choice (no floor): {rawa.mean():.1%} HF   (n={len(rawa)})")
gap=(lab.mean()-rawa.mean())*100
print(f"gap                            : {gap:+.1f} percentage points")
print(f"\nHF label rate by rollout position tau:")
for i in sorted(bypos): print(f"   tau={i}: {np.mean(bypos[i]):.1%}  (n={len(bypos[i])})")
print("\n"+"="*66)
print(f"PRED 1 (labels exceed teacher's rate by >=10 pp): {gap:+.1f} pp -> "
      f"{'PASS -- miscalibration is INHERITED FROM LABELS' if gap>=10 else 'FAIL'}")
if gap<10:
    print("PRED 2 NULL: the floor is not responsible; the miscalibration")
    print("  originates in the head or its loss, not its training targets.")
print("  (Does NOT address uninformativeness: p sd = 2.4e-4, corr = 0.155.)")
print("="*66)
json.dump({"label_hf":float(lab.mean()),"raw_hf":float(rawa.mean()),"gap_pp":float(gap),
           "by_tau":{str(k):float(np.mean(v)) for k,v in bypos.items()}},
          open(os.path.join(os.path.dirname(__file__),"..","results","h34.json"),"w"),indent=2)
