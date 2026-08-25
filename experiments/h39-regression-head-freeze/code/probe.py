"""Would the frozen incumbent return under the REGRESSION head?

Hypothesis: candidate scoring MASKS the conditioning failure. Even with a
constant scoring rule w, the argmax moves between iterations because the
200-candidate pool is REDRAWN each iteration and its GP features update. The
regression head has no such external randomness: x = action_head(h), so if h is
near-constant the proposed point is near-constant -> frozen incumbent.

Trains with use_candidate_scoring=False so action_head actually receives
gradient (under candidate scoring it gets none and measuring it is meaningless).
"""
import os,sys
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
sys.path.insert(0,"/Users/yurucui/Desktop/DRO-Code/DRO-aistats-submission")
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization
FD=torch.float32

def spread(X):
    X=np.asarray(X,dtype=float)
    d=[np.linalg.norm(X[i]-X[j]) for i in range(len(X)) for j in range(i+1,len(X))]
    return float(np.mean(d))

for use_cs,label in [(False,"REGRESSION head (action_head)"),(True,"CANDIDATE scoring")]:
    hf=get_benchmark("Currin_2D_HF"); lf=get_benchmark("Currin_2D_LF")
    bounds=torch.tensor([hf["domain_min"],hf["domain_max"]],dtype=torch.float64)
    torch.manual_seed(44); np.random.seed(44)
    cfg=_build_mf_dro_config("rh","Currin_2D","r",44,bo_iterations=14,num_epochs=10,
        minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=1e9,initial_hf=5,
        initial_lf=15,dkl_threshold=9999,bes_delta=0.0,rollout_length=8)
    cfg.seed=44; cfg.rollout_reward="mes_entropy"; cfg.use_candidate_scoring=use_cs
    mf=DirectMFRegretOptimization(cfg,hf["make_objective"](),lf["make_objective"](),bounds)
    r=mf.run()
    xs=np.array(r["x_t_trace"]); reg=np.array(r["hf_regret_curve"])
    n_impr=int(sum(1 for i in range(1,len(reg)) if reg[i]<reg[i-1]-1e-12))
    print(f"\n=== {label} ===", flush=True)
    print(f"  use_candidate_scoring   : {mf.use_candidate_scoring}")
    print(f"  proposed-x mean pairwise spread : {spread(xs):.6f}")
    print(f"  per-coordinate sd               : {np.round(xs.std(axis=0),5).tolist()}")
    print(f"  distinct proposals (6dp)        : {len({tuple(np.round(x,6)) for x in xs})}/{len(xs)}")
    print(f"  incumbent improvements          : {n_impr}  -> {'FROZEN' if n_impr==0 else 'moving'}")
    print(f"  final regret                    : {reg[-1]:.4f}")
