import os,sys
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
sys.path.insert(0,"/Users/yurucui/Desktop/DRO-Code/DRO-aistats-submission")
import torch,numpy as np
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
import src.policy.mf_dro as M
rec={}
for K in (1,8):
    hf=get_benchmark("Hartmann_6D_HF"); lf=get_benchmark("Hartmann_6D_LF")
    bounds=torch.tensor([hf["domain_min"],hf["domain_max"]],dtype=torch.float64)
    torch.manual_seed(44); np.random.seed(44)
    cfg=_build_mf_dro_config("d","Hartmann_6D","d",44,bo_iterations=6,num_epochs=3,
        minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=1e9,initial_hf=36,
        initial_lf=60,dkl_threshold=9999,bes_delta=0.0,rollout_length=8)
    cfg.seed=44; cfg.rollout_reward="mes_entropy"; cfg.inference_context_k=K
    mf=M.DirectMFRegretOptimization(cfg,hf["make_objective"](),lf["make_objective"](),bounds)
    log=[]
    orig=mf.dt.propose_mf
    def wrapped(state,rtg,btg,**kw):
        h=kw.get("hist")
        out=orig(state,rtg,btg,**kw)
        log.append((len(h)+1 if h else 1, out[0].detach().numpy().copy(), int(out[1])))
        return out
    mf.dt.propose_mf=wrapped
    mf.run()
    rec[K]=log
    print(f"K={K}: " + " ".join(f"[t{t} ctx={c} x0={x[0]:.5f} ell={e}]" for t,(c,x,e) in enumerate(log)))
print()
for t in range(min(len(rec[1]),len(rec[8]))):
    c1,x1,e1=rec[1][t]; c8,x8,e8=rec[8][t]
    d=float(np.abs(np.asarray(x1)-np.asarray(x8)).max())
    print(f"  iter {t}: ctx {c1} vs {c8}   max|dx|={d:.3e}   ell {e1} vs {e8}")
