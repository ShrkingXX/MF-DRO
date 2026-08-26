"""H76 worker: SF-MES re-run with the full regret curve recorded.
Identical optimizer/config/seeds to h72 -- final_regret must reproduce h72
bit-for-bit, which the verdict script enforces."""
import os,sys,json,time
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[_v]="1"
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from src.baselines.mf_baselines import MultiFidelityBenchmark
from src.baselines.additive_mes import SFMESOptimizer
BUDGET=200.0
SPEC={"Currin_2D":5,"Hartmann_6D":6,"Borehole_8D":10}
RES=os.path.join(os.path.dirname(__file__),"..","results")
def run(bench,seed):
    t0=time.time(); b=MultiFidelityBenchmark(bench)
    torch.manual_seed(seed); np.random.seed(seed)
    opt=SFMESOptimizer(b,n_initial_hf=SPEC[bench],n_initial_lf=0,seed=seed,cost_budget=BUDGET)
    r=opt.run(bo_iterations=4000)
    curve=[float(v) for v in (r.get("hf_regret_curve") or r.get("regret_curve") or [])]
    return dict(bench=bench,method="SF-MES",seed=seed,budget=BUDGET,regret_curve=curve,
                final_regret=(curve[-1] if curve else float("nan")),n_iters=len(curve),
                wall_s=round(time.time()-t0,1))
if __name__=="__main__":
    bench,seed=sys.argv[1],int(sys.argv[2])
    r=run(bench,seed)
    os.makedirs(RES,exist_ok=True)
    out=os.path.join(RES,f"{bench}__SF-MES__seed{seed}.json")
    tmp=out+".tmp"; json.dump(r,open(tmp,"w"),default=float); os.replace(tmp,out)
    print(f"[done] {bench} s{seed} regret={r['final_regret']:.4f} iters={r['n_iters']}",flush=True)
