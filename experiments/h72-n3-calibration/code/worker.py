"""H72 worker: cheap baselines only (MI-Greedy, GP-UCB, SF-MES, SF-EI) at n=10.
Same specs as h57/h59 so results are comparable to the standings table."""
import os,sys,json,time
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[_v]="1"
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from botorch.acquisition.analytic import LogExpectedImprovement
from src.baselines.mf_baselines import MultiFidelityBenchmark, MFMIGreedyOptimizer, MFGPUCBOptimizer
from src.baselines.additive_mes import SFMESOptimizer

BUDGET=200.0
SPEC={"Currin_2D":dict(n_hf=5,n_lf=15),"Hartmann_6D":dict(n_hf=6,n_lf=45),
      "Borehole_8D":dict(n_hf=10,n_lf=20)}
RES=os.path.join(os.path.dirname(__file__),"..","results")

class SFEIOptimizer(SFMESOptimizer):
    def _propose_greedy(self, X_cand):
        acq=LogExpectedImprovement(model=self.gp,best_f=float(max(self.data_hf_y)),maximize=True)
        with torch.no_grad(): v=acq(X_cand.unsqueeze(1))
        return X_cand[int(torch.argmax(v))],1

def run(bench,method,seed):
    t0=time.time(); sp=SPEC[bench]; b=MultiFidelityBenchmark(bench)
    torch.manual_seed(seed); np.random.seed(seed)
    if method in ("MF-MI-Greedy","MF-GP-UCB"):
        common=dict(n_initial_hf=sp["n_hf"],n_initial_lf=sp["n_lf"],seed=seed,cost_budget=BUDGET)
        opt=MFMIGreedyOptimizer(b,**common) if method=="MF-MI-Greedy" else MFGPUCBOptimizer(b,**common)
    else:
        cls=SFEIOptimizer if method=="SF-EI" else SFMESOptimizer
        opt=cls(b,n_initial_hf=sp["n_hf"],n_initial_lf=0,seed=seed,cost_budget=BUDGET)
    r=opt.run(bo_iterations=4000)
    curve=[float(v) for v in (r.get("hf_regret_curve") or r.get("regret_curve") or [])]
    return dict(bench=bench,method=method,seed=seed,budget=BUDGET,
                final_regret=(curve[-1] if curve else float("nan")),
                n_iters=len(curve),wall_s=round(time.time()-t0,1))

if __name__=="__main__":
    bench,method,seed=sys.argv[1],sys.argv[2],int(sys.argv[3])
    r=run(bench,method,seed)
    out=os.path.join(RES,f"{bench}__{method}__seed{seed}.json")
    os.makedirs(os.path.dirname(out),exist_ok=True)
    tmp=out+".tmp"; json.dump(r,open(tmp,"w"),default=float); os.replace(tmp,out)
    print(f"[done] {bench} {method} s{seed} regret={r['final_regret']:.4f} wall={r['wall_s']:.1f}s",flush=True)
