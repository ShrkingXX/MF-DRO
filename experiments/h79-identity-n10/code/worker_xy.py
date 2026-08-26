"""H79c: both methods re-run with query COORDINATES recorded."""
import os,sys,json,time
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[_v]="1"
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from botorch.acquisition.analytic import LogExpectedImprovement
from src.baselines.mf_baselines import MultiFidelityBenchmark, MFMIGreedyOptimizer
from src.baselines.additive_mes import SFMESOptimizer
BUDGET=200.0; NHF=10; NLF=20
RES=os.path.join(os.path.dirname(__file__),"..","results_xy")

class SFEI(SFMESOptimizer):
    def _propose_greedy(self, X_cand):
        acq=LogExpectedImprovement(model=self.gp,best_f=float(max(self.data_hf_y)),maximize=True)
        with torch.no_grad(): v=acq(X_cand.unsqueeze(1))
        return X_cand[int(torch.argmax(v))],1

if __name__=="__main__":
    method,seed=sys.argv[1],int(sys.argv[2]); b=MultiFidelityBenchmark("Borehole_8D")
    t0=time.time(); torch.manual_seed(seed); np.random.seed(seed)
    if method=="MI-Greedy":
        opt=MFMIGreedyOptimizer(b,n_initial_hf=NHF,n_initial_lf=NLF,seed=seed,cost_budget=BUDGET)
        r=opt.run(bo_iterations=4000)
        X=[np.asarray(x if not torch.is_tensor(x) else x.cpu().numpy(),dtype=float).reshape(-1) for x in opt.data_H_x]
        Y=[float(v) for v in opt.data_H_y]
    else:
        opt=SFEI(b,n_initial_hf=NHF,n_initial_lf=0,seed=seed,cost_budget=BUDGET,n_candidates=1000)
        r=opt.run(bo_iterations=4000)
        X=[np.asarray(x if not torch.is_tensor(x) else x.cpu().numpy(),dtype=float).reshape(-1) for x in opt.data_hf_x]
        Y=[float(v) for v in opt.data_hf_y]
    curve=[float(v) for v in (r.get("hf_regret_curve") or r.get("regret_curve") or [])]
    i=int(np.argmax(Y))
    os.makedirs(RES,exist_ok=True)
    out=os.path.join(RES,f"{method}__seed{seed}.json")
    json.dump(dict(method=method,seed=seed,final_regret=(curve[-1] if curve else float("nan")),
                   best_y=Y[i],best_x=list(map(float,X[i])),n_hf=len(Y),
                   wall_s=round(time.time()-t0,1)),open(out+".tmp","w"),default=float)
    os.replace(out+".tmp",out)
    print(f"[done] {method} s{seed} best_y={Y[i]:.9f} regret={curve[-1]:.6f}",flush=True)
