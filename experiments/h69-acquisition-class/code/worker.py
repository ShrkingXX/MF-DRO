"""H69 worker. SF-EI = SF-MES's optimizer with LogExpectedImprovement swapped in
for the MES acquisition. IDENTICAL surrogate (_build_ko_style_gp), loop, initial
design and regret convention -- the acquisition is the only difference.

Arm MESCHECK restores the MES acquisition and must reproduce h59's SF-MES.
"""
import os,sys,json,time
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[_v]="1"
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from botorch.acquisition.analytic import LogExpectedImprovement
from src.baselines.mf_baselines import MultiFidelityBenchmark
from src.baselines.additive_mes import SFMESOptimizer

BUDGET=200.0
NHF={"Currin_2D":5,"Hartmann_6D":6,"Borehole_8D":10}   # matches h57/h59
RES=os.path.join(os.path.dirname(__file__),"..","results")

def _code():
    import subprocess
    def g(*a):
        try: return subprocess.run(["git",*a],cwd=REPO,capture_output=True,text=True,timeout=20).stdout.strip()
        except Exception: return ""
    dirty=g("status","--porcelain","src","dro_runner.py","benchmarks.py")
    return dict(commit=g("rev-parse","HEAD"),dirty=bool(dirty),dirty_files=dirty.splitlines()[:20])

class SFEIOptimizer(SFMESOptimizer):
    """Only _propose_greedy differs from SFMESOptimizer."""
    def _propose_greedy(self, X_cand):
        best_f=float(max(self.data_hf_y))
        acq=LogExpectedImprovement(model=self.gp, best_f=best_f, maximize=True)
        with torch.no_grad():
            vals=acq(X_cand.unsqueeze(1))          # [N,1,d] for q=1 analytic acq
        return X_cand[int(torch.argmax(vals))], 1

def run(bench,arm,seed):
    t0=time.time()
    b=MultiFidelityBenchmark(bench)
    cls = SFEIOptimizer if arm=="SF-EI" else SFMESOptimizer
    torch.manual_seed(seed); np.random.seed(seed)
    opt=cls(b, n_initial_hf=NHF[bench], n_initial_lf=0, seed=seed, cost_budget=BUDGET)
    r=opt.run(bo_iterations=4000)
    curve=[float(v) for v in (r.get("hf_regret_curve") or r.get("regret_curve") or [])]
    qx=[list(map(float,np.asarray(x if not torch.is_tensor(x) else x.cpu().numpy()).reshape(-1)))
        for x in opt.data_hf_x]
    qy=[float(v) for v in opt.data_hf_y]
    return dict(bench=bench,arm=arm,seed=seed,budget=BUDGET,regret_curve=curve,
                final_regret=(curve[-1] if curve else float("nan")),
                n_improvements=sum(1 for i in range(1,len(curve)) if curve[i]<curve[i-1]-1e-9),
                query_x=qx,query_y=qy,n_queries=len(qy),
                wall_s=round(time.time()-t0,1),code=_code())

if __name__=="__main__":
    bench,arm,seed=sys.argv[1],sys.argv[2],int(sys.argv[3])
    out=os.path.join(RES,f"{bench}__{arm}__seed{seed}.json")
    r=run(bench,arm,seed)
    tmp=out+".tmp"; json.dump(r,open(tmp,"w"),default=float); os.replace(tmp,out)
    print(f"[done] {bench} {arm} seed{seed} regret={r['final_regret']:.4f} "
          f"nq={r['n_queries']} improv={r['n_improvements']} wall={r['wall_s']/60:.1f}m",flush=True)
