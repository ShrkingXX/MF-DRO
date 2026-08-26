"""H70 worker. Two arms, each exactly ONE change from h69's SF-EI:
  POOL1000 -- n_candidates 200 -> 1000 (matches MFMIGreedyOptimizer)
  ALTGP    -- GP built by mf_baselines._build_gp instead of _build_ko_style_gp
  SF-EI    -- reproduction control, must match h69 bit-for-bit
"""
import os,sys,json,time
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[_v]="1"
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from botorch.acquisition.analytic import LogExpectedImprovement
from src.baselines.mf_baselines import MultiFidelityBenchmark, _build_gp as _alt_build_gp
from src.baselines.additive_mes import SFMESOptimizer

BUDGET=200.0
NHF={"Currin_2D":5,"Hartmann_6D":6,"Borehole_8D":10}
RES=os.path.join(os.path.dirname(__file__),"..","results")

def _code():
    import subprocess
    def g(*a):
        try: return subprocess.run(["git",*a],cwd=REPO,capture_output=True,text=True,timeout=20).stdout.strip()
        except Exception: return ""
    return dict(commit=g("rev-parse","HEAD"),
                dirty=bool(g("status","--porcelain","src")))

class SFEIOptimizer(SFMESOptimizer):
    def _propose_greedy(self, X_cand):
        best_f=float(max(self.data_hf_y))
        acq=LogExpectedImprovement(model=self.gp, best_f=best_f, maximize=True)
        with torch.no_grad(): vals=acq(X_cand.unsqueeze(1))
        return X_cand[int(torch.argmax(vals))], 1

class SFEIAltGP(SFEIOptimizer):
    """Only _update_model differs: mf_baselines._build_gp, no LogNormal
    lengthscale prior and no geometric-mean initialisation."""
    def _update_model(self):
        X=torch.stack(self.data_hf_x); Y=torch.tensor(self.data_hf_y,dtype=torch.float64)
        self.gp=_alt_build_gp(X,Y,self.bounds,self.d)

def run(bench,arm,seed):
    t0=time.time(); b=MultiFidelityBenchmark(bench)
    ncand = 1000 if arm=="POOL1000" else 200
    cls = SFEIAltGP if arm=="ALTGP" else SFEIOptimizer
    torch.manual_seed(seed); np.random.seed(seed)
    opt=cls(b,n_initial_hf=NHF[bench],n_initial_lf=0,seed=seed,
            cost_budget=BUDGET,n_candidates=ncand)
    r=opt.run(bo_iterations=4000)
    curve=[float(v) for v in (r.get("hf_regret_curve") or r.get("regret_curve") or [])]
    return dict(bench=bench,arm=arm,seed=seed,n_candidates=ncand,budget=BUDGET,
                regret_curve=curve,final_regret=(curve[-1] if curve else float("nan")),
                n_improvements=sum(1 for i in range(1,len(curve)) if curve[i]<curve[i-1]-1e-9),
                n_queries=len(opt.data_hf_y),wall_s=round(time.time()-t0,1),code=_code())

if __name__=="__main__":
    bench,arm,seed=sys.argv[1],sys.argv[2],int(sys.argv[3])
    r=run(bench,arm,seed)
    out=os.path.join(RES,f"{bench}__{arm}__seed{seed}.json")
    tmp=out+".tmp"; json.dump(r,open(tmp,"w"),default=float); os.replace(tmp,out)
    print(f"[done] {bench} {arm} s{seed} regret={r['final_regret']:.4f} nq={r['n_queries']} "
          f"wall={r['wall_s']:.1f}s",flush=True)
