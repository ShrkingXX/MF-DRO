"""H79b: MI-Greedy re-run with the full regret curve recorded."""
import os,sys,json,time
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[_v]="1"
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from src.baselines.mf_baselines import MultiFidelityBenchmark, MFMIGreedyOptimizer
BUDGET=200.0; SPEC={"Borehole_8D":dict(n_hf=10,n_lf=20)}
RES=os.path.join(os.path.dirname(__file__),"..","results_mig")
if __name__=="__main__":
    bench,seed=sys.argv[1],int(sys.argv[2]); sp=SPEC[bench]
    t0=time.time(); b=MultiFidelityBenchmark(bench)
    torch.manual_seed(seed); np.random.seed(seed)
    opt=MFMIGreedyOptimizer(b,n_initial_hf=sp["n_hf"],n_initial_lf=sp["n_lf"],
                            seed=seed,cost_budget=BUDGET)
    r=opt.run(bo_iterations=4000)
    curve=[float(v) for v in (r.get("hf_regret_curve") or r.get("regret_curve") or [])]
    os.makedirs(RES,exist_ok=True)
    out=os.path.join(RES,f"{bench}__MF-MI-Greedy__seed{seed}.json")
    json.dump(dict(bench=bench,method="MF-MI-Greedy",seed=seed,regret_curve=curve,
                   final_regret=(curve[-1] if curve else float("nan")),n_iters=len(curve),
                   wall_s=round(time.time()-t0,1)),open(out+".tmp","w"),default=float)
    os.replace(out+".tmp",out)
    print(f"[done] {bench} s{seed} regret={curve[-1]:.6f} iters={len(curve)}",flush=True)
