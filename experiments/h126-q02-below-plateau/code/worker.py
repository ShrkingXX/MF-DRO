"""H84 worker. MF-DRO only, four ROI arms. Adapted from h83's worker; the
trace-recording and checkpoint machinery is unchanged so h83's arm-A runs stay
directly comparable."""
import os, sys, json, time, threading
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v]="1"
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config

BUDGET=200.0; CKPT_S=20
SPEC={"Hartmann_6D":dict(n_hf=6,n_lf=45),"Borehole_8D":dict(n_hf=10,n_lf=20)}
# arm -> ROI configuration. ROI-OFF reproduces h83 exactly (bit-identity gated).
ARMS={
 "ROI-OFF": dict(use_roi=False),
 "ROI-FIX2": dict(use_roi=True, roi_beta_mode='fixed',    roi_beta_sqrt=2.0),
 "ROI-Q10":  dict(use_roi=True, roi_beta_mode='quantile', roi_target_accept=0.10),
 # H126: one point below the measured plateau. 5x tighter than ROI-Q10.
 "ROI-Q02":  dict(use_roi=True, roi_beta_mode='quantile', roi_target_accept=0.02),
 "ROI-ANN":  dict(use_roi=True, roi_beta_mode='quantile',
                  roi_accept_start=0.50, roi_accept_end=0.05),
}
RES=os.path.join(os.path.dirname(__file__),"..","results")

def _code_state():
    import subprocess
    def g(*a):
        try: return subprocess.run(["git",*a],cwd=REPO,capture_output=True,text=True,timeout=20).stdout.strip()
        except Exception: return ""
    dirty=g("status","--porcelain","src","dro_runner.py","benchmarks.py")
    return dict(commit=g("rev-parse","HEAD"),dirty=bool(dirty),dirty_files=dirty.splitlines()[:20])
CODE=_code_state()

def _atomic(path,obj):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    tmp=path+".tmp"; json.dump(obj,open(tmp,"w"),default=float); os.replace(tmp,path)

def run(bench, arm, seed, ckpt_path):
    sp=SPEC[bench]; t0=time.time()
    hf=get_benchmark(f"{bench}_HF"); lf=get_benchmark(f"{bench}_LF")
    bounds=torch.tensor([hf["domain_min"],hf["domain_max"]],dtype=torch.float64)
    c_H=float(hf["cost"]); c_L=float(lf["cost"])
    torch.manual_seed(seed); np.random.seed(seed)
    trace=[]; state={"phase":"init"}; acc={"c":0.0}
    n_init=sp["n_hf"]+sp["n_lf"]
    def rec(x,y,fid,is_init=None):
        x=np.asarray(x,dtype=float).reshape(-1)
        acc["c"]+= c_H if fid else c_L
        trace.append(dict(i=len(trace),fid=int(fid),x=x.tolist(),y=float(y),
                          cost_cum=acc["c"],
                          is_init=bool(len(trace)<n_init if is_init is None else is_init)))
    def dump():
        _atomic(ckpt_path,dict(bench=bench,method=arm,seed=seed,code=CODE,
                               phase=state["phase"],n_queries=len(trace),
                               elapsed_s=round(time.time()-t0,1),queries=trace[:]))
    stop=threading.Event()
    threading.Thread(target=lambda:[dump() for _ in iter(lambda:stop.wait(CKPT_S),True)],
                     daemon=True).start()
    try:
        from src.policy.mf_dro import DirectMFRegretOptimization
        cfg=_build_mf_dro_config("h84",bench,arm,seed,bo_iterations=4000,num_epochs=10,
            minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=BUDGET,
            initial_hf=sp["n_hf"],initial_lf=sp["n_lf"],dkl_threshold=9999,
            bes_delta=0.0,rollout_length=8)
        cfg.seed=seed; cfg.rollout_reward="mes_entropy"
        cfg.use_candidate_scoring=False              # regression head (the method)
        for k,v in ARMS[arm].items(): setattr(cfg,k,v)
        mf=DirectMFRegretOptimization(cfg,hf["make_objective"](),lf["make_objective"](),bounds)
        state["phase"]="running"
        def mirror():
            while not stop.wait(2.0):
                for e in mf.iteration_log[len(trace):]:
                    trace.append(dict(i=len(trace),fid=int(e["ell_t"]),
                        x=list(np.asarray(e["x_t"],dtype=float).reshape(-1)),
                        y=float(e["y_t"]),cost_cum=float(e["cumulative_cost"])))
        threading.Thread(target=mirror,daemon=True).start()
        r=mf.run()
        trace.clear(); acc["c"]=0.0
        for x,y in zip(list(mf.data_hf_x)[:sp["n_hf"]],list(mf.data_hf_y)[:sp["n_hf"]]):
            rec(x.cpu().numpy(),y,1,is_init=True)
        for x,y in zip(list(mf.data_lf_x)[:sp["n_lf"]],list(mf.data_lf_y)[:sp["n_lf"]]):
            rec(x.cpu().numpy(),y,0,is_init=True)
        for e in mf.iteration_log:
            trace.append(dict(i=len(trace),fid=int(e["ell_t"]),
                x=list(np.asarray(e["x_t"],dtype=float).reshape(-1)),
                y=float(e["y_t"]),cost_cum=float(e["cumulative_cost"]),
                regret=float(e["regret"])))
        # ROI diagnostics: summarise rather than store every rollout's record
        rs=getattr(mf,"roi_stats",None) or []
        if rs:
            def _m(k): 
                v=[d[k] for d in rs if d.get(k) is not None]
                return (float(np.mean(v)) if v else None)
            r["roi_summary"]=dict(n_records=len(rs),accept_frac=_m("accept_frac"),
                beta_sqrt=_m("beta_sqrt"),n_distinct=_m("n_distinct"),
                n_draws=_m("n_draws"),min_dist_to_xstar=_m("min_dist_to_xstar"),
                frac_within_02=_m("frac_within_0.2"))
    finally:
        stop.set()
    c=r.get("hf_regret_curve") or r.get("regret_curve") or []
    r={k:v for k,v in r.items() if k not in ("acq_info","X_hf","Y_hf")}
    r["queries"]=trace
    r["_meta"]=dict(bench=bench,method=arm,seed=seed,budget=BUDGET,c_H=c_H,c_L=c_L,**sp)
    r["_code"]=CODE; r["_wall_s"]=round(time.time()-t0,1)
    r["final_regret"]=float(c[-1]) if c else float("nan")
    state["phase"]="done"; dump()
    return r

if __name__=="__main__":
    bench,arm,seed=sys.argv[1],sys.argv[2],int(sys.argv[3])
    tag=f"{bench}__{arm}__seed{seed}"
    r=run(bench,arm,seed,os.path.join(RES,"ckpt",tag+".json"))
    _atomic(os.path.join(RES,tag+".json"),r)
    rs=r.get("roi_summary") or {}
    print(f"[done] {bench} {arm} seed{seed} regret={r['final_regret']:.4f} "
          f"queries={len(r['queries'])} "
          f"acc={rs.get('accept_frac')} beta={rs.get('beta_sqrt')} "
          f"wall={r['_wall_s']/60:.1f}m",flush=True)
