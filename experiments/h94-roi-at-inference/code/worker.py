"""H94 worker: the ROI applied to the QUERY (DRO Sec 4.2), not to the teacher.
Requires src/policy/mf_dro.py to carry code/roi_at_inference.patch; the worker
REFUSES TO RUN without it rather than silently producing an unpatched run whose
arm labels would be lies. Trace/checkpoint machinery unchanged from h90 so the
reused NO-ROI and ROI-Q10 arms stay directly comparable."""
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
SPEC={"Borehole_8D":dict(n_hf=10,n_lf=20)}   # Borehole only, by design
# DIAGNOSTIC ONLY -- feeds roi_stats' min_dist_to_xstar / frac_within_0.2, and
# is never read by the algorithm (mf_dro.py uses _roi_x_star solely inside the
# `if roi_stats is not None:` recording block).
#
# Two defects in the auto-lookup this works around:
#   1. dro_runner._KNOWN_OPTIMAL_X holds only Hartmann_6D and the Ackley
#      variants, so Borehole and Currin recorded NaN throughout h84 -- the ROI
#      diagnostics are missing on the very benchmark the result rests on.
#   2. Its "Ackley_10D" entry is the OLD SF-only benchmark ([0.0]*10 on a
#      [-32.768, 32.768]^10 domain). The MF pair used here is [0,1]^10 with its
#      optimum at [0.5]*10 -- same benchmark_base_name, different benchmark.
#      Relying on the lookup would have recorded distances to the wrong point.
KNOWN_X={"Currin_2D":[0.21666666328646256,0.008707968518137932],
         "Hartmann_6D":[0.2017,0.15,0.4769,0.2753,0.3116,0.6573],
         "Borehole_8D":[0.15,100.0,95090.978,1110.0,116.0,700.0,1120.0,12045.0],
         "Ackley_10D":[0.5]*10}
# arm -> ROI configuration. ROI-OFF reproduces h83 exactly (bit-identity gated).
# Both arms are RUN at these seeds. Nothing is reused from another seed set --
# reusing across seed sets is precisely what this experiment exists to avoid.
# q=0.10 is a module constant so no other ROI setting can be run on seeds 47-51.
ARMS={
 # C and D share an IDENTICAL training-time ROI (q=0.10, the h84/h90 setting).
 # The ONLY difference is what happens to the emitted point at inference, which
 # is what makes D a control for the snapping operation rather than for the ROI.
 "ROI-PROJECT":  dict(use_roi=True, roi_beta_mode='quantile', roi_target_accept=0.10,
                      roi_inference_mode='project'),
 # Snaps to an UNFILTERED pool, and snaps ALWAYS (it has no feasibility test, so
 # it cannot snap conditionally). It therefore snaps at least as often as
 # ROI-PROJECT does and OVER-estimates the snapping effect -- deliberately
 # biased AGAINST the hypothesis that the ROI is what matters.
 "SNAP-CONTROL": dict(use_roi=True, roi_beta_mode='quantile', roi_target_accept=0.10,
                      roi_inference_mode='snap_control'),
}
RES=os.path.join(os.path.dirname(__file__),"..","results")

def _require_patch():
    """Refuse to run if src/ lacks the H94 patch. An unpatched run would ignore
    roi_inference_mode entirely and write a file labelled ROI-PROJECT that is
    actually a plain ROI-Q10 run -- indistinguishable after the fact."""
    from src.policy.mf_dro import DirectMFRegretOptimization as _D
    if not hasattr(_D, "_roi_snap"):
        sys.exit("REFUSING TO RUN: src/policy/mf_dro.py lacks _roi_snap. "
                 "Apply experiments/h94-roi-at-inference/code/roi_at_inference.patch first.")

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
        cfg=_build_mf_dro_config("h90",bench,arm,seed,bo_iterations=4000,num_epochs=10,
            minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=BUDGET,
            initial_hf=sp["n_hf"],initial_lf=sp["n_lf"],dkl_threshold=9999,
            bes_delta=0.0,rollout_length=8)
        cfg.seed=seed; cfg.rollout_reward="mes_entropy"
        cfg.use_candidate_scoring=False              # regression head (the method)
        for k,v in ARMS[arm].items(): setattr(cfg,k,v)
        cfg.known_optimal_x=KNOWN_X[bench]   # diagnostic only; see KNOWN_X note
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
        # P5 evidence: did the manipulation actually intervene? Records the
        # fraction of real queries that were moved, not an inference from it.
        il=getattr(mf,"roi_inf_log",None) or []
        if il:
            _sn=[bool(d.get("snapped")) for d in il]
            _ds=[float(d.get("dist",0.0)) for d in il if d.get("snapped")]
            r["roi_inference"]=dict(
                n=len(il), snapped_frac=float(np.mean(_sn)) if _sn else None,
                mean_snap_dist=(float(np.mean(_ds)) if _ds else None),
                mean_accept_frac=float(np.mean([d.get("accept_frac",1.0) for d in il])),
                n_empty_roi=int(sum(1 for d in il if d.get("empty_roi"))),
                log=il)
        # D2: makes L_loc interpretable across arms (see findings.md).
        av=getattr(mf,"actions_x_var_per_iter",None)
        if av: r["actions_x_var_per_iter"]=[float(x) for x in av]
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
    _require_patch()
    bench,arm,seed=sys.argv[1],sys.argv[2],int(sys.argv[3])
    tag=f"{bench}__{arm}__seed{seed}"
    r=run(bench,arm,seed,os.path.join(RES,"ckpt",tag+".json"))
    _atomic(os.path.join(RES,tag+".json"),r)
    rs=r.get("roi_summary") or {}
    print(f"[done] {bench} {arm} seed{seed} regret={r['final_regret']:.4f} "
          f"queries={len(r['queries'])} "
          f"acc={rs.get('accept_frac')} beta={rs.get('beta_sqrt')} "
          f"snapped={(r.get('roi_inference') or {}).get('snapped_frac')} "
          f"wall={r['_wall_s']/60:.1f}m",flush=True)
