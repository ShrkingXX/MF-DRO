"""H57 worker. Records the REAL queried (x, y, fidelity, cost) at every
iteration for all four methods, and checkpoints that trace to disk every
CKPT_S seconds so a run in flight can be inspected and a killed run is not
total loss.

Trace source per method -- deliberately NOT a uniform objective wrapper:
  MF-DRO   mf.iteration_log (live). NOT a wrapper on f_hf: the run loop also
           calls f_hf once per iteration to score argmax mu_H for the
           inference-regret metric, which is not a real query and would
           pollute the trace.
  MF-MES   wraps the f_hf/f_lf handed to run_mf_mes, which calls each exactly
           once per iteration and only for real queries (verified at
           mf_mes_takeno.py lines 56/60).
  baselines  wraps opt._add_obs, through which every observation passes.
"""
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
ARM=None
SPEC={"Currin_2D":dict(n_hf=5,n_lf=15),
      "Hartmann_6D":dict(n_hf=6,n_lf=45),
      "Borehole_8D":dict(n_hf=10,n_lf=20)}
RES=os.path.join(os.path.dirname(__file__),"..","results")

def _code_state():
    """Record the exact code each job ran against. 36 jobs launched across a
    moving src/ are uninterpretable later; the hash makes 'was this run on the
    same code as that one' checkable rather than remembered."""
    import subprocess
    def g(*a):
        try: return subprocess.run(["git",*a],cwd=REPO,capture_output=True,
                                   text=True,timeout=20).stdout.strip()
        except Exception: return ""
    dirty=g("status","--porcelain","src","dro_runner.py","benchmarks.py")
    return dict(commit=g("rev-parse","HEAD"),
                dirty=bool(dirty), dirty_files=dirty.splitlines()[:20])
CODE=_code_state()

def _atomic(path,obj):
    tmp=path+".tmp"; json.dump(obj,open(tmp,"w"),default=float); os.replace(tmp,path)

def run(bench, method, seed, ckpt_path):
    sp=SPEC[bench]; t0=time.time()
    hf=get_benchmark(f"{bench}_HF"); lf=get_benchmark(f"{bench}_LF")
    bounds=torch.tensor([hf["domain_min"],hf["domain_max"]],dtype=torch.float64)
    c_H=float(hf["cost"]); c_L=float(lf["cost"])
    torch.manual_seed(seed); np.random.seed(seed)
    trace=[]; state={"phase":"init"}

    n_init=sp["n_hf"]+sp["n_lf"]
    acc={"c":0.0}
    def rec(x,y,fid,is_init=None):
        x=np.asarray(x,dtype=float).reshape(-1)
        acc["c"]+= c_H if fid else c_L
        trace.append(dict(i=len(trace),fid=int(fid),x=x.tolist(),y=float(y),
                          cost_cum=acc["c"],
                          is_init=bool(len(trace)<n_init if is_init is None else is_init)))
    def dump():
        _atomic(ckpt_path,dict(bench=bench,method=method,seed=seed,code=CODE,
                               phase=state["phase"],n_queries=len(trace),
                               elapsed_s=round(time.time()-t0,1),queries=trace[:]))
    stop=threading.Event()
    threading.Thread(target=lambda:[dump() for _ in iter(lambda:stop.wait(CKPT_S),True)],
                     daemon=True).start()
    try:
        if method in ("MF-DRO","MF-MES"):
            from src.policy.mf_dro import DirectMFRegretOptimization
            cfg=_build_mf_dro_config("h57",bench,method,seed,bo_iterations=4000,num_epochs=10,
                minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=BUDGET,
                initial_hf=sp["n_hf"],initial_lf=sp["n_lf"],dkl_threshold=9999,
                bes_delta=0.0,rollout_length=8)
            cfg.seed=seed; cfg.rollout_reward="mes_entropy"
            cfg.use_candidate_scoring=False              # regression head
            mf=DirectMFRegretOptimization(cfg,hf["make_objective"](),lf["make_objective"](),bounds)
            if method=="MF-DRO":
                state["phase"]="running"
                if ARM=="FLOOR":
                    # Mirror mf_dro.py:1373 -- which is inside
                    # simulate_mf_trajectory and therefore applies to SIMULATED
                    # rollouts only -- at REAL query time. Applied here by
                    # wrapping propose_mf rather than editing mf_dro.py, which is
                    # frozen for h57. Counts from iteration_log, i.e. the
                    # fidelities actually EXECUTED (real_hf_warmup can override
                    # what propose_mf returned), not the proposals.
                    _op=mf.dt.propose_mf
                    def _floored(st,rtg,btg,**kw):
                        x,ell=_op(st,rtg,btg,**kw)
                        lg=mf.iteration_log; t=len(lg)
                        if t>0 and int(ell)==0:
                            nhf=sum(1 for e in lg if int(e["ell_t"])==1)
                            if nhf/t < 0.25:
                                ell=1
                        return x,ell
                    mf.dt.propose_mf=_floored
                # mirror iteration_log into the trace as it fills
                def mirror():
                    while not stop.wait(2.0):
                        for e in mf.iteration_log[len(trace):]:
                            trace.append(dict(i=len(trace),fid=int(e["ell_t"]),
                                x=list(np.asarray(e["x_t"],dtype=float).reshape(-1)),
                                y=float(e["y_t"]),cost_cum=float(e["cumulative_cost"]),
                                regret=float(e["regret"]),
                                inference_regret=float(e.get("inference_regret",float("nan")))))
                threading.Thread(target=mirror,daemon=True).start()
                r=mf.run()
                trace.clear(); acc["c"]=0.0
                # initial design first, so MF-DRO's trace matches the other
                # methods (whose _add_obs / objective wrappers see the init too)
                for x,y in zip(list(mf.data_hf_x)[:sp["n_hf"]],list(mf.data_hf_y)[:sp["n_hf"]]):
                    rec(x.cpu().numpy(),y,1,is_init=True)
                for x,y in zip(list(mf.data_lf_x)[:sp["n_lf"]],list(mf.data_lf_y)[:sp["n_lf"]]):
                    rec(x.cpu().numpy(),y,0,is_init=True)
                for e in mf.iteration_log:
                    trace.append(dict(i=len(trace),fid=int(e["ell_t"]),
                        x=list(np.asarray(e["x_t"],dtype=float).reshape(-1)),
                        y=float(e["y_t"]),cost_cum=float(e["cumulative_cost"]),
                        regret=float(e["regret"]),
                        inference_regret=float(e.get("inference_regret",float("nan")))))
            else:
                from src.baselines.mf_mes_takeno import run_mf_mes
                mf._sample_initial_points(); state["phase"]="running"
                # The initial design is drawn through mf_dro's internal path, NOT
                # through the wrapped objectives, so the wrapper only ever sees
                # OPTIMIZATION queries. rec()'s positional len(trace)<n_init rule
                # is therefore wrong here -- it mislabelled the first n_init
                # optimization queries as init and lost the real init entirely.
                # Prepend the real init, then pin is_init=False on everything the
                # wrapper records.
                for x,y in zip(list(mf.data_hf_x)[:sp["n_hf"]],list(mf.data_hf_y)[:sp["n_hf"]]):
                    rec(x.cpu().numpy(),y,1,is_init=True)
                for x,y in zip(list(mf.data_lf_x)[:sp["n_lf"]],list(mf.data_lf_y)[:sp["n_lf"]]):
                    rec(x.cpu().numpy(),y,0,is_init=True)
                def wrap(fn,fid):
                    def g(X):
                        out=fn(X)
                        rec(np.asarray(X,dtype=float).reshape(-1),
                            float(np.asarray(out,dtype=float).reshape(-1)[0]),fid,
                            is_init=False)
                        return out
                    return g
                res=run_mf_mes(f_hf=wrap(mf.f_hf,1),f_lf=wrap(mf.f_lf,0),bounds=bounds,
                    c_H=mf.c_H,c_L=mf.c_L,cost_budget=BUDGET,true_opt=-float(cfg.true_opt),
                    X_lf0=torch.stack(list(mf.data_lf_x)),Y_lf0=torch.tensor(list(mf.data_lf_y)),
                    X_hf0=torch.stack(list(mf.data_hf_x)),Y_hf0=torch.tensor(list(mf.data_hf_y)),
                    seed=seed,verbose=False,ko_kwargs=dict(dkl_threshold=9999))
                r={k:v for k,v in res.items() if k not in ("acq_info","X_hf","Y_hf")}
                r["hf_regret_curve"]=res["regret_curve"]; r["fidelity_trace"]=res["fidelity_history"]
        else:
            from src.baselines.mf_baselines import (MultiFidelityBenchmark,
                                                    MFMIGreedyOptimizer, MFGPUCBOptimizer)
            b=MultiFidelityBenchmark(bench)
            common=dict(n_initial_hf=sp["n_hf"],n_initial_lf=sp["n_lf"],seed=seed,
                        cost_budget=BUDGET)
            opt=(MFMIGreedyOptimizer(b,**common) if method=="MF-MI-Greedy"
                 else MFGPUCBOptimizer(b,**common))
            _orig=opt._add_obs
            def _add(x,y,fidelity):
                rec(np.asarray(x if not torch.is_tensor(x) else x.cpu().numpy(),
                               dtype=float).reshape(-1), float(y), 1 if fidelity=='H' else 0)
                return _orig(x,y,fidelity)
            opt._add_obs=_add; state["phase"]="running"
            r=opt.run(bo_iterations=4000)
            r.setdefault("hf_regret_curve",r.get("regret_curve"))
    finally:
        stop.set()
    c=r.get("hf_regret_curve") or []
    r["queries"]=trace
    r["_meta"]=dict(bench=bench,method=method,seed=seed,budget=BUDGET,c_H=c_H,c_L=c_L,**sp)
    r["_code"]=CODE
    r["_arm"]=ARM
    r["_wall_s"]=round(time.time()-t0,1)
    r["final_regret"]=float(c[-1]) if c else float("nan")
    r["n_improvements"]=sum(1 for i in range(1,len(c)) if c[i]<c[i-1]-1e-9)
    state["phase"]="done"; dump()
    return r

if __name__=="__main__":
    bench,ARM,seed=sys.argv[1],sys.argv[2],int(sys.argv[3])
    method="MF-DRO"
    tag=f"{bench}__{ARM}__seed{seed}"
    r=run(bench,method,seed,os.path.join(RES,"ckpt",tag+".json"))
    _atomic(os.path.join(RES,tag+".json"),r)
    c=r["hf_regret_curve"]
    print(f"[done] {bench} {method} seed{seed} regret={r['final_regret']:.4f} "
          f"iters={len(c)} queries={len(r['queries'])} improv={r['n_improvements']} "
          f"wall={r['_wall_s']/60:.1f}m",flush=True)
