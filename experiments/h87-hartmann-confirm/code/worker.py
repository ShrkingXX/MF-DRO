"""H87 worker: clean confirmation of the Hartmann flip at fresh seeds 47-51.

H57 header follows.

H57 worker. Records the REAL queried (x, y, fidelity, cost) at every
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
SPEC={"Hartmann_6D":dict(n_hf=6,n_lf=45)}   # Hartmann only, by design
# ROI configuration is FIXED IN ADVANCE at q=0.10 and applies to MF-DRO only.
# The protocol forbids running any other ROI setting on these seeds: doing so
# would recreate the selection problem this experiment exists to remove.
ROI_CFG=dict(use_roi=True, roi_beta_mode='quantile', roi_target_accept=0.10)
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
            cfg=_build_mf_dro_config("h87",bench,method,seed,bo_iterations=4000,num_epochs=10,
                minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=BUDGET,
                initial_hf=sp["n_hf"],initial_lf=sp["n_lf"],dkl_threshold=9999,
                bes_delta=0.0,rollout_length=8)
            cfg.seed=seed; cfg.rollout_reward="mes_entropy"
            cfg.use_candidate_scoring=False              # regression head
            if method=="MF-DRO":
                for k,v in ROI_CFG.items(): setattr(cfg,k,v)
                cfg.known_optimal_x=[0.2017,0.15,0.4769,0.2753,0.3116,0.6573]  # diagnostic only
            mf=DirectMFRegretOptimization(cfg,hf["make_objective"](),lf["make_objective"](),bounds)
            if method=="MF-DRO":
                state["phase"]="running"
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
        elif method=="SF-DRO":
            # Single fidelity: every query costs c_H, so the post-init budget
            # buys exactly BUDGET//c_H optimization queries. Path copied from
            # h73/h59's worker, including its three hard-won gotchas:
            #   - rtg_schema must be one of fixed/dynamic/floored/quantile/
            #     joint/entropy_joint
            #   - entry point is run_optimization(), results in
            #     iteration_log_history
            #   - checkpoint.setup_dirs(exp_name) must run before the first log
            #     write or it crashes
            from checkpoint import setup_dirs
            from dro_runner import _build_dro_config
            from src.policy.dro import DirectRegretOptimization
            n_iter=int(BUDGET//c_H)
            exp=f"h83_{bench}_{seed}"; setup_dirs(exp)
            cfg=_build_dro_config(exp,bench,"sfdro",seed,
                use_mes_reward=False, rtg_schema="fixed", alpha_floor=0.5,
                alpha_inference=None, lambda_rtg=1.0, rtg_warmup=3, benchmark_spec=hf,
                gp_num_models=10, rollouts_per_iter=200, rollout_length=8,
                bo_iterations=n_iter, initial_points=sp["n_hf"],
                dt_hidden=64, dt_layers=2, dt_heads=4, dt_lr=1e-3,
                gp_kernel="rbf", gp_ard=True, verbose=False)
            cfg.seed=seed
            dro=DirectRegretOptimization(cfg, hf["make_objective"]())
            state["phase"]="running"; dro.run_optimization()
            # iteration_log_history has NO per-iteration x/y (keys are regret,
            # best, mean_reward, zero_frac, rtg_target, batch_max_rtg,
            # running_max_rtg). Reading d["x"]/d["y"] silently produced EMPTY
            # traces on three h59 Currin runs. The queries live on the optimizer.
            qx=dro.data_x.detach().cpu().numpy(); qy=np.asarray(dro.data_y.detach().cpu().numpy()).reshape(-1)
            for j,(x,y) in enumerate(zip(qx,qy)):
                rec(x,float(y),1,is_init=(j<sp["n_hf"]))
            curve=[float(d["regret"]) for d in dro.iteration_log_history]
            r={"hf_regret_curve":curve,"regret_curve":curve,
               "fidelity_trace":[1]*max(len(curve),0),"n_iter_cap":n_iter}
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
    r["_wall_s"]=round(time.time()-t0,1)
    r["final_regret"]=float(c[-1]) if c else float("nan")
    r["n_improvements"]=sum(1 for i in range(1,len(c)) if c[i]<c[i-1]-1e-9)
    state["phase"]="done"; dump()
    return r

if __name__=="__main__":
    bench,method,seed=sys.argv[1],sys.argv[2],int(sys.argv[3])
    tag=f"{bench}__{method}__seed{seed}"
    r=run(bench,method,seed,os.path.join(RES,"ckpt",tag+".json"))
    _atomic(os.path.join(RES,tag+".json"),r)
    c=r["hf_regret_curve"]
    print(f"[done] {bench} {method} seed{seed} regret={r['final_regret']:.4f} "
          f"iters={len(c)} queries={len(r['queries'])} improv={r['n_improvements']} "
          f"wall={r['_wall_s']/60:.1f}m",flush=True)
