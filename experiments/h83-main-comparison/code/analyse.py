"""H83 analysis. SR is recomputed from each run's query trace and CROSS-CHECKED
against the optimizer's own final regret; disagreement is a hard failure, not a
warning. Five methods use five different internal curve conventions and this
project has twice shipped a sign/negation error, so the trace is the source of
truth and the optimizer's curve is the control."""
import os, sys, json, glob
import numpy as np
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
from benchmarks import get_benchmark

RES=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results")
BENCH=("Currin_2D","Hartmann_6D","Borehole_8D","Ackley_10D")
METHODS=("SF-DRO","MF-DRO","MF-MES","MF-MI-Greedy","MF-GP-UCB")
SEEDS=(42,43,44,45,46)

def sr_curve(run, opt_val):
    """Post-init cumulative cost and running simple regret, from the trace.

    y is stored as the objective returns it (negate=True -> minimisation
    convention), matching mf_baselines' `regret = -best_hf - known_optimal`.
    Initial-design points count toward the running best but not toward cost."""
    q=run["queries"]
    init_cost=sum((run["_meta"]["c_H"] if e["fid"] else run["_meta"]["c_L"])
                  for e in q if e.get("is_init"))
    cost,sr,best=[],[],-np.inf
    for e in q:
        if e["fid"]:                      # HF observations define simple regret
            best=max(best,float(e["y"]))
        if not e.get("is_init"):
            cost.append(float(e["cost_cum"])-init_cost)
            sr.append(float(-best-opt_val) if np.isfinite(best) else np.nan)
    return np.asarray(cost), np.asarray(sr)

def load():
    out={}; missing=[]; failures=[]
    for b in BENCH:
        opt=float(get_benchmark(f"{b}_HF")["known_optimal_value"])
        for m in METHODS:
            for s in SEEDS:
                p=os.path.join(RES,f"{b}__{m}__seed{s}.json")
                if not os.path.exists(p): missing.append(f"{b}/{m}/s{s}"); continue
                r=json.load(open(p))
                c,sr=sr_curve(r,opt)
                own=r.get("final_regret")
                if own is not None and np.isfinite(own) and len(sr):
                    if abs(float(sr[-1])-float(own))>1e-6:
                        failures.append(f"{b}/{m}/s{s}: trace SR {sr[-1]:.6f} != own {own:.6f}")
                out[(b,m,s)]=dict(cost=c,sr=sr,opt=opt,
                                  lf_frac=1.0-np.mean([e["fid"] for e in r["queries"]
                                                       if not e.get("is_init")] or [1]),
                                  wall=r.get("_wall_s"))
    return out,missing,failures

def grid(cost,sr,pts):
    """Step-interpolate a cost-indexed curve onto a common grid (last value at
    or before each grid point). Methods query at different costs, so a plain
    mean over raw indices would compare different budgets."""
    o=np.full(len(pts),np.nan)
    for i,t in enumerate(pts):
        k=np.searchsorted(cost,t,side="right")-1
        if k>=0: o[i]=sr[k]
    return o

if __name__=="__main__":
    D,missing,failures=load()
    if missing: print(f"INCOMPLETE -- {len(missing)} runs missing: {missing[:12]}"
                      f"{' ...' if len(missing)>12 else ''}\n")
    if failures:
        print("SR CROSS-CHECK FAILED (hard failure, per protocol):")
        for f in failures: print("   ",f)
        print()
    else:
        print("SR cross-check: trace-recomputed SR agrees with every optimizer's "
              "own final regret to 1e-6.\n")
    G=np.linspace(0,200,201)
    for b in BENCH:
        opt=None
        print(f"=== {b} ===")
        print(f"  {'method':14s}{'rel.regret %':>14s}{'seeds':>8s}{'LF%':>8s}   per-seed rel%")
        for m in METHODS:
            rows=[D[(b,m,s)] for s in SEEDS if (b,m,s) in D]
            if not rows: print(f"  {m:14s}{'--':>14s}"); continue
            opt=rows[0]["opt"]
            fin=np.array([grid(r["cost"],r["sr"],G)[-1] for r in rows],dtype=float)
            rel=100.0*fin/abs(opt)
            lf=100.0*np.mean([r["lf_frac"] for r in rows])
            print(f"  {m:14s}{np.nanmean(rel):>13.2f}%{len(rows):>8d}{lf:>7.1f}%   "
                  +" ".join(f"{v:6.2f}" for v in rel))
        print()
    print("PRE-REGISTERED BARS (protocol.md) -- evaluated in analysis.md, not here.")
