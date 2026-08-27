"""H84 analysis. PRIMARY metric is mean HF query quality, the direct measure of
budget waste; regret is SECONDARY and reported separately because a better mean
query does not imply a better final best-point."""
import os, sys, json, glob
import numpy as np
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
from benchmarks import get_benchmark
H84=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results")
H83=os.path.join(REPO,"experiments","h83-main-comparison","results")
BENCH=("Hartmann_6D","Borehole_8D"); SEEDS=(42,43,44,45,46)
ARMS=("ROI-OFF","ROI-FIX2","ROI-Q10","ROI-ANN")

def path(bench,arm,seed):
    p=os.path.join(H84,f"{bench}__{arm}__seed{seed}.json")
    if os.path.exists(p): return p,"h84"
    if arm=="ROI-OFF":                      # arm A reused from h83 under the gate
        p=os.path.join(H83,f"{bench}__MF-DRO__seed{seed}.json")
        if os.path.exists(p): return p,"h83"
    return None,None

def score(bench,p):
    r=json.load(open(p))
    yopt=-float(get_benchmark(f"{bench}_HF")["known_optimal_value"])
    opt=abs(float(get_benchmark(f"{bench}_HF")["known_optimal_value"]))
    q=r["queries"]
    init=[e["y"] for e in q if e.get("is_init") and e["fid"]==1]
    if not init: return None
    b0=max(init); den=yopt-b0
    post=[e for e in q if e["fid"]==1 and not e.get("is_init")]
    if not post: return None
    sc=(np.array([e["y"] for e in post])-b0)/den
    return dict(mean=float(sc.mean()), best=float(sc.max()),
                frac_neg=float(np.mean(sc<0)), n_hf=len(post),
                rel=100.0*float(r["final_regret"])/opt,
                roi=r.get("roi_summary") or {})

def reproduction_control():
    """The h83 reuse is VOID unless the live re-runs reproduce it exactly."""
    print("REPRODUCTION CONTROL (ROI-OFF re-run vs h83's MF-DRO, seeds 42/43)")
    ok=True; ran=0
    for b in BENCH:
        for s in (42,43):
            a=os.path.join(H84,f"{b}__ROI-OFF__seed{s}.json")
            c=os.path.join(H83,f"{b}__MF-DRO__seed{s}.json")
            if not (os.path.exists(a) and os.path.exists(c)):
                print(f"  {b} s{s}: NOT YET RUN -- reuse unverified"); ok=False; continue
            ran+=1
            ra,rc=json.load(open(a)),json.load(open(c))
            d=abs(float(ra["final_regret"])-float(rc["final_regret"]))
            xa=np.array([e["x"] for e in ra["queries"]]); xc=np.array([e["x"] for e in rc["queries"]])
            dx=(np.abs(xa-xc).max() if xa.shape==xc.shape else float("inf"))
            good=(d<1e-9 and dx==0.0)
            print(f"  {b} s{s}: |dregret|={d:.3e}  max|dx|={dx:.3e}  {'OK' if good else 'MISMATCH'}")
            ok&=good
    print(f"  -> {'PASS: h83 reuse valid' if (ok and ran) else 'INCOMPLETE/FAIL: do not rely on reused arm A'}\n")
    return ok and ran

if __name__=="__main__":
    reproduction_control()
    miss=[]
    for b in BENCH:
        print(f"=== {b} ===")
        print(f"  {'arm':10s}{'mean q-score':>13s}{'best':>7s}{'<init':>8s}{'HFq':>6s}"
              f"{'rel.reg':>9s}{'ROI acc':>9s}{'beta_t':>8s}   per-seed mean")
        base=None
        for a in ARMS:
            rows=[]
            for s in SEEDS:
                p,src=path(b,a,s)
                if p is None: miss.append(f"{b}/{a}/s{s}"); continue
                v=score(b,p)
                if v: rows.append(v)
            if not rows: print(f"  {a:10s}{'--':>13s}"); continue
            m=np.array([r["mean"] for r in rows]); rel=np.array([r["rel"] for r in rows])
            acc=[r["roi"].get("accept_frac") for r in rows if r["roi"].get("accept_frac") is not None]
            bet=[r["roi"].get("beta_sqrt") for r in rows if r["roi"].get("beta_sqrt") is not None]
            if a=="ROI-OFF": base=m
            win=""
            if base is not None and a!="ROI-OFF" and len(m)==len(base):
                win=f"  wins {int((m>base).sum())}/{len(m)}"
            print(f"  {a:10s}{m.mean():>13.3f}{np.mean([r['best'] for r in rows]):>7.3f}"
                  f"{100*np.mean([r['frac_neg'] for r in rows]):>7.1f}%"
                  f"{np.mean([r['n_hf'] for r in rows]):>6.0f}{rel.mean():>8.2f}%"
                  f"{(np.mean(acc) if acc else float('nan')):>9.1%}"
                  f"{(np.mean(bet) if bet else float('nan')):>8.2f}   "
                  + " ".join(f"{v:.2f}" for v in m) + win)
        print()
    if miss: print(f"INCOMPLETE -- {len(miss)} runs missing: {miss[:10]}{' ...' if len(miss)>10 else ''}")
    print("\nMF-MES reference (h83): Hartmann mean q-score 0.747, Borehole 0.669")
    print("PRE-REGISTERED BARS in protocol.md -- evaluated in analysis.md, not here.")
