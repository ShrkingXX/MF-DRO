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
        # Collect per-seed values first so every arm-vs-control comparison can be
        # made PAIRED on the seeds both arms actually have. Comparing arm means
        # over different seed subsets is not a comparison: on Hartmann, ROI-OFF's
        # mean is dominated by seed 44 (q-score -0.94), so an arm holding only
        # seed 43 appears to beat it by +0.45 while being tied on the seed they
        # share.
        per={}
        for a in ARMS:
            per[a]={}
            for s in SEEDS:
                p_,src=path(b,a,s)
                if p_ is None:
                    if a!="ROI-OFF": miss.append(f"{b}/{a}/s{s}")
                    continue
                v=score(b,p_)
                if v: per[a][s]=v
        print(f"  {'arm':10s}{'n':>3s}{'q-score':>9s}{'rel.reg':>9s}{'<init':>7s}{'HFq':>5s}"
              f"{'acc':>7s}{'beta':>7s}   per-seed q-score (42..46)")
        for a in ARMS:
            r=per[a]
            if not r: print(f"  {a:10s}{0:>3d}{'--':>9s}"); continue
            m=np.array([r[s]["mean"] for s in sorted(r)])
            acc=[r[s]["roi"].get("accept_frac") for s in sorted(r) if r[s]["roi"].get("accept_frac") is not None]
            bet=[r[s]["roi"].get("beta_sqrt") for s in sorted(r) if r[s]["roi"].get("beta_sqrt") is not None]
            ps=" ".join((f"{r[s]['mean']:6.2f}" if s in r else "     .") for s in SEEDS)
            print(f"  {a:10s}{len(r):>3d}{m.mean():>9.3f}"
                  f"{np.mean([r[s]['rel'] for s in sorted(r)]):>8.2f}%"
                  f"{100*np.mean([r[s]['frac_neg'] for s in sorted(r)]):>6.1f}%"
                  f"{np.mean([r[s]['n_hf'] for s in sorted(r)]):>5.0f}"
                  f"{(np.mean(acc) if acc else float('nan')):>7.1%}"
                  f"{(np.mean(bet) if bet else float('nan')):>7.2f}   {ps}")
        # PAIRED comparisons against the control, common seeds only
        ctl=per.get("ROI-OFF",{})
        print(f"\n  PAIRED vs ROI-OFF (common seeds only):")
        for a in ARMS:
            if a=="ROI-OFF": continue
            com=sorted(set(per.get(a,{})) & set(ctl))
            if not com: print(f"    {a:10s} no common seeds yet"); continue
            dq=np.array([per[a][s]["mean"]-ctl[s]["mean"] for s in com])
            dr=np.array([per[a][s]["rel"]-ctl[s]["rel"] for s in com])
            print(f"    {a:10s} n={len(com)}  d(q-score)={dq.mean():+.3f} "
                  f"(wins {int((dq>0).sum())}/{len(com)})   "
                  f"d(rel.regret)={dr.mean():+.2f}pts (better {int((dr<0).sum())}/{len(com)})"
                  f"   seeds {com}")
        print()
    if miss: print(f"INCOMPLETE -- {len(miss)} runs missing: {miss[:10]}{' ...' if len(miss)>10 else ''}")
    print("\nMF-MES reference (h83): Hartmann q-score 0.747, Borehole 0.669")
    print("PRE-REGISTERED BARS in protocol.md -- evaluated in analysis.md, not here.")
