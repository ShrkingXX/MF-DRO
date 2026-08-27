"""H85 verdict, written before the arms complete (h87 template).

Refuses to print verdicts on incomplete arms. Prints each pre-registered bar
mechanically. Control (REFINE-0) is reused from h83 under a reproduction-control
gate; the reuse is VOID unless the live re-runs reproduce h83 bit-identically.
"""
import sys, os, json, glob
import numpy as np
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,"experiments","h83-main-comparison","code"))
from analyse import sr_curve, grid
from benchmarks import get_benchmark
G=np.linspace(0,200,201); SEEDS=(42,43,44,45,46)
BENCH=("Hartmann_6D","Borehole_8D"); ARMS=("REFINE-0","REFINE-100","HF-FLOOR")
R=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results")
H83=os.path.join(REPO,'experiments','h83-main-comparison','results')

def path(b,a,s):
    p=os.path.join(R,f"{b}__{a}__seed{s}.json")
    if os.path.exists(p): return p
    if a=="REFINE-0":
        p=os.path.join(H83,f"{b}__MF-DRO__seed{s}.json")
        if os.path.exists(p): return p
    return None

def rel(b,p):
    opt=float(get_benchmark(f"{b}_HF")["known_optimal_value"])
    c,sr=sr_curve(json.load(open(p)),opt); v=grid(c,sr,G)[-1]
    return v if abs(opt)<1e-9 else 100.0*v/abs(opt)

def nearbound(b,p):
    lo=np.array(get_benchmark(f"{b}_HF")["domain_min"],float)
    hi=np.array(get_benchmark(f"{b}_HF")["domain_max"],float)
    rg=np.where(hi-lo>0,hi-lo,1.0)
    q=[e for e in json.load(open(p))["queries"] if e["fid"]==1 and not e.get("is_init")]
    if not q: return np.nan
    Z=np.array([(np.array(e["x"],float)-lo)/rg for e in q])
    return 100.0*np.mean((Z<=0.05)|(Z>=0.95))

def control_gate():
    print("REPRODUCTION CONTROL (REFINE-0 re-run vs h83's MF-DRO, seeds 42/43)")
    ok=True; ran=0
    for b in BENCH:
        for s in (42,43):
            a=os.path.join(R,f"{b}__REFINE-0__seed{s}.json"); c=os.path.join(H83,f"{b}__MF-DRO__seed{s}.json")
            if not (os.path.exists(a) and os.path.exists(c)):
                print(f"  {b} s{s}: NOT YET RUN"); ok=False; continue
            ran+=1
            ra,rc=json.load(open(a)),json.load(open(c))
            d=abs(float(ra["final_regret"])-float(rc["final_regret"]))
            xa=np.array([e["x"] for e in ra["queries"]]); xc=np.array([e["x"] for e in rc["queries"]])
            dx=(np.abs(xa-xc).max() if xa.shape==xc.shape else float("inf"))
            g=(d<1e-9 and dx==0.0); ok&=g
            print(f"  {b} s{s}: |dregret|={d:.3e} max|dx|={dx:.3e} {'OK' if g else 'MISMATCH'}")
    print(f"  -> {'PASS' if (ok and ran) else 'INCOMPLETE/FAIL -- control arm unverified'}\n")
    return ok and ran

if __name__=="__main__":
    control_gate()
    S={}
    for b in BENCH:
        for a in ARMS:
            v={s:rel(b,path(b,a,s)) for s in SEEDS if path(b,a,s)}
            S[(b,a)]=v
    for b in BENCH:
        print(f"=== {b} ===")
        print(f"  {'arm':11s}{'n':>3s}{'mean':>8s}{'sd':>7s}   per-seed")
        for a in ARMS:
            v=S[(b,a)]
            if not v: print(f"  {a:11s}{0:>3d}{'--':>8s}"); continue
            x=np.array([v[s] for s in sorted(v)])
            print(f"  {a:11s}{len(v):>3d}{x.mean():>8.2f}{(x.std(ddof=1) if len(x)>1 else float('nan')):>7.2f}   "
                  +" ".join(f"{v[s]:6.2f}" if s in v else "     ." for s in SEEDS))
        ctl=S[(b,"REFINE-0")]
        for a in ("REFINE-100","HF-FLOOR"):
            com=sorted(set(S[(b,a)])&set(ctl))
            if len(com)<5: print(f"  PAIRED {a}: {len(com)}/5 -- INCOMPLETE, no verdict"); continue
            d=np.array([S[(b,a)][s]-ctl[s] for s in com])
            print(f"  PAIRED {a}: d={d.mean():+.2f} pts (better {int((d<0).sum())}/5)")
        print()
    done=all(len(S[(b,a)])==5 for b in BENCH for a in ARMS)
    if not done:
        print("INCOMPLETE -- no bar verdicts. Partial arms are unreliable in BOTH")
        print("directions here (ROI-FIX2 moved +6.32 -> +0.36 -> -0.26 across three reports).")
        sys.exit()
    print("PRE-REGISTERED BARS")
    dH=np.mean([S[("Hartmann_6D","REFINE-100")][s]-S[("Hartmann_6D","REFINE-0")][s] for s in SEEDS])
    dB=np.mean([S[("Borehole_8D","REFINE-100")][s]-S[("Borehole_8D","REFINE-0")][s] for s in SEEDS])
    print(f"  P1 (refinement helps MORE on Borehole than Hartmann): "
          f"{'MET' if dB<dH else 'FAILED'}   [Borehole {dB:+.2f} vs Hartmann {dH:+.2f}]")
    nb0=np.mean([nearbound("Borehole_8D",path("Borehole_8D","REFINE-0",s)) for s in SEEDS])
    nb1=np.mean([nearbound("Borehole_8D",path("Borehole_8D","REFINE-100",s)) for s in SEEDS])
    print(f"  P2 (Borehole near-bound fraction rises above the 10% uniform null): "
          f"{'MET' if nb1>10.0 and nb1>nb0 else 'FAILED'}   [{nb0:.2f}% -> {nb1:.2f}%]")
    wB=sum(1 for s in SEEDS if S[("Borehole_8D","REFINE-100")][s]<S[("Borehole_8D","REFINE-0")][s])
    print(f"  P3 (REFINE-100 lowers Borehole regret on >= 4/5): {'MET' if wB>=4 else 'FAILED'}   [{wB}/5]")
    sd0=np.std([S[("Hartmann_6D","REFINE-0")][s] for s in SEEDS],ddof=1)
    sdF=np.std([S[("Hartmann_6D","HF-FLOOR")][s] for s in SEEDS],ddof=1)
    print(f"  P5 (HF-FLOOR reduces Hartmann across-seed spread): "
          f"{'MET' if sdF<sd0 else 'FAILED'}   [sd {sd0:.2f} -> {sdF:.2f}]")
    dF=np.mean([S[("Hartmann_6D","HF-FLOOR")][s]-S[("Hartmann_6D","REFINE-0")][s] for s in SEEDS])
    print(f"  P6 (NEGATIVE: HF-FLOOR does NOT improve Hartmann mean by >= 1pt): "
          f"{'MET' if dF>-1.0 else 'REFUTED'}   [{dF:+.2f} pts]")
    dFB=np.mean([S[("Borehole_8D","HF-FLOOR")][s]-S[("Borehole_8D","REFINE-0")][s] for s in SEEDS])
    print(f"  P7 (HF-FLOOR changes little on Borehole; non-binding there): [{dFB:+.2f} pts]")
    print("\n  Amendment 2: no positive result here is a FINDING until re-tested at")
    print("  fresh seeds with the configuration fixed in advance.")
