"""H89 verdict. Bars implemented exactly as registered in protocol.md.

PROCESS MISS, disclosed: the protocol said this script would be committed BEFORE
the treatment arm finished. It was not -- the Hartmann HF-FLOOR arm completed
first, and seed 53's outcome (control and floor identical at 0.0272) had already
been observed. The bars below are transcribed from the protocol without
modification, but the h87-style guarantee ("verdict written blind") does not
hold for the floor arm. It does still hold for the Borehole refinement arm,
which is unfinished as of writing.
"""
import sys, os, json
import numpy as np
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,"experiments","h83-main-comparison","code"))
from analyse import sr_curve, grid
from benchmarks import get_benchmark
G=np.linspace(0,200,201); SEEDS=(52,53,54,55,56)
R=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results")

def rel(b,p):
    opt=float(get_benchmark(f"{b}_HF")["known_optimal_value"])
    c,sr=sr_curve(json.load(open(p)),opt); v=grid(c,sr,G)[-1]
    return v if abs(opt)<1e-9 else 100.0*v/abs(opt)

def streak(p):
    q=[e for e in json.load(open(p))["queries"] if not e.get("is_init")]
    b=cur=0
    for e in q: cur=0 if e["fid"]==1 else cur+1; b=max(b,cur)
    return b

def arm(b,a):
    out={}
    for s in SEEDS:
        p=os.path.join(R,f"{b}__{a}__seed{s}.json")
        if os.path.exists(p): out[s]=p
    return out

if __name__=="__main__":
    # ---- HF floor arm, Hartmann ----
    C=arm("Hartmann_6D","CONTROL"); F=arm("Hartmann_6D","HF-FLOOR")
    com=sorted(set(C)&set(F))
    print(f"=== HF FLOOR, Hartmann, fresh seeds ({len(com)}/5 paired) ===\n")
    if len(com)==5:
        c=np.array([rel("Hartmann_6D",C[s]) for s in com])
        f=np.array([rel("Hartmann_6D",F[s]) for s in com])
        st=np.array([streak(C[s]) for s in com])
        print(f"  {'seed':>5}{'control':>10}{'floor':>9}{'diff':>9}{'ctl LF streak':>15}")
        for i,s in enumerate(com):
            print(f"  {s:>5}{c[i]:>9.2f}%{f[i]:>8.2f}%{f[i]-c[i]:>+9.2f}{st[i]:>15d}")
        d=f-c
        print(f"\n  control mean {c.mean():.2f}% sd {c.std(ddof=1):.2f}"
              f"   floor mean {f.mean():.2f}% sd {f.std(ddof=1):.2f}")
        print(f"  paired mean {d.mean():+.2f} pts, better {int((d<0).sum())}/5\n")
        print("  BARS")
        p1 = f.std(ddof=1) < c.std(ddof=1)
        print(f"    P1 PRIMARY (floor reduces across-seed sd): {'MET' if p1 else 'FAILED'}"
              f"   [{c.std(ddof=1):.2f} -> {f.std(ddof=1):.2f}]")
        print(f"    P2 (paired mean negative): {'MET' if d.mean()<0 else 'FAILED'}   [{d.mean():+.2f}]")
        ncol=int((st>3).sum())
        p3 = ncol>=3
        print(f"    P3 MECHANISM (control collapses, LF streak >3, on >=3/5): "
              f"{'MET' if p3 else 'FAILED'}   [{ncol}/5, streaks {list(st)}]")
        if st.std()>0 and d.std()>0:
            r=float(np.corrcoef(st,-d)[0,1])
            print(f"    P4 HONESTY (gain concentrated in collapsed seeds): "
                  f"corr(streak, gain) = {r:+.2f}")
        else:
            print(f"    P4 HONESTY: undefined (no variation in streak or gain)")
        if not p1:
            print("\n    *** P1 FAILED -> withdraw the variance claim from findings.md and")
            print("        any report, as prominently as it was made. ***")
        if not p3:
            print("\n    *** P3 FAILED -> withdraw the MECHANISM claim even if P1/P2 pass.")
            print("        A working intervention with a wrong explanation is still a")
            print("        wrong explanation. ***")
    else:
        print("  INCOMPLETE -- no verdict.")
    # ---- refinement arm, Borehole ----
    C2=arm("Borehole_8D","CONTROL"); R2=arm("Borehole_8D","REFINE-100")
    com2=sorted(set(C2)&set(R2))
    print(f"\n=== TEACHER REFINEMENT, Borehole, fresh seeds ({len(com2)}/5 paired) ===")
    if len(com2)==5:
        c=np.array([rel("Borehole_8D",C2[s]) for s in com2])
        f=np.array([rel("Borehole_8D",R2[s]) for s in com2]); d=f-c
        for i,s in enumerate(com2): print(f"  {s:>5}{c[i]:>9.2f}%{f[i]:>8.2f}%{d[i]:>+9.2f}")
        print(f"\n  paired mean {d.mean():+.2f} pts, better {int((d<0).sum())}/5")
        print(f"    P5 PRIMARY (negative on >=4/5): {'MET' if (d<0).sum()>=4 else 'FAILED'}")
        print(f"    P6 (>= half of h85's -5.85, i.e. <= -2.9): {'MET' if d.mean()<=-2.9 else 'FAILED'}"
              f"   [{d.mean():+.2f}]")
    else:
        print("  INCOMPLETE -- no verdict.")
