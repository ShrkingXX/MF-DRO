"""H111: does q=0.05 make the ROI work outside Borehole? Committed before any run.
Comparators are h83's MF-DRO control and h84's ROI-Q10, both at seeds 42-46."""
import sys, os, json
import numpy as np
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,"experiments","h83-main-comparison","code"))
from analyse import sr_curve, grid
from benchmarks import get_benchmark
G=np.linspace(0,200,201); SEEDS=(42,43,44,45,46); BAR=0.59
R111=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results"))
H84=os.path.join(REPO,"experiments","h84-roi-strategy","results")
H83=os.path.join(REPO,"experiments","h83-main-comparison","results")
BENCH=("Hartmann_6D","Ackley_10D")
# the q=0.10 record each benchmark is being re-tested against
Q10REF={"Hartmann_6D":"-1.62 pts, 3/5 (magnitude clears 0.59, split fails 4/5)",
        "Ackley_10D":"-0.09 pts, 1/5 (absent, not weak)"}
def path(b,arm,s):
    if arm=="ROI-Q05": return os.path.join(R111,f"{b}__ROI-Q05__seed{s}.json")
    if arm=="ROI-Q10": return os.path.join(H84,f"{b}__ROI-Q10__seed{s}.json")
    return os.path.join(H83,f"{b}__MF-DRO__seed{s}.json")
if __name__=="__main__":
    print("H111 -- does q=0.05 make the ROI work outside Borehole?\n")
    for b in BENCH:
        opt=float(get_benchmark(f"{b}_HF")["known_optimal_value"])
        absr = abs(opt)<1e-9
        def at200(p):
            c,sr=sr_curve(json.load(open(p)),-abs(opt)); v=float(grid(c,sr,G)[-1])
            return v if absr else 100.0*v/abs(opt)
        print(f"=== {b}" + ("   [f(x*)=0 -> ABSOLUTE SR]" if absr else "") + " ===")
        print(f"  q=0.10 record: {Q10REF[b]}")
        have=[s for s in SEEDS if os.path.exists(path(b,"ROI-Q05",s))]
        if not have: print("  no ROI-Q05 runs yet\n"); continue
        print("  G3 GATE (accept_frac in [0.045,0.055]) -- measured:")
        void=False
        for s in have:
            rs=(json.load(open(path(b,"ROI-Q05",s))).get("roi_summary") or {})
            a=rs.get("accept_frac"); ok=a is not None and 0.045<=a<=0.055; void=void or not ok
            print(f"    seed {s}: {a if a is None else round(a,4)}  {'OK' if ok else '*** OUT OF RANGE ***'}")
        if void: print("    *** GATE FAILED -- runs VOID for this benchmark ***\n"); continue
        print("    -> PASS")
        rows=[];miss=[]
        for s in SEEDS:
            ps=[path(b,a,s) for a in ("ROI-Q05","ROI-Q10","NO-ROI")]
            if all(os.path.exists(p) for p in ps): rows.append((s,*[at200(p) for p in ps]))
            else: miss.append(s)
        if rows:
            print(f"  {'seed':>5}{'Q05':>9}{'Q10':>9}{'no ROI':>9}{'Q05-noROI':>11}{'Q05-Q10':>10}")
            for s,a,bb,c in rows: print(f"  {s:>5}{a:>9.2f}{bb:>9.2f}{c:>9.2f}{a-c:>+11.2f}{a-bb:>+10.2f}")
        if miss or not rows: print(f"  INCOMPLETE -- pending {miss}\n"); continue
        d1=np.array([a-c for _,a,_,c in rows]); d2=np.array([a-bb for _,a,bb,_ in rows])
        w1=int((d1<0).sum()); w2=int((d2<0).sum())
        print(f"\n  Q05 vs no-ROI: mean {d1.mean():+.2f}  sd {d1.std(ddof=1):.2f}  better {w1}/5")
        print(f"  Q05 vs Q10   : mean {d2.mean():+.2f}  sd {d2.std(ddof=1):.2f}  better {w2}/5")
        sep = abs(d1.mean())>BAR and max(w1,5-w1)>=4
        if sep and d1.mean()<0:
            print(f"\n  *** {b}: q=0.05 SEPARATES from no-ROI ({abs(d1.mean()):.2f} > {BAR} bar, {w1}/5).")
            print(f"      'The ROI works only on Borehole' is FALSE and was an artefact of q=0.10. ***")
        elif sep:
            print(f"\n  {b}: q=0.05 is separably WORSE than no-ROI ({d1.mean():+.2f}).")
        else:
            print(f"\n  {b}: no separable effect ({abs(d1.mean()):.2f} vs {BAR} bar, {w1}/5).")
            print(f"      Borehole-only survives a real test here rather than an untested assumption.")
        print()
