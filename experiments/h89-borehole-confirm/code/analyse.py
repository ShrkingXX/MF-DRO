"""H88: does the Borehole ROI gain survive fresh seeds? Committed BEFORE the
treatment arm finishes, as h87's was. Same metric as everything else."""
import sys, os, json
import numpy as np
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,"experiments","h83-main-comparison","code"))
from analyse import sr_curve, grid
from benchmarks import get_benchmark
G=np.linspace(0,200,201); B="Borehole_8D"; SEEDS=(47,48,49,50,51)
R=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results")
OPT=abs(float(get_benchmark(f"{B}_HF")["known_optimal_value"]))
def at200(p):
    c,sr=sr_curve(json.load(open(p)),-OPT); return 100.0*grid(c,sr,G)[-1]/OPT
if __name__=="__main__":
    rows=[];miss=[]
    for s in SEEDS:
        a=os.path.join(R,f"{B}__ROI-Q10__seed{s}.json"); c=os.path.join(R,f"{B}__NO-ROI__seed{s}.json")
        if os.path.exists(a) and os.path.exists(c): rows.append((s,at200(a),at200(c)))
        else: miss.append(s)
    print("H89 -- Borehole ROI gain at FRESH seeds, q=0.10 fixed in advance.\n")
    if not rows: print("  no complete pairs yet"); sys.exit()
    print(f"  {'seed':>5}{'ROI-Q10':>10}{'no ROI':>9}{'diff':>9}")
    for s,a,c in rows: print(f"  {s:>5}{a:>10.2f}{c:>9.2f}{a-c:>+9.2f}")
    d=np.array([a-c for _,a,c in rows]); wins=int((d<0).sum())
    print(f"\n  n={len(d)}  paired mean {d.mean():+.2f}  "
          f"sd {d.std(ddof=1) if len(d)>1 else float('nan'):.2f}  ROI better {wins}/{len(d)}")
    print(f"  seeds 42-46 for reference: mean -4.22, sd 2.43, 5/5")
    if miss: print(f"  INCOMPLETE -- pairs pending: {miss}"); sys.exit()
    print("\n  REGISTERED BARS")
    p1 = wins>=4 and d.mean()<0
    print(f"    P1 (negative on >=4/5 AND negative mean): {'MET' if p1 else 'FAILED'}")
    print(f"    P2 (margin shrinks vs -4.22): "
          f"{'MET' if abs(d.mean())<4.22 or d.mean()>0 else 'NOT MET'}")
    roi=np.array([a for _,a,_ in rows])
    print(f"    P3 (still does NOT beat MF-MES 6.40): "
          f"{'MET (still behind)' if roi.mean()>6.40 else '*** REFUTED -- investigate ***'}")
    if not p1:
        print("\n  *** P1 FAILED -> per the protocol's falsifier, the Borehole gain is")
        print("      WITHDRAWN, and the ROI has NO surviving regret result on any")
        print("      benchmark. Only the controllability argument remains. State that")
        print("      plainly in findings.md and the report. ***")
