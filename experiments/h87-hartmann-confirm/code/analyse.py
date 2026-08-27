"""H87: does the Hartmann flip survive fresh seeds with q=0.10 fixed in advance?

Same metric as everything else: h83's sr_curve/grid, SR at exactly cost 200.
Reports the PAIRED difference, which is the quantity that was tight (sd 0.45)
on the original seeds -- the marginal spreads are ~7x larger because seed
difficulty is common-mode and cancels in the pairing.
"""
import sys, os, json
import numpy as np
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,"experiments","h83-main-comparison","code"))
from analyse import sr_curve, grid
from benchmarks import get_benchmark
G=np.linspace(0,200,201); B="Hartmann_6D"; SEEDS=(47,48,49,50,51)
R=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results")
OPT=abs(float(get_benchmark(f"{B}_HF")["known_optimal_value"]))
def at200(p):
    c,sr=sr_curve(json.load(open(p)),-OPT); return 100.0*grid(c,sr,G)[-1]/OPT

if __name__=="__main__":
    rows=[]; miss=[]
    for s in SEEDS:
        pd_=os.path.join(R,f"{B}__MF-DRO__seed{s}.json"); pm=os.path.join(R,f"{B}__MF-MES__seed{s}.json")
        if os.path.exists(pd_) and os.path.exists(pm): rows.append((s,at200(pd_),at200(pm)))
        else: miss.append(s)
    print("H87 -- Hartmann flip at FRESH seeds, q=0.10 fixed in advance, one arm.\n")
    if not rows: print("  no paired seeds yet"); sys.exit()
    print(f"  {'seed':>5}{'MF-DRO+ROI':>13}{'MF-MES':>10}{'diff':>9}")
    for s,a,m in rows: print(f"  {s:>5}{a:>13.2f}{m:>10.2f}{a-m:>+9.2f}")
    d=np.array([a-m for _,a,m in rows])
    wins=int((d<0).sum())
    print(f"\n  n={len(d)}  mean paired diff {d.mean():+.2f} pts"
          f"   sd {d.std(ddof=1) if len(d)>1 else float('nan'):.2f}"
          f"   MF-DRO better on {wins}/{len(d)}")
    print(f"  original seeds 42-46 for reference: mean -0.68, sd 0.45, 4/5")
    if miss: print(f"  INCOMPLETE -- seeds still running: {miss}")
    if len(d)==5:
        print("\n  REGISTERED BARS")
        print(f"    P1 paired diff negative on >=4/5 AND mean negative: "
              f"{'MET' if (wins>=4 and d.mean()<0) else 'FAILED'}")
        print(f"    P2 margin shrinks vs -0.68: "
              f"{'MET' if abs(d.mean())<0.68 or d.mean()>0 else 'NOT MET (margin held or grew)'}")
        mdro=np.array([a for _,a,_ in rows]); mmes=np.array([m for _,_,m in rows])
        print(f"    P3 h83's full bar (lower mean AND >=4/5): "
              f"{'MET' if (mdro.mean()<mmes.mean() and wins>=4) else 'FAILED'}")
        if wins<4 or d.mean()>=0:
            print("\n  P1 FAILED -> per the protocol's falsifier, the Hartmann flip must be")
            print("  WITHDRAWN from findings.md and the to_human report as prominently as")
            print("  it was announced.")
