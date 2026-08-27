"""H87 verdict. Every comparison is WITHIN h87 at matched seeds -- never against
h83's MF-MES, whose seeds are 3.27 pts easier (Amendment 1).

Uses h83's own sr_curve/grid so the metric matches every other experiment.
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
    rows=[]
    for s in SEEDS:
        pd=os.path.join(R,f"{B}__MF-DRO__seed{s}.json"); pm=os.path.join(R,f"{B}__MF-MES__seed{s}.json")
        if os.path.exists(pd) and os.path.exists(pm): rows.append((s,at200(pd),at200(pm)))
    if not rows: print("no paired seeds yet"); sys.exit()
    print(f"H87 -- Hartmann, fresh seeds, q=0.10 fixed in advance. {len(rows)}/5 paired.\n")
    print(f"  {'seed':>5}{'MF-DRO+ROI':>13}{'MF-MES':>10}{'diff':>9}")
    for s,d,m in rows: print(f"  {s:>5}{d:>13.2f}{m:>10.2f}{d-m:>+9.2f}")
    D=np.array([d for _,d,_ in rows]); M=np.array([m for _,_,m in rows]); dif=D-M
    wins=int((dif<0).sum())
    print(f"\n  paired mean diff {dif.mean():+.2f} pts   sd {dif.std(ddof=1) if len(dif)>1 else float('nan'):.2f}"
          f"   wins {wins}/{len(rows)}")
    print(f"  means: MF-DRO+ROI {D.mean():.2f}   MF-MES {M.mean():.2f}")
    if len(rows)<5:
        print("\n  INCOMPLETE -- no verdict. Partial arms are unreliable in either"
              "\n  direction in this project (ROI-FIX2 moved +6.32 -> +0.36 -> -0.26).")
        sys.exit()
    print("\n  PRE-REGISTERED BARS")
    p1 = (wins>=4 and dif.mean()<0)
    print(f"    P1 (paired diff negative, >= 4/5 seeds AND negative mean): "
          f"{'MET' if p1 else 'FAILED'}   [{wins}/5, mean {dif.mean():+.2f}]")
    print(f"    P2 (margin SHRINKS vs h84's -0.68): "
          f"{'MET' if abs(dif.mean())<0.68 or dif.mean()>-0.68 else 'NOT MET'}"
          f"   [h84 -0.68, here {dif.mean():+.2f}]")
    p3 = (D.mean()<M.mean() and wins>=4)
    print(f"    P3 (h83's full bar: lower mean AND >= 4/5): {'MET' if p3 else 'FAILED'}")
    print("\n  Amendment 1: seeds 47-51 are 3.27 pts harder and ~3x more dispersed for")
    print("  MF-MES than h83's, so P3's mean half is unstable; P1 is the weighted bar.")
    if not p1:
        print("\n  *** P1 FAILED -> the protocol's falsifier applies: the Hartmann flip")
        print("      must be WITHDRAWN from findings.md and the published report, as")
        print("      prominently as it was announced. ***")
