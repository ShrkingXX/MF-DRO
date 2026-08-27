"""H91: MF-DRO vs MF-MES at seeds 52-56, paired. Committed before the runs."""
import sys, os, json
import numpy as np
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,"experiments","h83-main-comparison","code"))
from analyse import sr_curve, grid
from benchmarks import get_benchmark
G=np.linspace(0,200,201); B="Hartmann_6D"; SEEDS=(52,53,54,55,56)
OPT=abs(float(get_benchmark(f"{B}_HF")["known_optimal_value"]))
R=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results")
H89=os.path.join(REPO,"experiments","h89-hffloor-confirm","results")
def at200(p):
    c,sr=sr_curve(json.load(open(p)),-OPT); return 100.0*grid(c,sr,G)[-1]/OPT
if __name__=="__main__":
    rows=[];miss=[]
    for s in SEEDS:
        m=os.path.join(R,f"{B}__MF-MES__seed{s}.json"); d=os.path.join(H89,f"{B}__CONTROL__seed{s}.json")
        if os.path.exists(m) and os.path.exists(d): rows.append((s,at200(d),at200(m)))
        else: miss.append(s)
    print("H91 -- MF-DRO vs MF-MES at fresh seeds 52-56 (MF-DRO from H89's control arm).\n")
    if not rows: print("  no pairs yet"); sys.exit()
    print(f"  {'seed':>5}{'MF-DRO':>9}{'MF-MES':>9}{'diff':>9}")
    for s,d,m in rows: print(f"  {s:>5}{d:>9.2f}{m:>9.2f}{d-m:>+9.2f}")
    v=np.array([d-m for _,d,m in rows]); w=int((v<0).sum())
    print(f"\n  n={len(v)}  paired mean {v.mean():+.2f}  MF-DRO better {w}/{len(v)}")
    print(f"  seeds 42-46 for reference: MF-DRO 7.99, MF-MES 6.62, deficit +1.37")
    if miss: print(f"  INCOMPLETE -- pending {miss}"); sys.exit()
    print("\n  REGISTERED BARS")
    print(f"    P1 (MF-DRO beats MF-MES on >=3/5): {'MET' if w>=3 else 'FAILED'}")
    print(f"    P2 (deficit smaller than +1.37):   {'MET' if v.mean()<1.37 else 'NOT MET'}   [{v.mean():+.2f}]")
    if v.mean()<1.37:
        print("\n  -> The founding diagnosis's deficit is SMALLER at these seeds. State how")
        print("     much of h84-h90's programme was aimed at a seed-set artefact.")
