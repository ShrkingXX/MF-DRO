"""H107: does q=0.05's advantage over q=0.10 survive at seeds 42-46?
Committed before any h107 result exists. Reuses h84's ROI-Q10 and h83's MF-DRO
control at matched seeds; the use_roi=False branch is byte-identical across every
commit involved (md5 ff70f008c0ac, established in h105)."""
import sys, os, json
import numpy as np
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,"experiments","h83-main-comparison","code"))
from analyse import sr_curve, grid
from benchmarks import get_benchmark
G=np.linspace(0,200,201); B="Borehole_8D"; SEEDS=(42,43,44,45,46)
OPT=abs(float(get_benchmark(f"{B}_HF")["known_optimal_value"]))
R107=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results"))
H84=os.path.join(REPO,"experiments","h84-roi-strategy","results")
H83=os.path.join(REPO,"experiments","h83-main-comparison","results")
BAR=0.59   # registered separability bar: two settings 2.1x apart differ by this
def at200(p):
    c,sr=sr_curve(json.load(open(p)),-OPT); return 100.0*grid(c,sr,G)[-1]/OPT
def path(arm,s):
    if arm=="ROI-Q05": return os.path.join(R107,f"{B}__ROI-Q05__seed{s}.json")
    if arm=="ROI-Q10": return os.path.join(H84,f"{B}__ROI-Q10__seed{s}.json")
    return os.path.join(H83,f"{B}__MF-DRO__seed{s}.json")
if __name__=="__main__":
    print("H107 -- does q=0.05 still beat q=0.10 at the ORIGINAL seeds 42-46?\n")
    have=[s for s in SEEDS if os.path.exists(path("ROI-Q05",s))]
    print("  G3 GATE (accept_frac must be in [0.045, 0.055]) -- measured, not read back:")
    if not have: print("    no ROI-Q05 runs yet\n"); sys.exit()
    void=False; accs=[]
    for s in have:
        rs=(json.load(open(path("ROI-Q05",s))).get("roi_summary") or {})
        a=rs.get("accept_frac"); ok=a is not None and 0.045<=a<=0.055
        void=void or not ok; accs.append(a)
        print(f"    seed {s}: accept_frac={a if a is None else round(a,4)}  {'OK' if ok else '*** OUT OF RANGE ***'}")
    if void:
        print("\n    *** GATE FAILED -- the q=0.05 setting did not take. Runs VOID. ***"); sys.exit(1)
    print(f"    -> PASS (mean {np.mean([a for a in accs if a]):.4f}, target 0.05)\n")
    rows=[];miss=[]
    for s in SEEDS:
        ps=[path(a,s) for a in ("ROI-Q05","ROI-Q10","NO-ROI")]
        if all(os.path.exists(p) for p in ps): rows.append((s,*[at200(p) for p in ps]))
        else: miss.append(s)
    if not rows: print("  no complete triples yet"); sys.exit()
    print(f"  {'seed':>5}{'Q05':>9}{'Q10':>9}{'no ROI':>9}{'Q05-noROI':>11}{'Q05-Q10':>10}")
    for s,a,b,c in rows: print(f"  {s:>5}{a:>9.2f}{b:>9.2f}{c:>9.2f}{a-c:>+11.2f}{a-b:>+10.2f}")
    if miss: print(f"\n  INCOMPLETE -- pending {miss}"); sys.exit()
    d1=np.array([a-c for _,a,_,c in rows]); d2=np.array([a-b for _,a,b,_ in rows])
    print(f"\n  Q05 vs no-ROI: mean {d1.mean():+.2f}  sd {d1.std(ddof=1):.2f}  better {int((d1<0).sum())}/5")
    print(f"  Q05 vs Q10   : mean {d2.mean():+.2f}  sd {d2.std(ddof=1):.2f}  Q05 better {int((d2<0).sum())}/5")
    print(f"  h97 at seeds 47-51 measured: Q05 vs no-ROI -5.01 (5/5); Q05 vs Q10 -1.52 (4/5)")
    print("\n  REGISTERED BARS")
    print(f"    P1 (Q05 beats no-ROI, >=4/5 and negative mean): "
          f"{'MET' if (int((d1<0).sum())>=4 and d1.mean()<0) else 'FAILED'}")
    clears = abs(d2.mean())>BAR and max(int((d2<0).sum()),int((d2>0).sum()))>=4
    if clears and d2.mean()<0:
        print(f"    P2 (no direction registered): **q=0.05 BEATS q=0.10 AGAIN** by {abs(d2.mean()):.2f},")
        print(f"       clearing the {BAR} bar on a SECOND seed set. Pooled n=10 with h97.")
    elif clears:
        print(f"    P2: *** REVERSED -- q=0.10 beats q=0.05 by {d2.mean():.2f} here. h97's")
        print(f"       ordering is seed-set specific and must be WITHDRAWN. ***")
    else:
        print(f"    P2: INDISTINGUISHABLE ({abs(d2.mean()):.2f} vs the {BAR} bar, split "
              f"{int((d2<0).sum())}/5). h97's ordering does NOT replicate; the honest")
        print(f"       reading reverts to 'the region matters, the threshold does not'.")
    q5=np.array([a for _,a,_,_ in rows]).mean()
    print(f"    P3 (still not competitive vs MF-MES 6.40): "
          f"{'MET' if q5>6.40 else '*** REFUTED ***'}  (Q05 mean {q5:.2f}%)")
