"""H108: does the L1 regret gain survive at seeds 42-46? Committed before any run.
Control is h83's MF-DRO at matched seeds; the use_roi=False branch is
byte-identical across every commit involved (md5 ff70f008c0ac, h105)."""
import sys, os, json
import numpy as np
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,"experiments","h83-main-comparison","code"))
from analyse import sr_curve, grid
from benchmarks import get_benchmark
G=np.linspace(0,200,201); B="Borehole_8D"; SEEDS=(42,43,44,45,46); BAR=0.59
hf=get_benchmark(f"{B}_HF"); lo=np.array(hf["domain_min"],float); hi=np.array(hf["domain_max"],float)
OPT=abs(float(hf["known_optimal_value"]))
XSTAR=(np.array([0.15,100.0,95090.978,1110.0,116.0,700.0,1120.0,12045.0])-lo)/(hi-lo)
BD=[i for i,v in enumerate(XSTAR) if v<0.02 or v>0.98]
R108=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results"))
H83=os.path.join(REPO,"experiments","h83-main-comparison","results")
def path(arm,s):
    return (os.path.join(R108,f"{B}__L1-LOSS__seed{s}.json") if arm=="L1"
            else os.path.join(H83,f"{B}__MF-DRO__seed{s}.json"))
def at200(p):
    c,sr=sr_curve(json.load(open(p)),-OPT); return 100.0*grid(c,sr,G)[-1]/OPT
def bound_frac(p):
    q=[x for x in json.load(open(p))["queries"] if not x.get("is_init")]
    X=(np.array([x["x"] for x in q],float)-lo)/(hi-lo)
    return float(((X[:,BD]<0.02)|(X[:,BD]>0.98)).mean())
def lloc(p):
    v=(json.load(open(p)).get("L_loc_per_iter") or [])
    return float(np.mean(v[-10:])) if len(v)>=10 else float('nan')
if __name__=="__main__":
    print("H108 -- does the L1 regret gain survive at the ORIGINAL seeds 42-46?\n")
    have=[s for s in SEEDS if os.path.exists(path("L1",s))]
    print("  G3 GATE: L1 must report final L_loc > 0.10 (MSE runs give 0.033-0.038).")
    if not have: print("    no L1-LOSS runs yet\n"); sys.exit()
    void=False
    for s in have:
        v=lloc(path("L1",s)); ok=v>0.10; void=void or not ok
        print(f"    seed {s}: L_loc={v:.4f}  {'OK' if ok else '*** LOOKS LIKE MSE ***'}")
    if void: print("\n    *** GATE FAILED -- runs VOID. ***"); sys.exit(1)
    print("    -> PASS\n")
    rows=[];miss=[]
    for s in SEEDS:
        a,c=path("L1",s),path("NO-ROI",s)
        if os.path.exists(a) and os.path.exists(c):
            rows.append((s,at200(a),at200(c),bound_frac(a),bound_frac(c)))
        else: miss.append(s)
    print(f"  {'seed':>5}{'L1 reg':>9}{'MSE reg':>9}{'diff':>8}{'L1 bnd%':>10}{'MSE bnd%':>10}")
    for s,a,c,ba,bc in rows: print(f"  {s:>5}{a:>9.2f}{c:>9.2f}{a-c:>+8.2f}{100*ba:>10.2f}{100*bc:>10.2f}")
    if miss: print(f"\n  INCOMPLETE -- pending {miss}"); sys.exit()
    d=np.array([a-c for _,a,c,_,_ in rows]); db=np.array([ba-bc for _,_,_,ba,bc in rows])
    print(f"\n  regret     paired mean {d.mean():+.2f}  sd {d.std(ddof=1):.2f}  L1 better {int((d<0).sum())}/5")
    print(f"  bound frac paired mean {100*db.mean():+.2f} pts  L1 LOWER {int((db<0).sum())}/5")
    print(f"  h102 at seeds 47-51 measured: regret -2.08 (4/5); bound frac -0.81 pts (lower 4/5)")
    print("\n  REGISTERED BARS")
    clears = abs(d.mean())>BAR and max(int((d<0).sum()),int((d>0).sum()))>=4
    if clears and d.mean()<0:
        print(f"    P1 (no direction registered): **the gain REPLICATES** at {abs(d.mean()):.2f}, "
              f"clearing the {BAR} bar on a second seed set.")
    elif clears:
        print(f"    P1: *** REVERSED -- L1 is WORSE by {d.mean():.2f} here. h102's gain must be WITHDRAWN. ***")
    else:
        print(f"    P1: does NOT replicate ({abs(d.mean()):.2f} vs the {BAR} bar, split "
              f"{int((d<0).sum())}/5). h102 joins the withdrawn list.")
    p2 = db.mean()<0 and int((db<0).sum())>=4
    print(f"    P2 (L1 reaches bounds LESS, >=4/5 -- predicted from h102's MEASUREMENT): "
          f"{'MET' if p2 else 'FAILED'}")
    print("       This tests my reasoning, not the method: h102's P1 failed because it")
    print("       predicted from an ASSUMED distribution shape. If P2 also fails, then")
    print("       predicting from measurement is not sufficient either, and I should say so.")
