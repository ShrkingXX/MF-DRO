"""H102: is boundary aversion caused by the L2 loss fitting a conditional mean?
Committed before any run. Reuses h90's NO-ROI control at the same seeds."""
import sys, os, json
import numpy as np
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,"experiments","h83-main-comparison","code"))
from analyse import sr_curve, grid
from benchmarks import get_benchmark
G=np.linspace(0,200,201); B="Borehole_8D"; SEEDS=(47,48,49,50,51)
R102=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results"))
R90=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..",
                                 "h90-borehole-confirm","results"))
hf=get_benchmark(f"{B}_HF"); lo=np.array(hf["domain_min"],float); hi=np.array(hf["domain_max"],float)
OPT=abs(float(hf["known_optimal_value"]))
XSTAR=(np.array([0.15,100.0,95090.978,1110.0,116.0,700.0,1120.0,12045.0])-lo)/(hi-lo)
BD=[i for i,v in enumerate(XSTAR) if v<0.02 or v>0.98]     # 7 of 8 dims
def path(arm,s): return os.path.join(R102 if arm=="L1-LOSS" else R90, f"{B}__{arm}__seed{s}.json")
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
    print(f"H102 -- does an L1 (median) loss cure boundary aversion? Borehole, seeds 47-51.")
    print(f"  x* is on the domain boundary in dims {BD} ({len(BD)} of 8)\n")
    have=[s for s in SEEDS if os.path.exists(path("L1-LOSS",s))]
    print("  G3 GATE -- the manipulation must be OBSERVED, not read back from config.")
    print("  L1 reports a larger loss than MSE at equal fit (|r|>r^2 below 1);")
    print("  h90's MSE runs end at 0.033-0.038, so L1 must exceed 0.10.")
    if not have: print("    no L1-LOSS runs yet\n"); sys.exit()
    void=False
    for s in have:
        v=lloc(path("L1-LOSS",s)); ok=v>0.10
        void=void or not ok
        print(f"    seed {s}: final L_loc={v:.4f}  {'OK' if ok else '*** LOOKS LIKE MSE -- flag did not take ***'}")
    if void:
        print("\n    *** GATE FAILED -- runs are VOID regardless of regret. Do not read on. ***")
        sys.exit(1)
    print("    -> PASS\n")
    rows=[];miss=[]
    for s in SEEDS:
        a,c=path("L1-LOSS",s),path("NO-ROI",s)
        if os.path.exists(a) and os.path.exists(c):
            rows.append((s,at200(a),at200(c),bound_frac(a),bound_frac(c)))
        else: miss.append(s)
    print(f"  {'seed':>5}{'L1 reg':>9}{'MSE reg':>9}{'diff':>8}{'L1 bound%':>11}{'MSE bound%':>11}")
    for s,a,c,ba,bc in rows:
        print(f"  {s:>5}{a:>9.2f}{c:>9.2f}{a-c:>+8.2f}{100*ba:>11.2f}{100*bc:>11.2f}")
    if miss: print(f"\n  INCOMPLETE -- pending {miss}"); sys.exit()
    d=np.array([a-c for _,a,c,_,_ in rows])
    db=np.array([ba-bc for _,_,_,ba,bc in rows])
    print(f"\n  regret     paired mean {d.mean():+.2f}  sd {d.std(ddof=1):.2f}  L1 better {int((d<0).sum())}/5")
    print(f"  bound frac paired mean {100*db.mean():+.2f} pts  L1 higher {int((db>0).sum())}/5")
    print("\n  REGISTERED BARS")
    p1 = db.mean()>0 and int((db>0).sum())>=4
    print(f"    P1 (mechanism: L1 reaches bounds MORE, >=4/5): {'MET' if p1 else 'FAILED'}")
    real = abs(d.mean())>0.59 and max(int((d<0).sum()),int((d>0).sum()))>=4
    print(f"    P2 (no direction registered) -> "
          + (f"{'L1 better' if d.mean()<0 else 'L1 worse'} by {abs(d.mean()):.2f}, and it CLEARS"
             " the 0.59 separability bar" if real else
             f"INDISTINGUISHABLE ({abs(d.mean()):.2f} <= 0.59 bar, or split <4/5)"))
    if p1 and not real:
        print("\n    *** P3 FALSIFIER TRIGGERED: the head reaches bounds more and regret does")
        print("        NOT improve. Boundary aversion is then NOT the cause of the residual")
        print("        Borehole gap, and the explanation this project has been building")
        print("        toward should be retired. State that plainly. ***")
    if not p1:
        print("\n    P1 failed: the intervention did not do the thing it was chosen for,")
        print("    so P2 says nothing about boundary aversion either way.")
