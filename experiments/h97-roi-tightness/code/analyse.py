"""H97: is q=0.10 the right ROI tightness? Committed before any run.

Reuses h90's ROI-Q10 and NO-ROI arms at the same seeds (same code, same commit)
and imports h83's frozen sr_curve/grid so the metric is unchanged.
"""
import sys, os, json
import numpy as np
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,"experiments","h83-main-comparison","code"))
from analyse import sr_curve, grid
from benchmarks import get_benchmark
G=np.linspace(0,200,201); B="Borehole_8D"; SEEDS=(47,48,49,50,51)
R97=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results"))
R90=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..",
                                 "h90-borehole-confirm","results"))
OPT=abs(float(get_benchmark(f"{B}_HF")["known_optimal_value"]))
def at200(p):
    c,sr=sr_curve(json.load(open(p)),-OPT); return 100.0*grid(c,sr,G)[-1]/OPT
def path(arm,s):
    return os.path.join(R97 if arm=="ROI-Q05" else R90, f"{B}__{arm}__seed{s}.json")
if __name__=="__main__":
    print("H97 -- ROI tightness: q=0.05 vs the confirmed q=0.10, Borehole seeds 47-51.\n")
    # ---- G3 GATE: the ON path's side effect must be OBSERVED before any regret is read
    print("  G3 GATE (accept_frac must be in [0.045, 0.055]):")
    accs=[];void=False
    for s in SEEDS:
        p=path("ROI-Q05",s)
        if not os.path.exists(p): continue
        rs=(json.load(open(p)).get("roi_summary") or {})
        a=rs.get("accept_frac")
        ok = a is not None and 0.045<=a<=0.055
        void = void or not ok
        print(f"    seed {s}: accept_frac={a if a is None else round(a,4)}  {'OK' if ok else '*** OUT OF RANGE ***'}")
        if a is not None: accs.append(a)
    if not accs: print("    no ROI-Q05 runs yet\n")
    elif void:
        print("\n    *** GATE FAILED -- the q=0.05 setting did not take. Runs are VOID")
        print("        regardless of what the regret says. Do not read further. ***"); sys.exit(1)
    else: print(f"    -> PASS (mean {np.mean(accs):.4f}, target 0.05)\n")
    rows=[];miss=[]
    for s in SEEDS:
        a,b,c=path("ROI-Q05",s),path("ROI-Q10",s),path("NO-ROI",s)
        if all(os.path.exists(x) for x in (a,b,c)): rows.append((s,at200(a),at200(b),at200(c)))
        else: miss.append(s)
    if not rows: print("  no complete triples yet"); sys.exit()
    print(f"  {'seed':>5}{'Q05':>9}{'Q10':>9}{'no ROI':>9}{'Q05-noROI':>11}{'Q05-Q10':>10}")
    for s,q5,q10,off in rows:
        print(f"  {s:>5}{q5:>9.2f}{q10:>9.2f}{off:>9.2f}{q5-off:>+11.2f}{q5-q10:>+10.2f}")
    d1=np.array([q5-off for _,q5,_,off in rows]); d2=np.array([q5-q10 for _,q5,q10,_ in rows])
    print(f"\n  n={len(rows)}")
    print(f"    Q05 vs no-ROI: mean {d1.mean():+.2f}  sd {d1.std(ddof=1) if len(d1)>1 else float('nan'):.2f}  better {int((d1<0).sum())}/{len(d1)}")
    print(f"    Q05 vs Q10   : mean {d2.mean():+.2f}  sd {d2.std(ddof=1) if len(d2)>1 else float('nan'):.2f}  Q05 better {int((d2<0).sum())}/{len(d2)}")
    print(f"    reference: Q10 vs no-ROI was -3.49, 4/5")
    if miss: print(f"\n  INCOMPLETE -- pending {miss}"); sys.exit()
    print("\n  REGISTERED BARS")
    print(f"    P1 (Q05 beats no-ROI, >=4/5 and negative mean): "
          f"{'MET' if (int((d1<0).sum())>=4 and d1.mean()<0) else 'FAILED'}")
    print(f"    P2 (no direction predicted) -> ", end="")
    if abs(d2.mean())<0.5 and 2<=int((d2<0).sum())<=3:
        print("INDISTINGUISHABLE: the gain is robust to tightness over 2x.")
        print("       That is the more useful claim -- the mechanism is the region, not the threshold.")
    elif d2.mean()<0: print(f"Q05 BEATS Q10 by {abs(d2.mean()):.2f} -> the optimum is BELOW 0.10 and every")
    else: print(f"Q10 beats Q05 by {d2.mean():.2f} -> 0.10 is at/near a local optimum from below;")
    q5m=np.array([q5 for _,q5,_,_ in rows]).mean()
    print(f"    P3 (still not competitive vs MF-MES 6.40): "
          f"{'MET' if q5m>6.40 else '*** REFUTED -- investigate ***'}  (Q05 mean {q5m:.2f}%)")
