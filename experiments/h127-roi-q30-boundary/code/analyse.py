"""H127: where does the ROI stop working? Committed before any result.

Reads q=0.30 against every other measured acceptance level on Borehole, all
against the same no-ROI control at matched seeds.

NOTE ON THE INTERVAL'S EDGES (recorded before results, from a peer's h125):
the hard points on this curve are q=0.100 and q=0.493 -- both realized EXACTLY,
every seed, three decimals. ROI-FIX2's 0.214 is a FLOATING quantity, per-seed
range 0.162-0.265, because it holds beta fixed and lets acceptance drift. So the
untested interval's lower edge is soft, and a result near FIX2 must not be read
as pinned to 0.21.
"""
import sys, os, json
import numpy as np
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,"experiments","h83-main-comparison","code"))
from analyse import sr_curve, grid
from benchmarks import get_benchmark
G=np.linspace(0,200,201); B="Borehole_8D"; BAR=0.59
OPT=abs(float(get_benchmark(f"{B}_HF")["known_optimal_value"]))
E=lambda *p: os.path.join(REPO,"experiments",*p)
R127=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results"))
S1=(42,43,44,45,46); S2=(47,48,49,50,51)
SRC={  # arm -> {seedset: path template}.  Explicit, never globbed.
 "no-ROI": {S1:E("h83-main-comparison","results",f"{B}__MF-DRO__seed%d.json"),
            S2:E("h90-borehole-confirm","results",f"{B}__NO-ROI__seed%d.json")},
 "q=0.10": {S1:E("h84-roi-strategy","results",f"{B}__ROI-Q10__seed%d.json"),
            S2:E("h90-borehole-confirm","results",f"{B}__ROI-Q10__seed%d.json")},
 "q=0.30": {S1:os.path.join(R127,f"{B}__ROI-Q30__seed%d.json"),
            S2:os.path.join(R127,f"{B}__ROI-Q30__seed%d.json")},
 "q~0.21": {S1:E("h84-roi-strategy","results",f"{B}__ROI-FIX2__seed%d.json")},
 "q=0.49": {S1:E("h84-roi-strategy","results",f"{B}__ROI-ANN__seed%d.json")},
}
def at(p):
    r=json.load(open(p)); c,sr=sr_curve(r,-OPT)
    return 100.0*grid(c,sr,G)[-1]/OPT, (r.get("roi_summary") or {}).get("accept_frac")
def cell(arm):
    out={}
    for ss,tpl in SRC[arm].items():
        for s in ss:
            p=tpl%s
            if os.path.exists(p): out[s]=at(p)
    return out
if __name__=="__main__":
    print("H127 -- where does the ROI stop working?\n")
    q30=cell("q=0.30"); base=cell("no-ROI")
    have=sorted(q30)
    print("  G3 GATE: every run's accept_frac must lie in [0.29, 0.31] (measured, not config).")
    if not have: print("    no ROI-Q30 runs yet\n"); sys.exit()
    void=False
    for s in have:
        a=q30[s][1]; ok=a is not None and 0.29<=a<=0.31; void=void or not ok
        print(f"    seed {s}: accept_frac={a if a is None else round(a,4)}  {'OK' if ok else '*** OUT OF RANGE ***'}")
    if void: print("\n    *** GATE FAILED -- calibration did not hit 0.30. Arm VOID. ***"); sys.exit(1)
    print(f"    -> PASS (mean {np.mean([q30[s][1] for s in have]):.4f})\n")
    miss=[s for s in S1+S2 if s not in q30]
    if miss: print(f"  INCOMPLETE -- pending {miss}\n")
    print("  THE CURVE (Borehole, each arm vs the SAME no-ROI control at matched seeds)")
    print(f"  {'arm':9s}{'realized q':>12}{'n':>4}{'vs no-ROI':>11}{'better':>9}")
    for arm in ("q=0.10","q~0.21","q=0.30","q=0.49"):
        c=cell(arm); ss=[s for s in sorted(c) if s in base]
        if not ss: continue
        d=np.array([c[s][0]-base[s][0] for s in ss]); qq=np.mean([c[s][1] for s in ss if c[s][1]])
        soft=" (floats 0.162-0.265)" if arm=="q~0.21" else ""
        print(f"  {arm:9s}{qq:>12.3f}{len(ss):>4}{d.mean():>+11.2f}{int((d<0).sum()):>6}/{len(ss)}{soft}")
    if miss: sys.exit()
    q10=cell("q=0.10")
    d2=np.array([q30[s][0]-q10[s][0] for s in sorted(q30) if s in q10])
    print(f"\n  Q30 vs Q10: mean {d2.mean():+.2f}  sd {d2.std(ddof=1):.2f}  Q30 better {int((d2<0).sum())}/{len(d2)}")
    sep = abs(d2.mean())>BAR and max(int((d2<0).sum()),len(d2)-int((d2<0).sum()))>=8
    print("\n  REGISTERED BARS")
    d1=np.array([q30[s][0]-base[s][0] for s in sorted(q30) if s in base])
    print(f"    P1 (Q30 still beats no-ROI, >=8/10): "
          f"{'MET' if (d1.mean()<0 and int((d1<0).sum())>=8) else 'FAILED'}")
    if not sep:
        print(f"    P2 (no direction registered): Q30 is INDISTINGUISHABLE from Q10")
        print(f"       -> the plateau extends to at least 0.30, and the collapse is SHARP")
        print(f"          somewhere in (0.30, 0.49). The knob is forgiving over a 6x range.")
    elif d2.mean()>0:
        print(f"    P2: Q30 is WORSE than Q10 by {d2.mean():.2f}, clearing the bar")
        print(f"       -> the decline begins at or below 0.30; 'q <= 0.10' is a real")
        print(f"          recommendation rather than a conservative one.")
    else:
        print(f"    P2: *** Q30 BEATS Q10 by {abs(d2.mean()):.2f} -- investigate ***")
