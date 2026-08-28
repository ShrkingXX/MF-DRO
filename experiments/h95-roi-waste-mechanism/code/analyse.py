"""H95: does the ROI's confirmed regret gain come from wasting less HF budget?
Zero new compute; reads h90's completed runs. Committed BEFORE computing."""
import sys, os, json
import numpy as np
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
from benchmarks import get_benchmark
B="Borehole_8D"; SEEDS=(47,48,49,50,51)
H90=os.path.join(REPO,"experiments","h90-borehole-confirm","results")
hf=get_benchmark(f"{B}_HF"); OPT=abs(float(hf["known_optimal_value"]))
LO=np.asarray(hf["domain_min"],dtype=float); HI=np.asarray(hf["domain_max"],dtype=float)

def measures(path):
    q=json.load(open(path))["queries"]
    init_hf=[e for e in q if e.get("is_init") and e["fid"]]
    post_hf=[e for e in q if not e.get("is_init") and e["fid"]]
    if not init_hf or not post_hf: return None
    best_init=max(float(e["y"]) for e in init_hf)
    ys=np.array([float(e["y"]) for e in post_hf])
    X=np.array([np.asarray(e["x"],dtype=float) for e in post_hf])
    Xn=(X-LO)/(HI-LO)
    return dict(n=len(post_hf),
                W=float((ys<best_init).mean()),
                D=float(Xn.std(axis=0).mean()),
                Q=float(np.mean(100.0*(OPT-ys)/OPT)))

if __name__=="__main__":
    rows=[]
    for s in SEEDS:
        a=os.path.join(H90,f"{B}__ROI-Q10__seed{s}.json"); c=os.path.join(H90,f"{B}__NO-ROI__seed{s}.json")
        if not (os.path.exists(a) and os.path.exists(c)): continue
        ma,mc=measures(a),measures(c)
        if ma and mc: rows.append((s,ma,mc))
    print("H95 -- does the ROI waste less HF budget? Borehole, seeds 47-51, h90 runs.\n")
    if len(rows)<5:
        print(f"  INCOMPLETE: {len(rows)}/5 pairs. No verdict."); sys.exit()
    for key,name,lo_is_better in (("W","WASTE FRACTION (y worse than best initial HF)",True),
                                  ("D","DISPERSION (mean per-dim std, normalized)",True),
                                  ("Q","MEAN QUERY REGRET (%)",True)):
        print(f"  {name}")
        print(f"    {'seed':>5}{'ROI':>10}{'no-ROI':>10}{'diff':>10}")
        d=[]
        for s,ma,mc in rows:
            print(f"    {s:>5}{ma[key]:>10.3f}{mc[key]:>10.3f}{ma[key]-mc[key]:>+10.3f}")
            d.append(ma[key]-mc[key])
        d=np.array(d); better=int((d<0).sum())
        print(f"    paired mean {d.mean():+.3f}   ROI better {better}/5")
        print()
    W=np.array([ma["W"]-mc["W"] for _,ma,mc in rows])
    m1 = int((W<0).sum())>=4
    print(f"  M1 (waste fraction falls on >=4/5): {'MET' if m1 else 'FAILED'}")
    D=np.array([ma["D"]-mc["D"] for _,ma,mc in rows]); Q=np.array([ma["Q"]-mc["Q"] for _,ma,mc in rows])
    print(f"  M2 (dispersion falls on >=4/5):     {'MET' if int((D<0).sum())>=4 else 'FAILED'}")
    print(f"  M3 (mean query regret falls >=4/5): {'MET' if int((Q<0).sum())>=4 else 'FAILED'}")
    if not m1:
        print()
        print("  *** M1 FAILED -> M4 applies. The ROI lowers FINAL REGRET without")
        print("      reducing wasted HF queries: it improves the BEST query, not the")
        print("      average one. Per this protocol's falsifier, the sentence 'the ROI")
        print("      stops MF-DRO wasting HF budget' must NOT be written. The")
        print("      defensible claim is 'the ROI lowers final regret on Borehole'. ***")
