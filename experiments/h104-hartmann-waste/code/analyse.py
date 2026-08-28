"""H104: does the ROI reduce waste on Hartmann, where it fails on regret?
Zero compute. Measures identical to h95's so the two benchmarks compare."""
import sys, os, json
import numpy as np
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
from benchmarks import get_benchmark
B="Hartmann_6D"; SEEDS=(42,43,44,45,46)
H84=os.path.join(REPO,"experiments","h84-roi-strategy","results")
H83=os.path.join(REPO,"experiments","h83-main-comparison","results")
hf=get_benchmark(f"{B}_HF"); OPT=abs(float(hf["known_optimal_value"]))
LO=np.asarray(hf["domain_min"],float); HI=np.asarray(hf["domain_max"],float)

def ctrl_path(s):
    p=os.path.join(H84,f"{B}__ROI-OFF__seed{s}.json")
    return p if os.path.exists(p) else os.path.join(H83,f"{B}__MF-DRO__seed{s}.json")

def measures(path):
    q=json.load(open(path))["queries"]
    init=[e for e in q if e.get("is_init") and e["fid"]]
    post=[e for e in q if not e.get("is_init") and e["fid"]]
    if not init or not post: return None
    b0=max(float(e["y"]) for e in init)
    ys=np.array([float(e["y"]) for e in post])
    X=(np.array([np.asarray(e["x"],float) for e in post])-LO)/(HI-LO)
    return dict(n=len(post), W=float((ys<b0).mean()),
                Q=float(np.mean(100.0*(OPT-ys)/OPT)), D=float(X.std(axis=0).mean()))

if __name__=="__main__":
    rows=[]
    for s in SEEDS:
        a=os.path.join(H84,f"{B}__ROI-Q10__seed{s}.json"); c=ctrl_path(s)
        if not (os.path.exists(a) and os.path.exists(c)): continue
        ma,mc=measures(a),measures(c)
        if ma and mc: rows.append((s,ma,mc))
    print("H104 -- waste on Hartmann, where the ROI FAILS on regret (h87 2/5, withdrawn).\n")
    if len(rows)<5: print(f"  INCOMPLETE {len(rows)}/5"); sys.exit()
    out={}
    for key,name in (("W","WASTE FRACTION (y worse than best initial HF)"),
                     ("Q","MEAN QUERY REGRET (%)"),
                     ("D","DISPERSION (mean per-dim std, normalized)")):
        print(f"  {name}")
        print(f"    {'seed':>5}{'ROI':>10}{'no-ROI':>10}{'diff':>10}")
        d=[];floor=[]
        for s,ma,mc in rows:
            mark=""
            if key=="W" and ma[key]==0.0 and mc[key]==0.0: mark="  <- AT FLOOR in both arms"; floor.append(s)
            print(f"    {s:>5}{ma[key]:>10.3f}{mc[key]:>10.3f}{ma[key]-mc[key]:>+10.3f}{mark}")
            d.append(ma[key]-mc[key])
        d=np.array(d); out[key]=d
        better=int((d<0).sum())
        print(f"    paired mean {d.mean():+.3f}   ROI better {better}/5"
              + (f"   ({len(floor)} seed(s) at the floor in BOTH arms: {floor})" if floor else ""))
        print()
    W,Q=out["W"],out["Q"]
    w1=int((W<0).sum())>=3; w2=int((Q<0).sum())>=3
    print("  REGISTERED BARS")
    print(f"    W1 (waste falls on >=3/5): {'MET' if w1 else 'FAILED'}")
    print(f"    W2 (mean query regret falls on >=3/5): {'MET' if w2 else 'FAILED'}")
    print(f"    reference -- Borehole (h95): waste better 3/5 (2 at floor), query regret 5/5 at -4.15")
    print()
    if w1 or w2:
        print("    *** W3 APPLIES. Waste and/or query quality improve on Hartmann while")
        print("        the REGRET result there is failed and withdrawn. WASTE AND REGRET")
        print("        ARE SEPARABLE: the commission's target quantity does not determine")
        print("        the outcome it was asked to produce. State that as a finding about")
        print("        the brief. ***")
    else:
        print("    *** W4 APPLIES. Waste does NOT fall on Hartmann either. The ROI's")
        print("        waste reduction is Borehole-specific, like its relocation and its")
        print("        regret gain -- leaving NO benchmark-general effect of any kind. ***")
