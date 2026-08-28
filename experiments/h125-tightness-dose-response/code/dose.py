import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import json, numpy as np
R="experiments/h84-roi-strategy/results"; S=[42,43,44,45,46]
def load(b,a,s):
    d=json.load(open(f"{R}/{b}__{a}__seed{s}.json"))
    Q=d["queries"]
    i=[q["y"] for q in Q if q["fid"]==1 and q.get("is_init")]
    h=[q["y"] for q in Q if q["fid"]==1 and not q.get("is_init")]
    waste=float(np.mean([y<max(i) for y in h])) if i and h else float("nan")
    return dict(reg=float(d["final_regret"]), waste=waste,
                q=float(d["roi_summary"]["accept_frac"]))
def contrast(b,a0,a1,key,label):
    v0=[load(b,a0,s)[key] for s in S]; v1=[load(b,a1,s)[key] for s in S]
    d=np.array(v1)-np.array(v0)
    e=abs(d.mean())/d.std(ddof=1) if d.std(ddof=1)>0 else float("nan")
    n=int((d<0).sum()) if d.mean()<0 else int((d>0).sum())
    sep=(n>=4) and np.isfinite(e) and e>=1.0
    print(f"  {b:12s} {label:22s} {a0} {np.mean(v0):8.3f} -> {a1} {np.mean(v1):8.3f}"
          f"  paired {d.mean():+8.3f}  sd {d.std(ddof=1):6.3f}  |m|/sd {e:5.2f}  {n}/5  "
          f"{'SEPARATES' if sep else 'no separation'}")
    return sep
print("  realized acceptance actually used (mean over seeds):")
for b in ["Borehole_8D","Hartmann_6D"]:
    qs={a:np.mean([load(b,a,s)["q"] for s in S]) for a in ["ROI-Q10","ROI-ANN","ROI-FIX2"]}
    print(f"    {b:12s} Q10 {qs['ROI-Q10']:.3f}   ANN {qs['ROI-ANN']:.3f}   (FIX2 {qs['ROI-FIX2']:.3f}, excluded)")
print("\n  PRIMARY: q=0.100 vs q~0.495, a 5x range")
any_sep=False
for b in ["Borehole_8D","Hartmann_6D"]:
    any_sep |= contrast(b,"ROI-Q10","ROI-ANN","reg","final_regret")
    any_sep |= contrast(b,"ROI-Q10","ROI-ANN","waste","waste_frac")
print("\n  SECONDARY (reported, excluded from the primary): ROI-FIX2, floating acceptance")
for b in ["Borehole_8D","Hartmann_6D"]:
    contrast(b,"ROI-Q10","ROI-FIX2","reg","final_regret")
print(f"\n  P1/P2 predicted NULL -> {'REFUTED, something separated' if any_sep else 'CONFIRMED: nothing separates'}")
