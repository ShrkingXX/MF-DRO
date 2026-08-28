import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import json, sys, numpy as np
sys.path.insert(0,".")
from benchmarks import get_benchmark
CK="experiments/h90-borehole-confirm/results/ckpt"
ARMS=["NO-ROI","ROI-Q10","REFINE-100"]; SEEDS=[47,48,49,50,51]; B="Borehole_8D"; NFLOOR=15
hf=get_benchmark(f"{B}_HF"); lo=np.array(hf["domain_min"],float); wd=np.array(hf["domain_max"],float)-lo
def run(a,s):
    d=json.load(open(f"{CK}/{B}__{a}__seed{s}.json"))
    Q=[q for q in d["queries"] if q["fid"]==1 and not q.get("is_init",False)]
    z=np.array([(q["x"][0]-lo[0])/wd[0] for q in Q]); y=np.array([q["y"] for q in Q])
    allq=[q for q in d["queries"] if q["fid"]==1]
    return z,y,max(q["y"] for q in allq)
W={}; R={}
for a in ARMS:
    for s in SEEDS:
        z,y,best=run(a,s)
        if len(z)<NFLOOR: W[(a,s)]=None; print(f"  EXCLUDED {a} seed{s}: n={len(z)}<{NFLOOR}"); continue
        W[(a,s)]=dict(n=len(z),w90=float((z<0.9).mean()),w95=float((z<0.95).mean()),best=float(best))
        R[(a,s)]=float(best)
for cut in ("w90","w95"):
    print(f"\n===== wasted HF fraction, cut z0 < {'0.9' if cut=='w90' else '0.95'} =====")
    print(f"  {'arm':11s} " + " ".join(f"s{s:>5d}" for s in SEEDS) + f" {'mean':>7s}")
    for a in ARMS:
        v=[W[(a,s)][cut] if W[(a,s)] else np.nan for s in SEEDS]
        print(f"  {a:11s} " + " ".join(f"{100*x:5.1f}%" for x in v) + f" {100*np.nanmean(v):6.1f}%")
    d=np.array([W[("ROI-Q10",s)][cut]-W[("NO-ROI",s)][cut] for s in SEEDS if W[("ROI-Q10",s)] and W[("NO-ROI",s)]])
    n_better=int((d<0).sum()); eff=abs(d.mean())/d.std(ddof=1) if d.std(ddof=1)>0 else float("nan")
    print(f"  PAIRED ROI-Q10 minus NO-ROI: mean {100*d.mean():+.2f} pts, sd {100*d.std(ddof=1):.2f}, "
          f"|mean|/sd = {eff:.2f}, ROI lower in {n_better}/{len(d)} seeds")
    if cut=="w90":
        print(f"  GATE (>=4/5 seeds AND effect>=1.0): {'PASS' if (n_better>=4 and eff>=1.0) else 'FAIL'}")
print("\n===== EXPLORATORY: waste vs final best HF y (15 runs, 3 non-independent arms) =====")
xs=[W[k]["w90"] for k in W if W[k]]; ys=[W[k]["best"] for k in W if W[k]]
print(f"  n={len(xs)}  Pearson r = {np.corrcoef(xs,ys)[0,1]:+.3f}  (description only, no inference)")
for a in ARMS:
    print(f"  {a:11s} mean best y {np.mean([W[(a,s)]['best'] for s in SEEDS if W[(a,s)]]):7.2f}")
