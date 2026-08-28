import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import json, sys, numpy as np
sys.path.insert(0,"tools"); sys.path.insert(0,".")
from perdim import agg
from benchmarks import get_benchmark
CK="experiments/h90-borehole-confirm/results/ckpt"; B="Borehole_8D"; S=[47,48,49,50,51]
ARMS=["NO-ROI","ROI-Q10"]
hf=get_benchmark(f"{B}_HF"); lo=np.array(hf["domain_min"],float); wd=np.array(hf["domain_max"],float)-lo

def metrics(a,s):
    d=json.load(open(f"{CK}/{B}__{a}__seed{s}.json")); Q=d["queries"]
    init_hf=[q for q in Q if q["fid"]==1 and q.get("is_init")]
    hfq=[q for q in Q if q["fid"]==1 and not q.get("is_init")]
    lfq=[q for q in Q if q["fid"]==0 and not q.get("is_init")]
    tot=max(q["cost_cum"] for q in Q)
    cost_hf=2.0*len(hfq); y=np.array([q["y"] for q in hfq])
    y0=np.array([q["y"] for q in init_hf]); spread=y0.std(ddof=1) if len(y0)>1 else 1.0
    best=y.max() if len(y) else -np.inf
    first=next((q["cost_cum"] for q in hfq if q["y"]>=best), tot)
    Z=np.array([[(q["x"][j]-lo[j])/wd[j] for j in range(len(lo))] for q in hfq])
    h=len(Z)//2
    if h>=3 and len(Z)-h>=3:
        e=agg(Z[:h].std(axis=0,ddof=1),B); l=agg(Z[h:].std(axis=0,ddof=1),B)
        contract=e/l if l>0 else np.nan
    else: contract=np.nan
    return dict(C1=cost_hf/tot, C2=len(hfq), C3=len(lfq), C4=first/tot,
                C5=(y.mean()-y0.mean())/spread, C6=float((y<y0.max()).mean()), C7=contract)

NAME={"C1":"HF share of cost budget","C2":"HF query count","C3":"LF query count",
      "C4":"time-to-incumbent (frac budget)","C5":"HF quality (sd of init design)",
      "C6":"frac HF worse than best init","C7":"early/late dispersion contraction"}
M={(a,s):metrics(a,s) for a in ARMS for s in S}
print(f"  {'':4s} {'quantity':34s} {'NO-ROI':>9s} {'ROI-Q10':>9s} {'paired':>9s} {'sd':>7s} {'|m|/sd':>7s} {'dir':>5s}")
sep=[]
for c in ["C1","C2","C3","C4","C5","C6","C7"]:
    a0=np.array([M[("NO-ROI",s)][c] for s in S],float); a1=np.array([M[("ROI-Q10",s)][c] for s in S],float)
    ok=~(np.isnan(a0)|np.isnan(a1)); d=a1[ok]-a0[ok]
    if len(d)<2: print(f"  {c}  {NAME[c]:34s} insufficient"); continue
    e=abs(d.mean())/d.std(ddof=1) if d.std(ddof=1)>0 else float("nan")
    nd=int((d>0).sum()) if d.mean()>0 else int((d<0).sum())
    flag=" <= separable" if (np.isfinite(e) and e>=1.0) else ""
    if np.isfinite(e) and e>=1.0: sep.append((c,e))
    print(f"  {c}  {NAME[c]:34s} {a0[ok].mean():>9.3f} {a1[ok].mean():>9.3f} {d.mean():>+9.3f} {d.std(ddof=1):>7.3f} {e:>7.2f} {nd:>3d}/{len(d)}{flag}")
print(f"\n  separable at |mean|/sd >= 1.0: {len(sep)} of 7 -> {[c for c,_ in sep] if sep else 'NONE'}")
