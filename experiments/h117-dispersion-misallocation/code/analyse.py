"""H117 registered analysis. P1-P3 (dispersion misallocation), P4-P6 (off-boundary
waste, Amendment 1). Amendment 2 narrows what a PASS may conclude: MF-MES refines
with box-constrained L-BFGS-B and lands on active constraints by construction."""
import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import json, sys, numpy as np
sys.path.insert(0,"tools"); sys.path.insert(0,".")
from perdim import agg
from benchmarks import get_benchmark
R="experiments/h117-dispersion-misallocation/results"; B="Borehole_8D"; S=[52,53,54,55,56]; NFLOOR=15
hf=get_benchmark(f"{B}_HF"); lo=np.array(hf["domain_min"],float); wd=np.array(hf["domain_max"],float)-lo
def Z(m,s):
    d=json.load(open(f"{R}/{B}__{m}__seed{s}.json"))
    X=np.array([q["x"] for q in d["queries"] if q["fid"]==1 and not q.get("is_init",False)],float)
    return (X-lo)/wd
print("=== H117 registered analysis (raw, @cost_curve 200 not used: these are query-space stats) ===")
rows=[];excl=0
for s in S:
    zd,zm=Z("MF-DRO",s),Z("MF-MES",s)
    if len(zd)<NFLOOR or len(zm)<NFLOOR: excl+=1; print(f"  EXCLUDED seed{s}: n={len(zd)}/{len(zm)} < {NFLOOR}"); continue
    sd_d,sd_m=zd.std(axis=0,ddof=1),zm.std(axis=0,ddof=1)
    wR=agg(sd_d,B)/agg(sd_m,B); uR=sd_d.sum()/sd_m.sum()
    rows.append((s,wR,uR,wR/uR,(zd[:,0]<0.9).mean(),(zm[:,0]<0.9).mean(),len(zd),len(zm)))
print(f"\n  {'seed':5s} {'wR':>7s} {'uR':>7s} {'wR/uR':>7s} {'DRO off-bnd':>12s} {'MES off-bnd':>12s} {'nDRO':>5s} {'nMES':>5s}")
for r in rows: print(f"  {r[0]:<5d} {r[1]:7.3f} {r[2]:7.3f} {r[3]:7.3f} {100*r[4]:11.1f}% {100*r[5]:11.1f}% {r[6]:5d} {r[7]:5d}")
ratio=np.array([r[3] for r in rows]); L=np.log(ratio)
n_ge2=int((ratio>=2.0).sum()); eff=abs(L.mean())/L.std(ddof=1)
print(f"\n  P1 wR/uR >= 2.0 in >=4/5 : {n_ge2}/{len(rows)}  -> {'PASS' if n_ge2>=4 else 'FAIL'}")
print(f"  P2 |mean log|/sd >= 2.0  : {eff:.2f}      -> {'PASS' if eff>=2.0 else 'FAIL'}")
print(f"  P3 direction wR > uR     : {int((ratio>1).sum())}/{len(rows)} -> {'PASS' if (ratio>1).sum()>=4 else 'FAIL'}")
od=np.array([r[4] for r in rows]); om=np.array([r[5] for r in rows]); d=od-om
e4=abs(d.mean())/d.std(ddof=1) if d.std(ddof=1)>0 else float('inf')
print(f"\n  P4 DRO off-bnd > MES in 5/5 : {int((d>0).sum())}/{len(rows)} -> {'PASS' if (d>0).sum()==len(rows) else 'FAIL'}")
print(f"  P5 DRO mean off-bnd >= 3%   : {100*od.mean():.1f}% -> {'PASS' if od.mean()>=0.03 else 'FAIL'}")
print(f"  P6 paired |mean|/sd >= 1.0  : {e4:.2f} -> {'PASS' if e4>=1.0 else 'FAIL'}")
print(f"\n  excluded by n floor: {excl}/5")
print("  AMENDMENT 2: a PASS shows MF-DRO reproducibly fails to reach a boundary optimum and")
print("  pays HF budget for it. It does NOT license 'the DT is boundary-averse' or 'MF-MES")
print("  searches better' -- MF-MES refines with box-constrained L-BFGS-B (mf_mes_takeno.py:297).")
print("  h118 further showed the waste does not predict regret.")
