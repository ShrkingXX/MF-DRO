import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import json, numpy as np
CK="experiments/h83-main-comparison/results/ckpt"
BENCH=["Hartmann_6D","Borehole_8D","Currin_2D","Ackley_10D"]; METH=["MF-DRO","MF-MES"]; S=[42,43,44,45,46]
def cell(b,m,s):
    d=json.load(open(f"{CK}/{b}__{m}__seed{s}.json")); Q=d["queries"]
    init=[q["y"] for q in Q if q["fid"]==1 and q.get("is_init")]
    hf=[q["y"] for q in Q if q["fid"]==1 and not q.get("is_init")]
    if not init or not hf: return None
    i=np.array(init,float); h=np.array(hf,float); sd=i.std(ddof=1) if len(i)>1 else 1.0
    return dict(waste=float((h<i.max()).mean()), score=float((h.mean()-i.mean())/(sd if sd>0 else 1.0)), n=len(h))
R={(b,m):[cell(b,m,s) for s in S] for b in BENCH for m in METH}
print(f"  {'benchmark':12s} {'method':7s} {'waste_frac per seed':38s} {'median':>7s} {'score':>7s} {'n_HF':>6s}")
for b in BENCH:
    for m in METH:
        c=[x for x in R[(b,m)] if x]
        w=[x["waste"] for x in c]
        print(f"  {b:12s} {m:7s} {str([round(100*v,1) for v in w]):38s} {100*np.median(w):6.1f}% {np.mean([x['score'] for x in c]):7.3f} {np.mean([x['n'] for x in c]):6.1f}")
    print()
hD=[x["waste"] for x in R[("Hartmann_6D","MF-DRO")] if x]; hM=[x["waste"] for x in R[("Hartmann_6D","MF-MES")] if x]
bD=[x["waste"] for x in R[("Borehole_8D","MF-DRO")] if x]
print(f"  P1  Hartmann MF-DRO waste median = {100*np.median(hD):.1f}%   recorded 20.8%   "
      f"in 10.8-30.8%? {'PASS' if 10.8<=100*np.median(hD)<=30.8 else 'FAIL'}")
print(f"  P2  Hartmann {100*np.median(hD):.1f}% > Borehole {100*np.median(bD):.1f}%? "
      f"{'PASS' if np.median(hD)>np.median(bD) else 'FAIL'}")
d=np.array(hD)-np.array(hM); n=int((d>0).sum())
print(f"  P3  MF-DRO > MF-MES on Hartmann in {n}/5 seeds  (paired mean {100*d.mean():+.1f} pts)  {'PASS' if n>=4 else 'FAIL'}")
print(f"\n  founding-diagnosis score check (Hartmann): MF-DRO {np.mean([x['score'] for x in R[('Hartmann_6D','MF-DRO')] if x]):.3f}"
      f"  MF-MES {np.mean([x['score'] for x in R[('Hartmann_6D','MF-MES')] if x]):.3f}   (recorded 0.336 vs 0.747)")
