import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import json, numpy as np
S=[42,43,44,45,46]; B="Borehole_8D"; OPT=309.5755876604079
def at200(p):
    d=json.load(open(p)); hr=np.array(d["hf_regret_curve"],float); c=np.array(d["cost_curve"],float)
    return float(np.interp(200.0,c,hr))
src={"control (h83 MF-DRO)": [f"experiments/h83-main-comparison/results/{B}__MF-DRO__seed{s}.json" for s in S],
     "tight  q=0.100":       [f"experiments/h84-roi-strategy/results/{B}__ROI-Q10__seed{s}.json" for s in S],
     "loose  q=0.493":       [f"experiments/h84-roi-strategy/results/{B}__ROI-ANN__seed{s}.json" for s in S]}
V={k:np.array([at200(f) for f in fs]) for k,fs in src.items()}
print("  raw regret @cost_curve 200, per seed")
for k,v in V.items(): print(f"    {k:22s} {np.round(v,2)}   mean {v.mean():7.3f}")
print()
ctl=V["control (h83 MF-DRO)"]
for k in ["tight  q=0.100","loose  q=0.493"]:
    d=V[k]-ctl; e=abs(d.mean())/d.std(ddof=1)
    worse=int((d>0).sum())
    print(f"  {k:16s} vs control: paired {d.mean():+7.3f}  sd {d.std(ddof=1):6.3f}  |m|/sd {e:5.2f}"
          f"  worse in {worse}/5  better in {5-worse}/5")
d=V["loose  q=0.493"]-ctl; e=abs(d.mean())/d.std(ddof=1); worse=int((d>0).sum())
p1 = (worse>=4) and np.isfinite(e) and e>=1.0
p2 = 2.0 <= d.mean() <= 8.0
print(f"\n  P1 (loose WORSE than control, >=4/5 and effect>=1.0): {'PASS' if p1 else 'FAIL'}")
print(f"  P2 (magnitude in +2..+8 pts, composition holds):      {'PASS' if p2 else 'FAIL'}  (got {d.mean():+.3f})")
print(f"\n  in frozen units (% of optimum {OPT:.1f}): loose vs control {100*d.mean()/OPT:+.3f}%  sd {100*d.std(ddof=1)/OPT:.3f}%")
print(f"  reference: tight vs loose = {(V['loose  q=0.493']-V['tight  q=0.100']).mean():+.3f} (h125 reported +9.018)")
