"""H126 registered analysis: is there anything BELOW the plateau?
GATE G1 (blocking, per-run): accept_frac in [0.018,0.022] AND n_distinct==600.
P1 (locked): q=0.02 does NOT differ from q=0.10 at >=4/5 seeds with |m|/sd>=1.0.
Read point: raw regret @cost_curve 200; also reported as rel% of optimum."""
import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import json, numpy as np
S=[42,43,44,45,46]; B="Borehole_8D"; OPT=309.5755876604079
Q02="experiments/h126-q02-below-plateau/results/%s__ROI-Q02__seed%%d.json" % B
Q10="experiments/h84-roi-strategy/results/%s__ROI-Q10__seed%%d.json" % B
CTL="experiments/h83-main-comparison/results/%s__MF-DRO__seed%%d.json" % B
def at(p,t=200.0):
    d=json.load(open(p)); hr=np.array(d["hf_regret_curve"],float); c=np.array(d["cost_curve"],float)
    return float(np.interp(t,c,hr))
def roi(p):
    r=json.load(open(p)).get("roi_summary") or {}
    return r.get("accept_frac"), r.get("n_distinct"), r.get("n_draws")
print("=== GATE G1: observed acceptance (blocking, per run) ===")
ok=[]
for s in S:
    a,nd,ndr=roi(Q02 % s)
    g=(a is not None) and (0.018<=a<=0.022) and (nd==600)
    ok.append(g)
    print(f"  seed{s}: accept_frac={a if a is None else round(a,5)}  n_distinct={nd}  n_draws={None if ndr is None else round(ndr,1)}  -> {'PASS' if g else 'FAIL'}")
kept=[s for s,g in zip(S,ok) if g]
print(f"  G1: {sum(ok)}/5 pass; runs failing G1 are EXCLUDED from the statistic (protocol).")
if len(kept)<2:
    print("  Too few runs clear G1 -- no verdict issued."); raise SystemExit(0)
a=np.array([at(Q02 % s) for s in kept]); b=np.array([at(Q10 % s) for s in kept]); c=np.array([at(CTL % s) for s in kept])
d=a-b; e=abs(d.mean())/d.std(ddof=1) if d.std(ddof=1)>0 else float("nan")
n_better=int((d<0).sum())
print(f"\n=== P1: q=0.02 vs q=0.10 (n={len(kept)} seeds clearing G1) ===")
print(f"  q=0.02 mean regret {a.mean():7.3f}   ({100*a.mean()/OPT:.3f}% of optimum)")
print(f"  q=0.10 mean regret {b.mean():7.3f}   ({100*b.mean()/OPT:.3f}%)")
print(f"  control mean       {c.mean():7.3f}   ({100*c.mean()/OPT:.3f}%)")
print(f"  paired q02-q10 {d.mean():+7.3f} raw = {100*d.mean()/OPT:+.3f}%  sd {d.std(ddof=1):.3f}  |m|/sd {e:.2f}  q02 better {n_better}/{len(kept)}")
sep = (n_better>=4 or (len(kept)-n_better)>=4) and np.isfinite(e) and e>=1.0
# Verdict NAMES are part of the registration. "CONFIRMED" for a predicted null
# reads as established equality when it means only that n=5 cannot distinguish
# them -- the most flattering reading, produced by wording rather than data.
# Renamed before any h126 result existed. (Peer's h137 lesson: their gate
# partitioned correctly and still returned the word "TIED" for "not separable".)
print(f"\n  P1 predicted NO difference -> {'REFUTED: they separate' if sep else 'NOT SEPARABLE at n=5 (this is NOT evidence of equality)'}")
print(f"  (P2 pre-committed readings: q02 BETTER = useful region extends below 0.05, needs fresh-seed")
print(f"   confirmation; q02 WORSE = the curve is U-shaped and over-restriction has its own cost.)")
for lbl,v in [("q=0.02 vs control",a-c),("q=0.10 vs control",b-c)]:
    ee=abs(v.mean())/v.std(ddof=1)
    print(f"  {lbl}: {100*v.mean()/OPT:+.3f}%  effect {ee:.2f}  better {int((v<0).sum())}/{len(kept)}")
