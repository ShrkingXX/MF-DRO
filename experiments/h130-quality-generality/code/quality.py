import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import json, numpy as np
S=[42,43,44,45,46]
ARM={"Borehole_8D":"experiments/h84-roi-strategy/results/Borehole_8D__ROI-Q10__seed%d.json",
     "Hartmann_6D":"experiments/h84-roi-strategy/results/Hartmann_6D__ROI-Q10__seed%d.json",
     "Currin_2D":  "experiments/h86-roi-full/results/Currin_2D__ROI-Q10__seed%d.json",
     "Ackley_10D": "experiments/h86-roi-full/results/Ackley_10D__ROI-Q10__seed%d.json"}
def hfy(p):
    d=json.load(open(p))
    return [q["y"] for q in d["queries"] if q["fid"]==1 and not q.get("is_init",False)]
print(f"  {'bench':12s} {'K per seed':26s} {'ctrl':>9s} {'ROI':>9s} {'delta':>9s} {'sd':>8s} {'|m|/sd':>7s} {'up':>5s} {'verdict':>10s}")
res={}
for b,pat in ARM.items():
    dm=[];du=[];Ks=[]
    for s in S:
        c=hfy(f"experiments/h83-main-comparison/results/{b}__MF-DRO__seed{s}.json"); a=hfy(pat % s)
        K=min(len(c),len(a)); Ks.append(K)
        dm.append(np.mean(a[:K])-np.mean(c[:K]))
        du.append(np.mean(a)-np.mean(c))
    dm=np.array(dm); du=np.array(du)
    e=abs(dm.mean())/dm.std(ddof=1); up=int((dm>0).sum())
    sep = (up>=4 or (5-up)>=4) and e>=1.0
    res[b]=(dm,du,e,up,sep)
    cm=np.mean([np.mean(hfy(f"experiments/h83-main-comparison/results/{b}__MF-DRO__seed{s}.json")[:Ks[i]]) for i,s in enumerate(S)])
    am=cm+dm.mean()
    print(f"  {b:12s} {str(Ks):26s} {cm:9.3f} {am:9.3f} {dm.mean():+9.3f} {dm.std(ddof=1):8.3f} {e:7.2f} {up:>3d}/5 {'SEPARATES' if sep else 'no':>10s}")
print()
print("  uncount-matched (confound visible):")
for b,(dm,du,e,up,sep) in res.items():
    print(f"    {b:12s} matched {dm.mean():+8.3f}   unmatched {du.mean():+8.3f}   diff {du.mean()-dm.mean():+8.3f}")
print()
bo=res["Borehole_8D"]; others={k:v for k,v in res.items() if k!="Borehole_8D"}
print(f"  P1 (Borehole reproduces): {'PASS' if bo[4] and bo[3]>=4 else 'FAIL'}  (effect {bo[2]:.2f}, up {bo[3]}/5)")
sep_others=[k for k,v in others.items() if v[4]]
print(f"  P2 (others do NOT separate): {'PASS -- none separate' if not sep_others else 'FAIL -- ' + ', '.join(sep_others)}")
