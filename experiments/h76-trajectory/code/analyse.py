"""H76 verdict. Bars verbatim from protocol.md."""
import sys,json,os; sys.path.insert(0,os.getcwd())
import numpy as np
from benchmarks import get_benchmark
BEN=("Hartmann_6D","Borehole_8D","Currin_2D"); S=list(range(42,52))
fs={b:abs(float(get_benchmark(f"{b}_HF")["known_optimal_value"])) for b in BEN}
DD=["h73-sfdro-n10","h74-sfdro-generalisation"]
def curves(b,arm,dirs):
    out=[]
    for s in S:
        for d in dirs:
            p=f"experiments/{d}/results/{b}__{arm}__seed{s}.json"
            if os.path.exists(p):
                c=json.load(open(p)).get("regret_curve") or []
                if c: out.append(np.array(c,float)/fs[b]*100)
                break
    return out
# control
bad=sum(1 for b in BEN for s in S
        if abs(json.load(open(f"experiments/h76-trajectory/results/{b}__SF-MES__seed{s}.json"))["final_regret"]
             - json.load(open(f"experiments/h72-n3-calibration/results/{b}__SF-MES__seed{s}.json"))["final_regret"])>=1e-9)
if bad: print(f"  REPRODUCTION CONTROL FAILED ({bad} cells). WITHHELD."); sys.exit(1)
print(f"  reproduction control PASS (30/30 bit-for-bit vs h72)\n")
res={}
for b in BEN:
    D=curves(b,"SF-DRO",DD)
    M=[np.array(json.load(open(f"experiments/h76-trajectory/results/{b}__SF-MES__seed{s}.json"))["regret_curve"],float)/fs[b]*100
       for s in S]
    if not D: print(f"  {b}: no SF-DRO curves (h59 seeds have the empty-trace bug)"); continue
    L=min(min(len(x) for x in D), min(len(x) for x in M))
    d=np.mean([x[:L] for x in D],0); m=np.mean([x[:L] for x in M],0)
    res[b]=(d,m,L,len(D))
    cross=next((i+1 for i in range(L) if d[i] < m[-1]), None)
    print(f"  {b}  (SF-DRO n={len(D)} curves, SF-MES n=10, {L} iterations compared)")
    print(f"    SF-DRO final {d[-1]:>6.2f}%   SF-MES final {m[-1]:>6.2f}%")
    q=[0,int(L*0.25),int(L*0.5),int(L*0.75),L-1]
    print(f"    iter      " + "".join(f"{i+1:>8}" for i in q))
    print(f"    SF-DRO    " + "".join(f"{d[i]:>8.2f}" for i in q))
    print(f"    SF-MES    " + "".join(f"{m[i]:>8.2f}" for i in q))
    print(f"    crossing (SF-DRO below SF-MES's FINAL): iteration {cross if cross else 'never'} of {L}\n")
d,m,L,_=res["Hartmann_6D"]
cross=next((i+1 for i in range(L) if d[i] < m[-1]), None)
print("  LOCKED PREDICTIONS")
p1 = cross is not None and cross<=12
print(f"  1 PRIMARY   Hartmann crossing by iter 12 of 25: got {cross} -> {'MET' if p1 else 'NOT MET'}")
gap=m-d; h2=gap[L//2:]
mono=bool(np.all(np.diff(h2)>=-1e-9))
print(f"  2 SECONDARY gap non-decreasing over 2nd half: {'MET' if mono else 'NOT MET'}"
      f"  (gap {gap[L//2]:.2f} -> {gap[-1]:.2f})")
print(f"  3 NULL      gap opens only in last third (iter>=17): "
      f"{'FIRED' if (cross and cross>=17) else 'no'}")
for b in ("Borehole_8D","Currin_2D"):
    if b in res:
        dd,mm,LL,_=res[b]
        c2=next((i+1 for i in range(LL) if dd[i] < mm[-1]), None)
        print(f"  4 CONTROL   {b}: crossing at {c2 if c2 else 'never'} of {LL}")
