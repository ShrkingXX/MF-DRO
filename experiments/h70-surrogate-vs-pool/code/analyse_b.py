"""H70b verdict. Locked bars in protocol_b.md, written before any n=10 data."""
import sys,json,os; sys.path.insert(0,os.getcwd())
import numpy as np
from benchmarks import get_benchmark
S=list(range(42,52)); R="experiments/h70-surrogate-vs-pool/results"
BEN=("Hartmann_6D","Borehole_8D")
fs={b:abs(float(get_benchmark(f"{b}_HF")["known_optimal_value"])) for b in BEN}
def rel(b,a):
    out=[]
    for s in S:
        p=f"{R}/{b}__{a}__seed{s}.json"
        out.append(json.load(open(p))["final_regret"]/fs[b]*100 if os.path.exists(p) else np.nan)
    return np.array(out)
print("  H70b -- KO-style GP vs plain builder, SF-EI loop, n=10 (seeds 42-51)\n")
res={}
for b in BEN:
    ko,alt=rel(b,"SF-EI"),rel(b,"ALTGP")
    miss=[s for s,v in zip(S,ko+alt) if not np.isfinite(v)]
    m=np.isfinite(ko)&np.isfinite(alt); ko,alt=ko[m],alt[m]
    wins=int((alt<ko).sum()); d=ko.mean()-alt.mean()
    res[b]=(ko,alt,wins,d)
    print(f"  {b}   (n={m.sum()}{', MISSING '+str(miss) if miss else ''})")
    print(f"    KO-style  mean {ko.mean():>6.2f}%  sd {ko.std(ddof=1):>5.2f}  per-seed {[round(x,1) for x in ko]}")
    print(f"    ALTGP     mean {alt.mean():>6.2f}%  sd {alt.std(ddof=1):>5.2f}  per-seed {[round(x,1) for x in alt]}")
    print(f"    ALTGP better by {d:+.2f} pts, wins {wins}/{m.sum()}\n")
hk,ha,hw,hd=res["Hartmann_6D"]; bk,ba,bw,bd=res["Borehole_8D"]
n=len(hk)
print("  LOCKED PREDICTIONS")
p1 = hd>=3.0 and hw>=7
print(f"  1 PRIMARY   Hartmann >=3.0 pts AND >=7/10 wins: got {hd:+.2f} pts, {hw}/{n} -> {'MET' if p1 else 'NOT MET'}")
print(f"  2 SECONDARY Borehole |diff| < 1.0 pt: got {abs(bd):.2f} -> {'MET' if abs(bd)<1.0 else 'NOT MET'}")
p3 = (hd<3.0) or (hw<=6)
print(f"  3 NULL      <3.0 pts or <=6/10 wins -> {'FIRED -- h70 6.41 was an n=3 artifact, WITHDRAW' if p3 else 'no'}")
print(f"  4 REVERSED  KO-style better by >=3.0: {'FIRED' if -hd>=3.0 else 'no'}")
h3=[i for i,s in enumerate(S) if s in (44,46,48)]
print(f"\n  built-in control (seeds 44/46/48 must reproduce h70's n=3):")
print(f"    KO-style {hk[h3].mean():.2f}% (h70 said 19.89)   ALTGP {ha[h3].mean():.2f}% (h70 said 13.48)")
