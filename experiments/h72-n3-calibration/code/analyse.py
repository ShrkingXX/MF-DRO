"""H72 verdict. Written and committed while the run was INCOMPLETE (76/120).
Refuses to report until every cell is present. Bars locked in protocol.md."""
import sys,json,os,itertools; sys.path.insert(0,os.getcwd())
import numpy as np
from benchmarks import get_benchmark
S=list(range(42,52)); R="experiments/h72-n3-calibration/results"
BEN=("Currin_2D","Hartmann_6D","Borehole_8D")
METH=("MF-MI-Greedy","MF-GP-UCB","SF-MES","SF-EI")
fs={b:abs(float(get_benchmark(f"{b}_HF")["known_optimal_value"])) for b in BEN}

def rel(b,m):
    v=[]
    for s in S:
        p=f"{R}/{b}__{m}__seed{s}.json"
        if not os.path.exists(p): return None
        v.append(json.load(open(p))["final_regret"]/fs[b]*100)
    return np.array(v)

data={(b,m):rel(b,m) for b in BEN for m in METH}
missing=[f"{b}/{m}" for (b,m),v in data.items() if v is None]
if missing:
    print(f"  INCOMPLETE -- {len(missing)} cells missing: {missing[:8]}"
          f"{' ...' if len(missing)>8 else ''}")
    print("  WITHHELD. Verdict not evaluated."); sys.exit(0)

print("  H72 -- how much can n=3 mislead? All C(10,3)=120 three-seed subsets.\n")
print(f"  {'benchmark':<13}{'method':<15}{'n=10 mean':>10}{'3-seed min':>12}{'3-seed max':>12}{'range':>9}")
rng={}
for b in BEN:
    for m in METH:
        v=data[(b,m)]
        sub=np.array([v[list(c)].mean() for c in itertools.combinations(range(10),3)])
        rng[(b,m)]=(v.mean(),sub.min(),sub.max())
        print(f"  {b:<13}{m:<15}{v.mean():>9.2f}%{sub.min():>11.2f}%{sub.max():>11.2f}%"
              f"{sub.max()-sub.min():>8.2f}")
    print()
widest=max(rng.items(),key=lambda kv:kv[1][2]-kv[1][1])
w=widest[1][2]-widest[1][0]+widest[1][0]-widest[1][1]
print("  LOCKED PREDICTIONS")
maxrange=max(v[2]-v[1] for v in rng.values())
print(f"  1 PRIMARY   some cell's 3-seed range >=5.0 pts: max={maxrange:.2f} "
      f"({widest[0][0]}/{widest[0][1]}) -> {'MET' if maxrange>=5.0 else 'NOT MET'}")
overlaps=[]
for b in BEN:
    for m1,m2 in itertools.combinations(METH,2):
        a,c=rng[(b,m1)],rng[(b,m2)]
        ordered = "<" if a[0]<c[0] else ">"
        if not (a[2]<c[1] or c[2]<a[1]):
            overlaps.append((b,m1,m2,ordered,a,c))
print(f"  2 SECONDARY some asserted ordering unresolved at n=3: "
      f"{len(overlaps)} overlapping pairs -> {'MET' if overlaps else 'NOT MET'}")
for b,m1,m2,o,a,c in overlaps[:6]:
    print(f"      {b:<13}{m1} {o} {m2}  ranges [{a[1]:.1f},{a[2]:.1f}] vs [{c[1]:.1f},{c[2]:.1f}]")
print(f"  3 NULL      all <5.0 and no overlap -> "
      f"{'FIRED -- n=3 was adequate' if maxrange<5.0 and not overlaps else 'no'}")
