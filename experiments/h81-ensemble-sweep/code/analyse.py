"""H81 verdict. Bars taken verbatim from protocol.md, which was committed before
any h81 run existed. Written at 13/15; refuses to report until all 15 land."""
import sys,json,os,itertools; sys.path.insert(0,os.getcwd())
import numpy as np
from benchmarks import get_benchmark
B="Hartmann_6D"; SEEDS=[42,43,44,45,46]; ARMS=["M3","M5","M10"]
fs=abs(float(get_benchmark(f"{B}_HF")["known_optimal_value"]))
R="experiments/h81-ensemble-sweep/results"
def get(a):
    v=[]
    for s in SEEDS:
        p=f"{R}/{B}__{a}__seed{s}.json"
        v.append(json.load(open(p))["final_regret"]/fs*100 if os.path.exists(p) else None)
    return v
D={a:get(a) for a in ARMS}
miss=[(a,s) for a in ARMS for s,v in zip(SEEDS,D[a]) if v is None]
if miss:
    print(f"  INCOMPLETE -- missing {miss}")
    print("  WITHHELD. Verdict not evaluated."); sys.exit(0)
A={a:np.array(D[a],float) for a in ARMS}
print(f"  H81 -- MF-DRO ensemble size on Hartmann, n=5, budget 300\n")
print(f"  {'seed':>6}" + "".join(f"{a:>10}" for a in ARMS))
for i,s in enumerate(SEEDS):
    print(f"  {s:>6}" + "".join(f"{A[a][i]:>9.2f}%" for a in ARMS))
print(f"  {'mean':>6}" + "".join(f"{A[a].mean():>9.2f}%" for a in ARMS))
print(f"  {'sd':>6}" + "".join(f"{A[a].std(ddof=1):>10.2f}" for a in ARMS))
gap = A["M3"].mean() - A["M10"].mean()          # +ve = M3 worse
wins = int(((A["M3"] <= A["M10"])).sum())        # wins or ties
span = max(A[a].mean() for a in ARMS) - min(A[a].mean() for a in ARMS)
print(f"\n  M3 - M10 mean gap: {gap:+.2f} pts   M3 wins/ties {wins}/5   arm span {span:.2f} pts")
try:
    from scipy.stats import wilcoxon
    print(f"  Wilcoxon M3 vs M10 p = {wilcoxon(A['M3'],A['M10']).pvalue:.4f}  (n=5, reported unconditionally)")
except Exception as e: print(f"  (wilcoxon: {e})")
print("\n  LOCKED PREDICTIONS")
p1 = (gap < 2.0) and wins >= 2
print(f"  1 PRIMARY   M3 within 2.0 pts of M10 AND wins/ties >=2/5: "
      f"gap {gap:+.2f}, {wins}/5 -> {'MET' if p1 else 'NOT MET'}")
p2 = span < 3.0
print(f"  2 SECONDARY three arms' means span < 3.0 pts: {span:.2f} -> {'MET' if p2 else 'NOT MET'}")
p3 = (gap >= 2.0) or wins <= 1
print(f"  3 NULL      M3 worse by >=2.0 or wins <=1/5 -> "
      f"{'FIRED -- M does not shrink safely; default stays 10' if p3 else 'no'}")
print("\n  DECISION RULE (fixed in advance): if PRIMARY and SECONDARY both hold,")
print("  the default becomes the SMALLEST M within 2.0 pts of M=10.")
if p1 and p2:
    ok=[a for a in ARMS if A[a].mean()-A["M10"].mean() < 2.0]
    pick=min(ok,key=lambda a:int(a[1:]))
    print(f"  -> arms within 2.0 pts of M10: {ok}   SMALLEST = {pick}")
else:
    print("  -> not triggered; M stays 10.")
