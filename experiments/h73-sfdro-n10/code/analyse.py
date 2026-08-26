"""H73 verdict. Written and committed while h73 was at 0/7, before any number.
Refuses to report until all 10 seeds are present. Bars locked in protocol.md."""
import sys,json,os; sys.path.insert(0,os.getcwd())
import numpy as np
from benchmarks import get_benchmark
B="Hartmann_6D"; S=list(range(42,52))
fs=abs(float(get_benchmark(f"{B}_HF")["known_optimal_value"]))
DIRS=["experiments/h73-sfdro-n10/results","experiments/h59-sfdro-baseline/results"]
def sfdro(s):
    for d in DIRS:
        p=f"{d}/{B}__SF-DRO__seed{s}.json"
        if os.path.exists(p): return json.load(open(p))["final_regret"]/fs*100
    return None
def sfmes(s):
    p=f"experiments/h72-n3-calibration/results/{B}__SF-MES__seed{s}.json"
    return json.load(open(p))["final_regret"]/fs*100 if os.path.exists(p) else None
D=[sfdro(s) for s in S]; M=[sfmes(s) for s in S]
miss=[s for s,a,b in zip(S,D,M) if a is None or b is None]
if miss:
    print(f"  INCOMPLETE -- SF-DRO missing seeds {[s for s,a in zip(S,D) if a is None]}, "
          f"SF-MES missing {[s for s,b in zip(S,M) if b is None]}")
    print("  WITHHELD. Verdict not evaluated."); sys.exit(0)
d=np.array(D); m=np.array(M)
gap=m.mean()-d.mean(); wins=int((d<m).sum())
print(f"  H73 -- SF-DRO vs SF-MES on Hartmann, n=10\n")
print(f"  {'seed':>5}{'SF-DRO':>10}{'SF-MES':>10}   winner")
for s,a,b in zip(S,d,m): print(f"  {s:>5}{a:>9.2f}%{b:>9.2f}%   {'SF-DRO' if a<b else 'SF-MES'}")
print(f"\n  SF-DRO  mean {d.mean():.2f}%  sd {d.std(ddof=1):.2f}  worst {d.max():.2f}%")
print(f"  SF-MES  mean {m.mean():.2f}%  sd {m.std(ddof=1):.2f}  worst {m.max():.2f}%")
print(f"  gap {gap:+.2f} pts   SF-DRO wins {wins}/10")
try:
    from scipy.stats import wilcoxon
    print(f"  Wilcoxon signed-rank p = {wilcoxon(d,m).pvalue:.4f}   (reported unconditionally)")
except Exception as e: print(f"  (wilcoxon unavailable: {e})")
i3=[S.index(x) for x in (44,46,48)]
print(f"\n  built-in control (seeds 44/46/48 must reproduce h59's n=3):")
print(f"    SF-DRO {d[i3].mean():.2f}% (h59 said 11.49)   SF-MES {m[i3].mean():.2f}% (h59 said 21.39)")
print("\n  LOCKED PREDICTIONS")
p1 = gap>=5.0 and wins>=7
print(f"  1 PRIMARY   >=5.0 pts AND >=7/10 wins: got {gap:+.2f} pts, {wins}/10 -> {'MET' if p1 else 'NOT MET'}")
print(f"  2 SECONDARY SF-DRO sd below SF-MES: {d.std(ddof=1):.2f} vs {m.std(ddof=1):.2f} -> "
      f"{'MET' if d.std(ddof=1)<m.std(ddof=1) else 'NOT MET'}")
p3 = gap<5.0 or wins<=6
print(f"  3 NULL      <5.0 pts or <=6/10 -> "
      f"{'FIRED -- the LAST pro-DRO result is WITHDRAWN, lesson 26 goes 4 for 4' if p3 else 'no'}")
print(f"  4 REVERSED  SF-MES ahead by >=5.0: {'FIRED' if -gap>=5.0 else 'no'}")
