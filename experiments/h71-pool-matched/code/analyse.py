"""H71 verdict. Written at 1/6, before any h71 regret was inspected.
Refuses to report until all 6 cells are present. Bars locked in protocol.md
(each names a magnitude -- h68 and h65 both passed direction-only bars while the
substantive claim failed)."""
import sys,json,os; sys.path.insert(0,os.getcwd())
import numpy as np
from benchmarks import get_benchmark
BEN=("Borehole_8D","Hartmann_6D"); S=(44,46,48)
fs={b:abs(float(get_benchmark(f"{b}_HF")["known_optimal_value"])) for b in BEN}
def get(b,arm,dirs):
    v=[]
    for s in S:
        hit=None
        for d in dirs:
            p=f"experiments/{d}/results/{b}__{arm}__seed{s}.json"
            if os.path.exists(p): hit=json.load(open(p))["final_regret"]/fs[b]*100; break
        v.append(hit)
    return v
P={b:get(b,"POOL1000",["h71-pool-matched"]) for b in BEN}
B={b:get(b,"MF-DRO",["h57-baseline-comparison"]) for b in BEN}
miss=[(b,s) for b in BEN for s,v in zip(S,P[b]) if v is None]
if miss:
    print(f"  INCOMPLETE -- POOL1000 missing {miss}")
    print("  WITHHELD. Verdict not evaluated."); sys.exit(0)
print("  H71 -- MF-DRO teacher pool 200 -> 1000, vs h57 BASE\n")
print(f"  {'benchmark':<13}{'seed':>5}{'BASE':>9}{'POOL1000':>11}   winner")
out={}
for b in BEN:
    p=np.array(P[b],float); q=np.array(B[b],float)
    for s,a,c in zip(S,q,p):
        print(f"  {b:<13}{s:>5}{a:>8.2f}%{c:>10.2f}%   {'POOL1000' if c<a else 'BASE'}")
    out[b]=(q.mean(),p.mean(),int((p<q).sum()))
    print(f"  {'':13}{'mean':>5}{q.mean():>8.2f}%{p.mean():>10.2f}%   POOL1000 wins {out[b][2]}/3\n")
bb,bp,bw=out["Borehole_8D"]; hb,hp,hw=out["Hartmann_6D"]
print("  LOCKED PREDICTIONS")
p1 = (bb-bp)>=2.0 and bw>=2
print(f"  1 PRIMARY   Borehole >=2.0 pts AND >=2/3: got {bb-bp:+.2f} pts, {bw}/3 -> {'MET' if p1 else 'NOT MET'}")
print(f"  2 SECONDARY POOL1000 stays above 12% on Borehole: {bp:.2f}% -> "
      f"{'MET (pool is NOT the whole story for MF-DRO)' if bp>12 else 'NOT MET'}")
p3 = (bb-bp)<2.0
print(f"  3 NULL      movement <2.0 pts -> "
      f"{'FIRED -- MF-DRO teacher pool NOT load-bearing though the baselines inference pool is; h61 POOL600 gain was noise' if p3 else 'no'}")
print(f"  4 CONTROL   Hartmann not worse by >2.0 pts: {hb-hp:+.2f} -> "
      f"{'MET' if (hb-hp)>=-2.0 else 'VIOLATED'}")
print(f"\n  reference: MI-Greedy Borehole 8.27% (n=3) / 9.29% (n=10)")
