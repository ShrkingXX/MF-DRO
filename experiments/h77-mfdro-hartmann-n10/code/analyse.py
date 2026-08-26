"""H75 verdict. Bars verbatim from protocol.md. Written at 0/7, before any number.

Gates enforced in code, not judgement:
  1. COMPLETENESS -- all 10 seeds present.
  2. REPRODUCTION CONTROL -- h75's worker on Borehole seed 44 must reproduce
     h57's published value bit-for-bit. h75 reuses seeds 44/46/48 from h57, so
     without this the 10-seed mean mixes code paths. h73's built-in control was
     vacuous for exactly this reason.
"""
import sys,json,os,itertools; sys.path.insert(0,os.getcwd())
import numpy as np
from benchmarks import get_benchmark
B="Hartmann_6D"; S=list(range(42,52)); PUB=14.68
fs=abs(float(get_benchmark(f"{B}_HF")["known_optimal_value"]))
H77="experiments/h77-mfdro-hartmann-n10/results"; H57="experiments/h57-baseline-comparison/results"

cp=f"{H77}/{B}__MF-DRO__seed44.json"; rp=f"{H57}/{B}__MF-DRO__seed44.json"
if not os.path.exists(cp):
    print("  REPRODUCTION CONTROL NOT RUN (h77 worker on Hartmann seed44 absent).")
    print("  WITHHELD -- verdict not reported without it."); sys.exit(0)
a=json.load(open(cp))["final_regret"]; b=json.load(open(rp))["final_regret"]
if abs(a-b)>=1e-9:
    print(f"  REPRODUCTION CONTROL FAILED: {a:.10f} vs h57 {b:.10f}, diff {abs(a-b):.3e}")
    print("  h77's new seeds are NOT comparable to h57's. WITHHELD."); sys.exit(1)
print(f"  reproduction control PASS (Borehole s44 {a:.10f} == h57, diff {abs(a-b):.3e})\n")

v=[]
for s in S:
    hit=None
    for d in (H75,H57):
        p=f"{d}/{B}__MF-DRO__seed{s}.json"
        if os.path.exists(p): hit=json.load(open(p))["final_regret"]/fs*100; break
    v.append(hit)
miss=[s for s,x in zip(S,v) if x is None]
if miss:
    print(f"  INCOMPLETE -- missing seeds {miss}. WITHHELD."); sys.exit(0)
v=np.array(v); sub=np.array([v[list(c)].mean() for c in itertools.combinations(range(10),3)])
shift=v.mean()-PUB; span=sub.max()-sub.min()
print(f"  MF-DRO Borehole n=10: mean {v.mean():.2f}%  sd {v.std(ddof=1):.2f}")
print(f"  per seed: {[round(x,1) for x in v]}")
print(f"  published n=3 (seeds 44/46/48): {PUB:.2f}%   shift {shift:+.2f} pts")
print(f"  C(10,3) three-seed range: [{sub.min():.2f}, {sub.max():.2f}]  span {span:.2f}\n")
print("  LOCKED PREDICTIONS")
print(f"  1 PRIMARY   3-seed span >= 8.0 pts: {span:.2f} -> {'MET' if span>=8.0 else 'NOT MET'}")
print(f"  2 SECONDARY |shift| >= 2.0 pts: {abs(shift):.2f} -> {'MET' if abs(shift)>=2.0 else 'NOT MET'}")
print(f"  3 NULL      span <8.0 AND |shift| <2.0 -> "
      f"{'FIRED -- Hartmann is stable for MF-DRO though unstable for every other method; published entry stands' if (span<8.0 and abs(shift)<2.0) else 'no'}")
print(f"\n  reference Hartmann 3-seed spans: SF-DRO 9.39, SF-MES 10.13, SF-EI 15.32, MI-Greedy 26.07, GP-UCB 51.83")
print(f"  reference Borehole spans (stable): 3.11-5.89, and MF-DRO's Borehole shift was -0.82")
print(f"  reference Hartmann n=10 means:   SF-DRO 8.46, MF-MES 8.24, SF-EI 18.61, SF-MES 21.17, MI-Greedy 36.61")
