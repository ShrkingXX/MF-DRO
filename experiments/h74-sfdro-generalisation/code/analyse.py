"""H74 verdict. Bars taken verbatim from protocol.md.

Written while the Borehole arm's 7 files existed on disk but WITHOUT inspecting
any of their values -- the script was authored from the protocol alone. Two gates
must clear before anything is reported:

  1. COMPLETENESS -- all 10 seeds present for the benchmark being judged.
  2. REPRODUCTION CONTROL -- h74's worker on Currin seed 44 must reproduce h59's
     published value bit-for-bit. h74 reuses seeds 44/46/48 from h59, so without
     this the 10-seed mean mixes two code paths. h73's "built-in control" was
     VACUOUS for exactly this reason (it compared h59's files to themselves), so
     the gate is enforced here in code rather than left to judgement.
"""
import sys,json,os; sys.path.insert(0,os.getcwd())
import numpy as np
from benchmarks import get_benchmark
S=list(range(42,52))
H74="experiments/h74-sfdro-generalisation/results"
H59="experiments/h59-sfdro-baseline/results"
H72="experiments/h72-n3-calibration/results"

def sfdro(b,s):
    for d in (H74,H59):
        p=f"{d}/{b}__SF-DRO__seed{s}.json"
        if os.path.exists(p): return json.load(open(p))["final_regret"]
    return None
def sfmes(b,s):
    p=f"{H72}/{b}__SF-MES__seed{s}.json"
    return json.load(open(p))["final_regret"] if os.path.exists(p) else None

# ---- Gate 2: reproduction control ----
cp=f"{H74}/Currin_2D__SF-DRO__seed44.json"; rp=f"{H59}/Currin_2D__SF-DRO__seed44.json"
if not os.path.exists(cp):
    print("  REPRODUCTION CONTROL NOT RUN -- h74 worker on Currin seed44 absent.")
    print("  WITHHELD. h74's PRIMARY is not reported without it."); sys.exit(0)
a=json.load(open(cp))["final_regret"]; bref=json.load(open(rp))["final_regret"]
if abs(a-bref)>=1e-9:
    print(f"  REPRODUCTION CONTROL FAILED: {a:.10f} vs h59 {bref:.10f}, diff {abs(a-bref):.3e}")
    print("  h74's 7 new seeds are NOT comparable to h59's 3. WITHHELD."); sys.exit(1)
print(f"  reproduction control PASS (Currin s44 {a:.10f} == h59, diff {abs(a-bref):.3e})\n")

for b,label in (("Borehole_8D","PRIMARY"),("Currin_2D","SECONDARY")):
    fs=abs(float(get_benchmark(f"{b}_HF")["known_optimal_value"]))
    D=[sfdro(b,s) for s in S]; M=[sfmes(b,s) for s in S]
    miss=[s for s,x,y in zip(S,D,M) if x is None or y is None]
    if miss:
        print(f"  {b} [{label}] INCOMPLETE -- missing seeds {miss}. WITHHELD.\n"); continue
    d=np.array(D,float)/fs*100; m=np.array(M,float)/fs*100
    gap=m.mean()-d.mean(); wins=int((d<m).sum())
    print(f"  {b} [{label}]")
    print(f"    SF-DRO {d.mean():>6.2f}%  sd {d.std(ddof=1):>5.2f}   SF-MES {m.mean():>6.2f}%  sd {m.std(ddof=1):>5.2f}")
    print(f"    gap {gap:+.2f} pts   SF-DRO wins {wins}/10")
    try:
        from scipy.stats import wilcoxon
        print(f"    Wilcoxon p = {wilcoxon(d,m).pvalue:.4f}  (reported unconditionally)")
    except Exception as e: print(f"    (wilcoxon: {e})")
    if b=="Borehole_8D":
        p1 = gap>=2.0 and wins>=7
        print(f"    1 PRIMARY  >=2.0 pts AND >=7/10: -> {'MET' if p1 else 'NOT MET'}")
        print(f"    3 NULL     <2.0 pts or <=6/10  -> "
              f"{'FIRED -- h73 is HARTMANN-SPECIFIC, one benchmark of three' if (gap<2.0 or wins<=6) else 'no'}")
        print(f"    4 REVERSED SF-MES ahead by >=2.0 -> {'FIRED' if -gap>=2.0 else 'no'}")
    else:
        print(f"    2 SECONDARY within 1.0 pt: |{gap:+.2f}| -> {'MET' if abs(gap)<=1.0 else 'NOT MET'}")
    print()
