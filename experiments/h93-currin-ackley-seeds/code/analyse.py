"""H93: do Currin's and Ackley's h83 deficits reproduce at fresh seeds 52-56?

Committed BEFORE any run, like h87's and h90's. Imports h83's own sr_curve/grid
so the metric is the frozen one, and follows h83's rule that a benchmark whose
optimum is exactly 0 reports ABSOLUTE simple regret (relative is undefined).
"""
import sys, os, json
import numpy as np
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,"experiments","h83-main-comparison","code"))
from analyse import sr_curve, grid
from benchmarks import get_benchmark
G=np.linspace(0,200,201); SEEDS=(52,53,54,55,56)
R=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results"))
# benchmark -> (its own best h83 baseline, that baseline's h83 mean, MF-DRO's h83 mean)
CASES={"Currin_2D":   ("MF-MI-Greedy", 0.00, 0.01),
       "Ackley_10D":  ("SF-DRO",       3.43, 3.83)}
def at200(p,opt):
    c,sr=sr_curve(json.load(open(p)),-abs(opt))
    v=float(grid(c,sr,G)[-1])
    return v if abs(opt)<1e-9 else 100.0*v/abs(opt)   # abs SR when f(x*)=0
if __name__=="__main__":
    print("H93 -- Currin and Ackley at FRESH seeds 52-56, vs each benchmark's OWN best baseline.\n")
    verdicts={}
    for b,(base,base83,dro83) in CASES.items():
        opt=float(get_benchmark(f"{b}_HF")["known_optimal_value"])
        unit="" if abs(opt)<1e-9 else "%"
        rows=[];miss=[]
        for s in SEEDS:
            a=os.path.join(R,f"{b}__MF-DRO__seed{s}.json"); c=os.path.join(R,f"{b}__{base}__seed{s}.json")
            if os.path.exists(a) and os.path.exists(c): rows.append((s,at200(a,opt),at200(c,opt)))
            else: miss.append(s)
        print(f"=== {b}  (MF-DRO vs {base})"
              + ("   [f(x*)=0 -> ABSOLUTE SR]" if abs(opt)<1e-9 else "") + " ===")
        if not rows: print(f"  no complete pairs yet (pending {miss})\n"); continue
        print(f"  {'seed':>5}{'MF-DRO':>10}{base[:9]:>11}{'diff':>9}")
        for s,a,c in rows: print(f"  {s:>5}{a:>10.2f}{c:>11.2f}{a-c:>+9.2f}")
        d=np.array([a-c for _,a,c in rows]); wins=int((d<0).sum())
        print(f"  n={len(d)}  paired mean {d.mean():+.2f}{unit}  "
              f"sd {d.std(ddof=1) if len(d)>1 else float('nan'):.2f}  MF-DRO better {wins}/{len(d)}")
        print(f"  h83 at seeds 42-46: MF-DRO {dro83:.2f} vs {base} {base83:.2f} "
              f"(margin {dro83-base83:+.2f})")
        if miss: print(f"  INCOMPLETE -- pending {miss}\n"); continue
        # A deficit "replicates" if MF-DRO is still worse on the MAJORITY of
        # seeds AND on the mean. Anything else and the h83 deficit did not hold.
        holds = wins<=2 and d.mean()>0
        verdicts[b]=holds
        print(f"  -> h83 deficit {'REPLICATES' if holds else 'does NOT replicate'}\n")
    if len(verdicts)==2:
        print("REGISTERED BARS")
        print(f"  P1 (Currin: MF-DRO does NOT beat MI-Greedy): "
              f"{'MET' if verdicts['Currin_2D'] else '*** REFUTED -- MF-DRO won Currin ***'}")
        print(f"  P2 (Ackley: no direction predicted) -> "
              f"{'deficit replicates' if verdicts['Ackley_10D'] else 'deficit does NOT replicate'}")
        p3 = not all(verdicts.values())
        print(f"  P3 (at least one deficit does NOT replicate): {'MET' if p3 else 'FAILED'}")
        n_real = 1 + sum(verdicts.values())   # Borehole is established real; Hartmann is not
        print(f"\n  MF-DRO now has {n_real} real deficit(s) of four benchmarks "
              f"(Borehole real, Hartmann not, Currin {'real' if verdicts['Currin_2D'] else 'not'}, "
              f"Ackley {'real' if verdicts['Ackley_10D'] else 'not'}).")
        if n_real==1:
            print("  *** ONE deficit of four -> h83's headline needs restating, and the")
            print("      boundary-optimum explanation becomes the whole story. ***")
