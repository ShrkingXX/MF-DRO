"""h211 readout. Frozen metric imported from h83's grid; finals only; endpoint only."""
import json, glob, sys, os
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "experiments/h83-main-comparison/code"))
from benchmarks import get_benchmark
from analyse import grid
G = np.linspace(0, 200, 201); BAND = 1.26
def rel(fn, b):
    OPT=float(get_benchmark(f"{b}_HF")["known_optimal_value"]); q=json.load(open(fn))["queries"]
    ini=[float(e["cost_cum"]) for e in q if e.get("is_init")]
    init=max(ini) if ini else float(q[0]["cost_cum"])-(2.0 if q[0]["fid"] else 1.0)
    c,s,bb=[],[],-np.inf
    for e in q:
        if e["fid"]: bb=max(bb,float(e["y"]))
        if not e.get("is_init") and float(e["cost_cum"])>init: c.append(float(e["cost_cum"])-init); s.append(float(-bb-OPT))
    return 100.0*grid(np.asarray(c),np.asarray(s),G)[-1]/abs(OPT)
def arm(pat,b): return {int(f.split("seed")[1].split(".")[0]): rel(f,b) for f in sorted(glob.glob(pat))}
def lfm(pat):
    v=[json.load(open(f)).get("lf_fraction") for f in sorted(glob.glob(pat))]; v=[x for x in v if x is not None]
    return float(np.mean(v)) if v else float('nan')
def line(nm,a,lf=None):
    if not a: return f"    {nm:32s} {'--':>7s}  0/5"
    e=f"   LF {lf:.3f}" if lf is not None and lf==lf else ""
    return f"    {nm:32s} {np.mean(list(a.values())):7.2f}  {len(a)}/5   "+"  ".join(f"{k}:{v:.2f}" for k,v in sorted(a.items()))+e
def paired(x,xn,y,yn):
    sh=sorted(set(x)&set(y))
    if len(sh)<2: return None
    d=[x[s]-y[s] for s in sh]
    print(f"    {xn:12s} - {yn:11s} {np.mean(d):+7.2f}   se {np.std(d,ddof=1)/np.sqrt(len(d)):5.2f}   {xn} better on {sum(1 for v in d if v<0)}/{len(d)}   per-seed {[round(v,2) for v in d]}")
    return float(np.mean(d)), sum(1 for v in d if v<0)
E=f"{REPO}/experiments"; RR=f"{E}/h211-mixr-on-k1/results"; H="Hartmann_6D"; B="Borehole_8D"
MB=arm(f"{RR}/{B}__H211MIXRK1__seed4[2-6].json",B); MH=arm(f"{RR}/{H}__H211MIXRK1__seed4[2-6].json",H)
TI=arm(f"{RR}/{H}__H211K1TI__seed4[2-6].json",H)
CB=arm(f"{E}/h84-roi-strategy/results/{B}__ROI-Q10__seed4[2-6].json",B)
CH=arm(f"{E}/h210-mean-vs-median/results/{H}__H210CTRLK1__seed4[2-6].json",H)
MIXR_B=arm(f"{E}/h207-random-rollout-dataset/results/{B}__H207MIXR__seed4[2-6].json",B)
MIXR_H=arm(f"{E}/h209-mixr-replication/results/{H}__H209MIXR__seed4[2-6].json",H)
NIR_H=arm(f"{E}/h209-mixr-replication/results/{H}__H209NIR__seed4[2-6].json",H)
MES_B=arm(f"{E}/h83-main-comparison/results/{B}__MF-MES__seed4[2-6].json",B)
MES_H=arm(f"{E}/h83-main-comparison/results/{H}__MF-MES__seed4[2-6].json",H)
print(f"\n  P-B -- BOREHOLE: does the random half's gain survive off the K=8 base?  (lean ~7)\n")
print(line("MIXR-K1-B", MB, lfm(f"{RR}/{B}__H211MIXRK1__seed4[2-6].json")))
print(line("CTRL-K1-B (the real reference)", CB)); print(line("MIXR-B on K=8 (h207)", MIXR_B)); print(line("MF-MES", MES_B))
if MB and CB:
    d,nb=paired(MB,"MIXR-K1-B",CB,"CTRL-K1-B")
    print("    -> "+("P-B SUPPORTED: the effect transfers off the window" if d<-BAND and nb>=4
          else f"P-B FAILED: {'no transfer (inside band)' if abs(d)<=BAND else f'WORSE by {d:+.2f}'} -> the effect is a K=8 INTERACTION; this protocol's premise retracted"))
print(f"\n  P-H -- HARTMANN: does the random half's harm survive off the window?  (lean ~8)\n")
print(line("MIXR-K1-H", MH, lfm(f"{RR}/{H}__H211MIXRK1__seed4[2-6].json")))
print(line("CTRL-K1-H", CH)); print(line("MIXR-H on K=8 (h209)", MIXR_H)); print(line("MF-MES", MES_H))
if MH and CH:
    d,nb=paired(MH,"MIXR-K1-H",CH,"CTRL-K1-H")
    print("    -> "+("P-H SUPPORTED: the harm is the random half, persists on the better base" if d>BAND
          else "P-H FAILED: harm vanishes on K=1 -> it was the WINDOW; a single config good on both is reachable" if abs(d)<=BAND
          else f"MIXR-K1-H is BETTER by {abs(d):.2f}"))
print(f"\n  P-LABEL -- HARTMANN: is the terminal_improvement label neutral at K=1?\n")
print(line("K1-TI-H", TI, lfm(f"{RR}/{H}__H211K1TI__seed4[2-6].json"))); print(line("CTRL-K1-H (mes_entropy)", CH)); print(line("NIR-H (K=8 + label)", NIR_H))
if TI and CH:
    d,nb=paired(TI,"K1-TI-H",CH,"CTRL-K1-H")
    print("    -> "+("P-LABEL SUPPORTED: label neutral; h210's 3.82-pt gap is the WINDOW alone" if abs(d)<=BAND
          else f"P-LABEL FAILED: the label alone costs {d:+.2f} -> h210's gap SPLITS into label + window"))
print()
