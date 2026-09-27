"""h210 readout. Frozen metric imported from h83's grid; finals only; endpoint only."""
import json, glob, sys, os
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "experiments/h83-main-comparison/code"))
from benchmarks import get_benchmark
from analyse import grid
G = np.linspace(0, 200, 201); BAND = 1.26
def rel(fn, b):
    OPT = float(get_benchmark(f"{b}_HF")["known_optimal_value"]); q = json.load(open(fn))["queries"]
    ini = [float(e["cost_cum"]) for e in q if e.get("is_init")]
    init = max(ini) if ini else float(q[0]["cost_cum"]) - (2.0 if q[0]["fid"] else 1.0)
    c, s, bb = [], [], -np.inf
    for e in q:
        if e["fid"]: bb = max(bb, float(e["y"]))
        if not e.get("is_init") and float(e["cost_cum"]) > init: c.append(float(e["cost_cum"]) - init); s.append(float(-bb - OPT))
    return 100.0 * grid(np.asarray(c), np.asarray(s), G)[-1] / abs(OPT)
def arm(pat, b): return {int(f.split("seed")[1].split(".")[0]): rel(f, b) for f in sorted(glob.glob(pat))}
def lfm(pat): 
    v=[json.load(open(f)).get("lf_fraction") for f in sorted(glob.glob(pat))]; v=[x for x in v if x is not None]
    return float(np.mean(v)) if v else float('nan')
def line(nm, a, lf=None):
    if not a: return f"    {nm:28s} {'--':>7s}  0/5"
    e = f"   LF {lf:.3f}" if lf==lf and lf is not None else ""
    return f"    {nm:28s} {np.mean(list(a.values())):7.2f}  {len(a)}/5   " + "  ".join(f"{k}:{v:.2f}" for k,v in sorted(a.items())) + e
def paired(x,xn,y,yn):
    sh=sorted(set(x)&set(y))
    if len(sh)<2: return None
    d=[x[s]-y[s] for s in sh]
    print(f"    {xn:11s} - {yn:9s} {np.mean(d):+7.2f}   se {np.std(d,ddof=1)/np.sqrt(len(d)):5.2f}   {xn} better on {sum(1 for v in d if v<0)}/{len(d)}   per-seed {[round(v,2) for v in d]}")
    return float(np.mean(d)), sum(1 for v in d if v<0)
E=f"{REPO}/experiments"; RR=f"{E}/h210-mean-vs-median/results"; H="Hartmann_6D"; B="Borehole_8D"
CT = arm(f"{RR}/{H}__H210CTRLK1__seed4[2-6].json", H)
L1H = arm(f"{RR}/{H}__H210L1NIR__seed4[2-6].json", H); L1B = arm(f"{RR}/{B}__H210L1NIR__seed4[2-6].json", B)
P72 = arm(f"{RR}/{B}__H210MIXRP72__seed4[2-6].json", B); L1M = arm(f"{RR}/{B}__H210L1MIXR__seed4[2-6].json", B)
ROI_H = arm(f"{E}/h84-roi-strategy/results/{H}__ROI-Q10__seed4[2-6].json", H)
NIR_H = arm(f"{E}/h209-mixr-replication/results/{H}__H209NIR__seed4[2-6].json", H)
MIXR_H = arm(f"{E}/h209-mixr-replication/results/{H}__H209MIXR__seed4[2-6].json", H)
NIR_B = arm(f"{E}/h207-random-rollout-dataset/results/{B}__H207NIR-MES-IR__seed4[2-6].json", B)
MIXR_B = arm(f"{E}/h207-random-rollout-dataset/results/{B}__H207MIXR__seed4[2-6].json", B)
print(f"\n  P-CTRL -- HARTMANN: is the K=8 gap config or code drift?  (lean: 5.93)\n")
print(line("CTRL-K1-H (h84 cfg, now)", CT)); print(line("h84 ROI-Q10 K=1 (last full run)", ROI_H)); print(line("NIR-H (K=8)", NIR_H))
if CT and ROI_H:
    d,_=paired(CT,"CTRL-K1-H",'ROI-Q10',ROI_H) if False else paired(CT,"CTRL-K1-H",ROI_H,"ROI-Q10")
    print("    -> " + ("P-CTRL SUPPORTED: reproduces the last full run; the ~4-pt Hartmann gap is the K=8/label CONFIG, and 'K=8 ~ K=1' was Borehole-only"
          if abs(d)<=BAND else "P-CTRL FAILED: current code does NOT reproduce 5.93 at K=1 -> code drift on Hartmann; bisection before any tuning"))
print(f"\n  P-L1-H -- HARTMANN: does the median head cost anything on an interior optimum? (lean: neutral)\n")
print(line("L1-NIR-H", L1H, lfm(f"{RR}/{H}__H210L1NIR__seed4[2-6].json"))); print(line("NIR-H (mse)", NIR_H))
if L1H and NIR_H:
    d,nb=paired(L1H,"L1-NIR-H",NIR_H,"NIR-H")
    print("    -> " + ("P-L1-H SUPPORTED: L1 is neutral where there is no boundary to gain" if abs(d)<=BAND
          else f"P-L1-H FAILED: L1 is {'worse' if d>0 else 'better'} by {abs(d):.2f} on an interior optimum"))
print(f"\n  P-L1-B -- BOREHOLE: does the median head alone recover MIXR's gain? (lean: ~8)\n")
print(line("L1-NIR-B", L1B, lfm(f"{RR}/{B}__H210L1NIR__seed4[2-6].json"))); print(line("NIR-B (mse)", NIR_B)); print(line("MIXR-B (target)", MIXR_B))
if L1B and NIR_B:
    d,nb=paired(L1B,"L1-NIR-B",NIR_B,"NIR-B"); paired(L1B,"L1-NIR-B",MIXR_B,"MIXR-B")
    print("    -> " + ("P-L1-B SUPPORTED" if d<-BAND and nb>=4 else f"P-L1-B FAILED: L1 alone is {'worse' if d>0 else 'better'} by {abs(d):.2f}, {nb}/5"))
print(f"\n  P-P72 / P-L1MIXR -- BOREHOLE (pending)\n"); print(line("MIXR-P72-B", P72)); print(line("L1-MIXR-B", L1M))
if P72 and NIR_B: paired(P72,"MIXR-P72",NIR_B,"NIR-B")
if L1M and NIR_B: paired(L1M,"L1-MIXR",NIR_B,"NIR-B")
print()
