"""h201 full readout: arm A (K=8) vs arm B (K=1) vs CTRL-K1. Quality = final simple regret only."""
import json, glob, sys, os
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "experiments/h83-main-comparison/code"))
from benchmarks import get_benchmark
from analyse import grid

OPT = float(get_benchmark("Borehole_8D_HF")["known_optimal_value"])
G = np.linspace(0, 200, 201)

def rel(fn):
    q = json.load(open(fn))["queries"]
    ini = [float(e["cost_cum"]) for e in q if e.get("is_init")]
    init = max(ini) if ini else float(q[0]["cost_cum"]) - (2.0 if q[0]["fid"] else 1.0)
    c, s, b = [], [], -np.inf
    for e in q:
        if e["fid"]: b = max(b, float(e["y"]))
        if not e.get("is_init") and float(e["cost_cum"]) > init:
            c.append(float(e["cost_cum"]) - init); s.append(float(-b - OPT))
    return 100.0 * grid(np.asarray(c), np.asarray(s), G)[-1] / abs(OPT)

def arm(pat):
    return {int(f.split("seed")[1].split(".")[0]): rel(f) for f in sorted(glob.glob(pat))}

if __name__ == "__main__":
    E = f"{REPO}/experiments"
    A = arm(f"{E}/h201-oracle-teacher-window/results/Borehole_8D__H201A-ORACLE-K8__seed4[2-6].json")
    B = arm(f"{E}/h201-oracle-teacher-window/results/Borehole_8D__H201B-ORACLE-K1__seed4[2-6].json")
    C = arm(f"{E}/h194-expert-plan-window/results/Borehole_8D__CTRL-K1__seed4[2-6].json")
    print(f"\n  FINAL SIMPLE REGRET, frozen rel% @ cost 200\n")
    for nm, a in (("h201A oracle + K=8 window", A), ("h201B oracle + K=1 (matched ctrl)", B),
                  ("h194 CTRL-K1 (MES, no window)", C)):
        if a: print(f"    {nm:35s} {np.mean(list(a.values())):8.2f}   n={len(a)}")
    print(f"\n    per-seed h201B: " + "  ".join(f"{k}:{v:.2f}" for k, v in sorted(B.items())))
    sh = sorted(set(A) & set(B)); d = [A[s] - B[s] for s in sh]
    print(f"\n    h201A - h201B (isolates the WINDOW, same teacher) on {len(sh)} seeds:")
    print(f"      paired {np.mean(d):+8.2f}   se {np.std(d, ddof=1)/np.sqrt(len(d)):6.2f}   better on {sum(1 for x in d if x<0)}/{len(d)}")
    sh2 = sorted(set(B) & set(C)); d2 = [B[s] - C[s] for s in sh2]
    print(f"\n    h201B - CTRL-K1 (isolates the TEACHER at K=1) on {len(sh2)} seeds:")
    print(f"      paired {np.mean(d2):+8.2f}   se {np.std(d2, ddof=1)/np.sqrt(len(d2)):6.2f}   better on {sum(1 for x in d2 if x<0)}/{len(d2)}")
