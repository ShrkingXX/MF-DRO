"""h199 readout. Quality is compared ONLY by FINAL SIMPLE REGRET (frozen rel% @ cost 200).

CEILING/DIAGNOSTIC, NOT A METHOD: x* and true f are not available at run time.

Gate vs h194 CTRL-K1 (11.59). Threshold = 1.26.
  P1 oracle-lookahead HELPS (has headroom) : diff < -1.26
  P2 no effect                             : |diff| <= 1.26
  P3 oracle-lookahead HURTS                : diff > +1.26
"""
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
    H = arm(f"{E}/h199-oracle-lookahead-ceiling/results/Borehole_8D__H199-ORACLE-LOOK__seed4[2-6].json")
    C = arm(f"{E}/h194-expert-plan-window/results/Borehole_8D__CTRL-K1__seed4[2-6].json")
    if len(H) < 5:
        print(f"h199 incomplete ({len(H)}/5 finals) -- no verdict."); sys.exit(0)
    print(f"\n  FINAL SIMPLE REGRET, frozen rel% of |optimum| @ cost 200\n")
    print(f"    h199 ORACLE-lookahead (ceiling)   {np.mean(list(H.values())):7.2f}   n={len(H)}")
    print(f"    h194 CTRL-K1 (no window, MES)     {np.mean(list(C.values())):7.2f}   n={len(C)}")
    print(f"\n    per-seed h199: " + "  ".join(f"{k}:{v:.2f}" for k, v in sorted(H.items())))
    sh = sorted(set(H) & set(C)); d = [H[s] - C[s] for s in sh]
    print(f"\n    h199 - CTRL-K1 on {len(sh)} shared seeds:")
    print(f"      paired {np.mean(d):+7.2f}   se {np.std(d, ddof=1)/np.sqrt(len(d)):5.2f}   "
          f"better on {sum(1 for x in d if x < 0)}/{len(d)}")
    print(f"      per-seed: {[round(x,2) for x in d]}")
    dm = np.mean(d)
    v = "P1 -- oracle-lookahead HELPS (real headroom exists)" if dm < -1.26 else \
        ("P3 -- oracle-lookahead HURTS" if dm > 1.26 else "P2 -- no effect")
    print(f"\n    VERDICT (threshold 1.26): {v}")
