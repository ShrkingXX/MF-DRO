"""h194 Stage 1a readout, written BEFORE any run finished.

Gate: WINDOW(K=8) - CTRL-K1, frozen rel% points. Threshold = the pre-existing 10.9%
worst-case harness floor on ROI-Q10's 11.59 = 1.26.
  P1 window HELPS : diff < -1.26
  P2 no effect    : |diff| <= 1.26
  P3 window HURTS : diff > +1.26   <- what the mechanism predicts

Also reports CTRL-K1 vs h84's ROI-Q10 (11.59) as a free drift check across 17 commits.
"""
import json, glob, sys, os
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments/h83-main-comparison/code"))
from benchmarks import get_benchmark
from analyse import grid

OPT = float(get_benchmark("Borehole_8D_HF")["known_optimal_value"])
G = np.linspace(0, 200, 201)


def rel(fn):
    q = json.load(open(fn))["queries"]
    init = max([float(e["cost_cum"]) for e in q if e.get("is_init")] + [0.0])
    c, s, b = [], [], -np.inf
    for e in q:
        if e["fid"]:
            b = max(b, float(e["y"]))
        if not e.get("is_init"):
            c.append(float(e["cost_cum"]) - init); s.append(float(-b - OPT))
    return 100.0 * grid(np.asarray(c), np.asarray(s), G)[-1] / abs(OPT)


def arm(pat):
    return {int(f.split("seed")[1].split(".")[0]): rel(f) for f in sorted(glob.glob(pat))}


if __name__ == "__main__":
    R = f"{REPO}/experiments/h194-expert-plan-window/results/ckpt"
    W = arm(f"{R}/Borehole_8D__WINDOW-K8__seed4[2-6].json")
    C = arm(f"{R}/Borehole_8D__CTRL-K1__seed4[2-6].json")
    H = arm(f"{REPO}/experiments/h84-roi-strategy/results/ckpt/Borehole_8D__ROI-Q10__seed4[2-6].json")
    sh = sorted(set(W) & set(C))
    if len(sh) < 5:
        print(f"  only {len(sh)}/5 paired seeds -- NOT READ"); sys.exit(0)
    if C and H:
        shc = sorted(set(C) & set(H))
        dc = np.array([C[s] - H[s] for s in shc])
        print(f"  DRIFT CHECK -- fresh CTRL-K1 vs h84's ROI-Q10 (11.59), 17 commits apart:")
        print(f"    fresh {np.mean([C[s] for s in shc]):6.2f}   h84 {np.mean([H[s] for s in shc]):6.2f}"
              f"   paired {dc.mean():+.2f}")
        print(f"    {'baseline STABLE, h84-era ROI controls remain quotable' if abs(dc.mean()) <= 1.26 else 'BASELINE MOVED -- h84-era ROI numbers are NOT quotable'}")
    d = np.array([W[s] - C[s] for s in sh])
    print(f"\n  frozen rel%   WINDOW(K=8) {np.mean([W[s] for s in sh]):6.2f}"
          f"   CTRL-K1 {np.mean([C[s] for s in sh]):6.2f}")
    print(f"  paired (WINDOW - CTRL) per seed: {[round(x,2) for x in d]}")
    print(f"  mean {d.mean():+.2f}  se {d.std(ddof=1)/np.sqrt(len(d)):.2f}  window worse on {int((d>0).sum())}/5")
    v = ("P1 window HELPS" if d.mean() < -1.26 else
         "P3 window HURTS -- as the mechanism predicted" if d.mean() > 1.26 else
         "P2 no effect")
    print(f"\n  REGISTERED VERDICT: {v}")
