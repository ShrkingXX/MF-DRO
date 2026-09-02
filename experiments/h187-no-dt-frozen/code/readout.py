"""h187 readout, written BEFORE any run finished.

Registered statistic: paired (teacher-only - MF-DRO) frozen rel% points, same seeds.
Threshold = the project's pre-existing harness noise floor, 10.9% worst-case on a
15.82 base = 1.72 rel% points.

  P1 competitive     : |diff| <= 1.72   -> synthesis leg 3 holds on the frozen metric
  P2 teacher BETTER  : diff  <  -1.72   -> the DT is a NET NEGATIVE. Report plainly.
  P3 teacher WORSE   : diff  >  +1.72   -> the DT adds value beyond averaging

lf_fraction is read FIRST, per protocol.
"""
import json, glob, sys, os
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments/h83-main-comparison/code"))
from benchmarks import get_benchmark
from analyse import grid                      # FROZEN metric, imported not re-derived

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
            c.append(float(e["cost_cum"]) - init)
            s.append(float(-b - OPT))
    return 100.0 * grid(np.asarray(c), np.asarray(s), G)[-1] / abs(OPT)


def load(pat):
    return {int(f.split("seed")[1].split(".")[0]): rel(f) for f in sorted(glob.glob(pat))}


if __name__ == "__main__":
    lfv = {}
    for f in sorted(glob.glob(f"{REPO}/experiments/h187-no-dt-frozen/results/"
                              f"Borehole_8D__NODT__seed4[2-6].json")):
        lfv[int(f.split("seed")[1].split(".")[0])] = float(json.load(open(f))["lf_fraction"])
    print("  lf_fraction FIRST, per protocol (MF-DRO control runs at 0.117):")
    print(f"    teacher-only: {[round(v,3) for v in lfv.values()]}")

    T = load(f"{REPO}/experiments/h187-no-dt-frozen/results/ckpt/Borehole_8D__NODT__seed4[2-6].json")
    M = load(f"{REPO}/experiments/h83-main-comparison/results/ckpt/Borehole_8D__MF-DRO__seed4[2-6].json")
    sh = sorted(set(T) & set(M))
    if len(sh) < 5:
        print(f"\n  only {len(sh)}/5 paired seeds available -- not read")
        sys.exit(0)
    d = np.array([T[s] - M[s] for s in sh])
    print(f"\n  frozen rel%   teacher-only {np.mean([T[s] for s in sh]):6.2f}   "
          f"MF-DRO {np.mean([M[s] for s in sh]):6.2f}")
    print(f"  paired (teacher - MF-DRO) per seed: {[round(x,2) for x in d]}")
    print(f"  mean {d.mean():+.2f}  se {d.std(ddof=1)/np.sqrt(len(d)):.2f}  "
          f"teacher better on {int((d<0).sum())}/5")
    v = ("P1 competitive -- synthesis leg 3 holds" if abs(d.mean()) <= 1.72 else
         "P2 teacher BETTER -- the DT is a NET NEGATIVE" if d.mean() < -1.72 else
         "P3 teacher WORSE -- the DT adds value beyond averaging")
    print(f"\n  REGISTERED VERDICT: {v}")
