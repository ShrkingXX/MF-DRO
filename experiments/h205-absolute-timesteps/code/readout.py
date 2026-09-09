"""h205 readout. Quality compared ONLY by FINAL SIMPLE REGRET (frozen rel% @ cost 200).

The metric is IMPORTED from h83's `grid` -- re-deriving it is how a read-point
mismatch gets in. Reads ONLY results/*.json finals, never results/ckpt/.
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

E = f"{REPO}/experiments"
R = f"{E}/h205-absolute-timesteps/results"
ARMS = [("A  prefix only        ", f"{R}/Borehole_8D__H205A-PREFIX__seed4[2-6].json"),
        ("B  absolute only      ", f"{R}/Borehole_8D__H205B-ABS__seed4[2-6].json"),
        ("C  both               ", f"{R}/Borehole_8D__H205C-BOTH__seed4[2-6].json")]
CTRL = arm(f"{E}/h194-expert-plan-window/results/Borehole_8D__CTRL-K1__seed4[2-6].json")

print(f"\n  FINAL SIMPLE REGRET -- frozen rel% of |optimum| @ cost 200  (Borehole_8D, seeds 42-46)\n")
print(f"    {'arm':24s} {'mean':>7s}  n   per-seed")
got = {}
for name, pat in ARMS:
    a = arm(pat); got[name.strip()[0]] = a
    ps = "  ".join(f"{k}:{v:.2f}" for k, v in sorted(a.items()))
    print(f"    {name} {np.mean(list(a.values())):7.2f}  {len(a)}   {ps}")
print(f"    {'CTRL-K1 (h194, ref)':24s} {np.mean(list(CTRL.values())):7.2f}  {len(CTRL)}   "
      + "  ".join(f"{k}:{v:.2f}" for k, v in sorted(CTRL.items())))
print(f"\n    reference points: saturation floor 43.94 (initial design, 0/5 improvement)")
print(f"                      h201B oracle-teacher K=1 = 43.94 (exactly the floor)")

print(f"\n  PAIRED vs CTRL-K1  (P1 registered: A beats CTRL, i.e. < -1.26)\n")
for k in "ABC":
    a = got[k]; sh = sorted(set(a) & set(CTRL)); d = [a[s] - CTRL[s] for s in sh]
    print(f"    {k} - CTRL   {np.mean(d):+7.2f}   se {np.std(d, ddof=1)/np.sqrt(len(d)):5.2f}   "
          f"better on {sum(1 for x in d if x < 0)}/{len(d)}   per-seed {[round(x,1) for x in d]}")

print(f"\n  PAIRED between arms  (ordering prediction registered: A >= C > B)\n")
for x, y in (("A","C"), ("A","B"), ("C","B")):
    sh = sorted(set(got[x]) & set(got[y])); d = [got[x][s] - got[y][s] for s in sh]
    print(f"    {x} - {y}      {np.mean(d):+7.2f}   se {np.std(d, ddof=1)/np.sqrt(len(d)):5.2f}   "
          f"{x} better on {sum(1 for v in d if v < 0)}/{len(d)}")

print(f"\n  LF fraction (fidelity-saturation diagnostic; CTRL-K1 reference 0.261)\n")
for name, pat in ARMS:
    lf = [json.load(open(f)).get("lf_fraction") for f in sorted(glob.glob(pat))]
    lf = [x for x in lf if x is not None]
    print(f"    {name} {np.mean(lf):.3f}   " + "  ".join(f"{x:.2f}" for x in lf))
print()
