"""h206 readout. Quality compared ONLY by FINAL SIMPLE REGRET (frozen rel% @ cost 200).

Metric IMPORTED from h83's `grid` -- re-deriving it is how a read-point mismatch
gets in. Reads ONLY results/*.json finals, never results/ckpt/.

Arm P is checked FIRST: it is the registered falsifier for h205 itself.
"""
import json, glob, sys, os
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "experiments/h83-main-comparison/code"))
from benchmarks import get_benchmark
from analyse import grid

OPT = float(get_benchmark("Borehole_8D_HF")["known_optimal_value"])
G = np.linspace(0, 200, 201)
BAND = 1.26

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
N = arm(f"{E}/h206-no-position/results/Borehole_8D__H206N-NOPOS__seed4[2-6].json")
P = arm(f"{E}/h206-no-position/results/Borehole_8D__H206P-ARANGE__seed4[2-6].json")
B = arm(f"{E}/h205-absolute-timesteps/results/Borehole_8D__H205B-ABS__seed4[2-6].json")
C = arm(f"{E}/h194-expert-plan-window/results/Borehole_8D__CTRL-K1__seed4[2-6].json")

def line(nm, a):
    if not a: return f"    {nm:26s} {'--':>7s}  0/5  (no finals yet)"
    return (f"    {nm:26s} {np.mean(list(a.values())):7.2f}  {len(a)}/5   "
            + "  ".join(f"{k}:{v:.2f}" for k, v in sorted(a.items())))

print(f"\n  FINAL SIMPLE REGRET -- frozen rel% of |optimum| @ cost 200 "
      f"(Borehole_8D, seeds 42-46)\n")
print(line("N  no position (K=8)", N))
print(line("P  arange (K=8, control)", P))
print(line("h205B absolute (K=8)", B))
print(line("h194 CTRL-K1 (no window)", C))
print(f"    {'h196 arange K8 (OLD code)':26s} {13.96:7.2f}  5/5   (for reference only)")

def paired(x, xn, y, yn):
    sh = sorted(set(x) & set(y))
    if len(sh) < 2: return None
    d = [x[s] - y[s] for s in sh]
    print(f"    {xn} - {yn}   {np.mean(d):+7.2f}   se "
          f"{np.std(d, ddof=1)/np.sqrt(len(d)):5.2f}   {xn} better on "
          f"{sum(1 for v in d if v < 0)}/{len(d)}")
    return float(np.mean(d))

# ---- P FIRST: the registered falsifier for h205 ------------------------------
print(f"\n  FALSIFIER CHECK (registered) -- is arange-K8 on CURRENT code still ~13.96?\n")
dPB = paired(P, "P", B, "B") if P and B else None
if dPB is not None:
    if abs(dPB) <= BAND:
        print(f"\n    *** P is within +/-{BAND} of h205B. h205's arms were NOT separated by")
        print(f"        labelling. h205 RE-OPENS, including the retraction it fired. ***")
    else:
        print(f"\n    P - B = {dPB:+.2f}, outside the +/-{BAND} band: h205's separation holds.")

# ---- the h206 question -------------------------------------------------------
print(f"\n  P1/P2/P3 -- does the positional embedding carry anything? "
      f"(band +/-{BAND} on N - B)\n")
dNB = paired(N, "N", B, "B") if N and B else None
if dNB is not None:
    v = ("P1 SUPPORTED -- position carries nothing; B's gain was SUBTRACTIVE"
         if abs(dNB) <= BAND else
         "P2 SUPPORTED -- absolute episode time is genuinely informative"
         if dNB > BAND else
         "P3 SUPPORTED -- the positional embedding is net HARMFUL at any labelling")
    print(f"\n    N - B = {dNB:+.2f}  ->  {v}")
print()
for x, xn in ((N, "N"), (P, "P")):
    if x and C: paired(x, xn, C, "CTRL-K1")

print(f"\n  LF fraction (fidelity-saturation diagnostic; CTRL-K1 reference 0.261)\n")
for nm, pat in (("N  no position", f"{E}/h206-no-position/results/Borehole_8D__H206N-NOPOS__seed4[2-6].json"),
                ("P  arange     ", f"{E}/h206-no-position/results/Borehole_8D__H206P-ARANGE__seed4[2-6].json")):
    lf = [json.load(open(f)).get("lf_fraction") for f in sorted(glob.glob(pat))]
    lf = [x for x in lf if x is not None]
    if lf: print(f"    {nm} {np.mean(lf):.3f}   " + "  ".join(f"{x:.2f}" for x in lf))
print()
