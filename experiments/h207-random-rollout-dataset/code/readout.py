"""h207 readout. Quality compared ONLY by FINAL SIMPLE REGRET (frozen rel% @ cost 200).
Metric IMPORTED from h83's `grid`. Finals only, never results/ckpt/.

Order: P-NIR first (is the label alone an intervention?), then P-R / P-W / P-MIX
against N-IR, then the H168 RTG-sensitivity that decides MIX's mechanism.
"""
import json, glob, sys, os
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "experiments/h83-main-comparison/code"))
from benchmarks import get_benchmark
from analyse import grid
OPT = float(get_benchmark("Borehole_8D_HF")["known_optimal_value"]); G = np.linspace(0, 200, 201); BAND = 1.26

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
def arm(pat): return {int(f.split("seed")[1].split(".")[0]): rel(f) for f in sorted(glob.glob(pat))}
def sens(fn, last=30):
    p = json.load(open(fn)).get("h168_probe", [])
    out = []
    for rec in p[-last:]:
        xs = {pr["rtg"]: np.array(pr["x"]) for pr in rec["probes"] if pr["state"] == "real"}
        if len(xs) >= 2:
            ref = xs[min(xs)]; out.append(max(np.abs(x - ref).max() for x in xs.values()))
    return float(np.mean(out)) if out else float("nan")
def sens_arm(pat): return {int(f.split("seed")[1].split(".")[0]): sens(f) for f in sorted(glob.glob(pat))}
def lf_arm(pat): return [json.load(open(f)).get("lf_fraction") for f in sorted(glob.glob(pat))]

E = f"{REPO}/experiments"; R = f"{E}/h207-random-rollout-dataset/results"
ARMS = {"NIR": "H207NIR-MES-IR", "MIXR": "H207MIXR", "MIXO": "H207MIXO"}
P = {k: f"{R}/Borehole_8D__{v}__seed4[2-6].json" for k, v in ARMS.items()}
A = {k: arm(p) for k, p in P.items()}
N = arm(f"{E}/h206-no-position/results/Borehole_8D__H206N-NOPOS__seed4[2-6].json")
C = arm(f"{E}/h194-expert-plan-window/results/Borehole_8D__CTRL-K1__seed4[2-6].json")

def line(nm, a):
    if not a: return f"    {nm:34s} {'--':>7s}  0/5"
    return (f"    {nm:34s} {np.mean(list(a.values())):7.2f}  {len(a)}/5   "
            + "  ".join(f"{k}:{v:.2f}" for k, v in sorted(a.items())))
def paired(x, xn, y, yn):
    sh = sorted(set(x) & set(y))
    if len(sh) < 2: print(f"    {xn} - {yn}: <2 shared seeds"); return None
    d = [x[s] - y[s] for s in sh]
    print(f"    {xn:4s} - {yn:5s} {np.mean(d):+7.2f}   se {np.std(d, ddof=1)/np.sqrt(len(d)):5.2f}   "
          f"{xn} better on {sum(1 for v in d if v < 0)}/{len(d)}   per-seed {[round(v,1) for v in d]}")
    return float(np.mean(d))

print(f"\n  FINAL SIMPLE REGRET -- frozen rel% of |optimum| @ cost 200 (Borehole_8D, seeds 42-46)\n")
print(line("NIR   20 MES/member, terminal_improvement", A["NIR"]))
print(line("MIXR  20 MES + 20 random", A["MIXR"]))
print(line("MIXO  20 MES + 20 ORACLE (ceiling)", A["MIXO"]))
print(line("h206N 20 MES, mes_entropy (ctrl)", N))
print(line("CTRL-K1 (no window)", C))
print(f"    {'h201A oracle-only K=8 (arange, ref)':34s} {0.00:7.2f}  suspected positional shortcut")
print(f"    {'h149 random teacher, K=1 (ref)':34s} {43.94:7.2f}  = saturation floor")

print(f"\n  P-NIR FIRST -- is the label alone an intervention?  (band +/-{BAND})\n")
d = paired(A["NIR"], "NIR", N, "N") if A["NIR"] and N else None
if d is not None:
    print(f"    -> {'P-NIR SUPPORTED: label alone within band' if abs(d) <= BAND else 'P-NIR REFUTED: the IR label is itself an intervention -- reframe the arms below'}")

print(f"\n  P-MIXR / P-MIXO against NIR  (registered lean: MIXO <= 3.0 with >= 2x sensitivity; MIXR not selective)\n")
for k in ("MIXR", "MIXO"):
    if A[k] and A["NIR"]: paired(A[k], k, A["NIR"], "NIR")
if A["MIXO"]:
    mo = float(np.mean(list(A["MIXO"].values())))
    print(f"\n    MIXO mean {mo:.2f}: " + ("P-MIXO-SELECT (<= 3.0): the DT reads RTG and emits the oracle half"
          if mo <= 3.0 else "P-MIXO-BLUR or PARTIAL -- see sensitivity below"))

print(f"\n  RTG SENSITIVITY (H168 probe, last 30 iters, max|dx| over sweep vs rtg=0)\n")
S = {k: sens_arm(p) for k, p in P.items()}
for k in ("NIR", "MIXR", "MIXO"):
    if S[k]:
        print(f"    {k:4s} mean {np.mean(list(S[k].values())):.4f}   " +
              "  ".join(f"{s}:{v:.4f}" for s, v in sorted(S[k].items())))
for k in ("MIXR", "MIXO"):
    if S[k] and S["NIR"]:
        ratio = np.mean(list(S[k].values())) / max(np.mean(list(S["NIR"].values())), 1e-12)
        print(f"    {k} / NIR sensitivity ratio = {ratio:.2f}x  (registered: >= 2x if RTG became selective)")

print(f"\n  LF fraction (CTRL-K1 0.261, h206N 0.462)\n")
for k in ("NIR", "MIXR", "MIXO"):
    lf = [x for x in lf_arm(P[k]) if x is not None]
    if lf: print(f"    {k:4s} {np.mean(lf):.3f}   " + "  ".join(f"{x:.2f}" for x in lf))
print()
