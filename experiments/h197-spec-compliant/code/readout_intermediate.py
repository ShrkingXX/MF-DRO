"""h197 INTERMEDIATE regret curve. NOT the frozen metric (which is cost 200).

Runs are still in flight, so this reads results/ckpt/ -- the partial checkpoints.
h196's readout once printed a full verdict off 22%-complete runs for exactly this
reason, so nothing here is a verdict: the comparison is capped at the LOWEST cost
every arm has actually reached, and the frozen number is deliberately not quoted.

Uses h83's own `grid`, imported. Re-deriving the metric by hand gives 91.59 where
the truth is 15.82.
"""
import json, glob, sys, os
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "experiments/h83-main-comparison/code"))
from benchmarks import get_benchmark
from analyse import grid

OPT = float(get_benchmark("Borehole_8D_HF")["known_optimal_value"])

def curve(fn, G):
    q = json.load(open(fn))["queries"]
    # In-flight checkpoints for h197 carry NO is_init entries (n_init=0), while a
    # COMPLETED arm's checkpoint has 30 of them totalling cost 40. Taking the
    # is_init max blindly therefore left h197 on an unshifted cost axis and CTRL
    # shifted by -40 -- a 40-unit offset between the two curves, which made h197
    # look far worse at every point. Fall back to deriving the offset from the
    # first recorded query's own fidelity cost, which reproduces 40.0 exactly.
    _ini = [float(e["cost_cum"]) for e in q if e.get("is_init")]
    if _ini:
        init = max(_ini)
    else:
        _f = q[0]
        init = float(_f["cost_cum"]) - (2.0 if _f["fid"] else 1.0)
    c, s, b = [], [], -np.inf
    for e in q:
        if e["fid"]:
            b = max(b, float(e["y"]))
        if not e.get("is_init"):
            c.append(float(e["cost_cum"]) - init); s.append(float(-b - OPT))
    return 100.0 * grid(np.asarray(c), np.asarray(s), G) / abs(OPT), max(c)

def arm(pat, G):
    out, reach = {}, []
    for f in sorted(glob.glob(pat)):
        cv, mx = curve(f, G); out[int(f.split("seed")[1].split(".")[0])] = cv; reach.append(mx)
    return out, (min(reach) if reach else 0.0)

if __name__ == "__main__":
    E = f"{REPO}/experiments"
    pats = {
        "h197 SPEC (K=8, L1, real-IG labels)": f"{E}/h197-spec-compliant/results/ckpt/Borehole_8D__H197-SPEC__seed4[2-6].json",
        "h196 WINDOW (K=8, real actions)":     f"{E}/h196-window-real-actions/results/ckpt/Borehole_8D__H196-REALACT__seed4[2-6].json",
        "h194 CTRL-K1 (no window)":            f"{E}/h194-expert-plan-window/results/ckpt/Borehole_8D__CTRL-K1__seed4[2-6].json",
    }
    G = np.linspace(0, 200, 201)
    arms, reach = {}, {}
    for k, p in pats.items():
        a, r = arm(p, G)
        if a: arms[k], reach[k] = a, r
    cap = min(reach.values())
    print(f"\nlowest cost reached by ANY seed of ANY arm: {cap:.0f}  "
          f"-> comparison capped there (frozen metric is cost 200, NOT reported)\n")
    pts = [c for c in (25, 50, 75, 100, 125, 150) if c <= cap]
    hdr = "  " + "arm".ljust(38) + "".join(f"{f'c={c}':>9}" for c in pts) + "   seeds"
    print(hdr); print("  " + "-"*(len(hdr)-2))
    for k, a in arms.items():
        m = [np.mean([a[s][c] for s in a]) for c in pts]
        print("  " + k.ljust(38) + "".join(f"{v:9.2f}" for v in m) + f"   {len(a)}")
    if "h197 SPEC (K=8, L1, real-IG labels)" in arms and "h194 CTRL-K1 (no window)" in arms:
        A, B = arms["h197 SPEC (K=8, L1, real-IG labels)"], arms["h194 CTRL-K1 (no window)"]
        sh = sorted(set(A) & set(B))
        print(f"\n  paired h197 - CTRL-K1 on {len(sh)} shared seeds {sh}:")
        for c in pts:
            d = [A[s][c] - B[s][c] for s in sh]
            print(f"    c={c:<4} mean {np.mean(d):+7.2f}  se {np.std(d,ddof=1)/np.sqrt(len(d)):5.2f}  "
                  f"better on {sum(1 for x in d if x<0)}/{len(d)}")
