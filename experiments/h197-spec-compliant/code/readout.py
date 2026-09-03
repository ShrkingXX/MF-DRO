"""h197 readout. Quality is compared ONLY by FINAL SIMPLE REGRET (frozen rel% @ cost 200).

Reads results/*.json -- finals only, written by _atomic on completion. Partial
checkpoints are NOT read: a readout that did so once printed a full verdict from
22%-complete runs.

Gate vs h194 CTRL-K1 (no window, 11.59). Threshold = 1.26 (the pre-existing 10.9%
worst-case harness floor on ROI-Q10's 11.59).
  P1 spec arm HELPS : diff < -1.26
  P2 no effect      : |diff| <= 1.26
  P3 spec arm HURTS : diff > +1.26
Also reports h196 (window + real actions, 13.96) to separate the spec's extra
ingredients (L1 loss, real-query info-gain labels) from the window itself.
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
    init = max([float(e["cost_cum"]) for e in q if e.get("is_init")] + [0.0])
    c, s, b = [], [], -np.inf
    for e in q:
        if e["fid"]: b = max(b, float(e["y"]))
        if not e.get("is_init"):
            c.append(float(e["cost_cum"]) - init); s.append(float(-b - OPT))
    return 100.0 * grid(np.asarray(c), np.asarray(s), G)[-1] / abs(OPT)

def arm(pat):
    return {int(f.split("seed")[1].split(".")[0]): rel(f) for f in sorted(glob.glob(pat))}

if __name__ == "__main__":
    E = f"{REPO}/experiments"
    H = arm(f"{E}/h197-spec-compliant/results/Borehole_8D__H197-SPEC__seed4[2-6].json")
    C = arm(f"{E}/h194-expert-plan-window/results/Borehole_8D__CTRL-K1__seed4[2-6].json")
    W = arm(f"{E}/h196-window-real-actions/results/Borehole_8D__H196-REALACT__seed4[2-6].json")
    if len(H) < 5:
        print(f"h197 incomplete ({len(H)}/5 finals) -- no verdict."); sys.exit(0)
    print(f"\n  FINAL SIMPLE REGRET, frozen rel% of |optimum| @ cost 200\n")
    for nm, a in (("h197 SPEC (K=8, L1, real-IG labels)", H),
                  ("h196 WINDOW (K=8, real actions)", W),
                  ("h194 CTRL-K1 (no window)", C)):
        if a: print(f"    {nm:38s} {np.mean(list(a.values())):7.2f}   n={len(a)}")
    print(f"\n    per-seed h197: " + "  ".join(f"{k}:{v:.2f}" for k, v in sorted(H.items())))
    for nm, B, ref in (("CTRL-K1", C, 11.59), ("h196", W, 13.96)):
        sh = sorted(set(H) & set(B))
        if not sh: continue
        d = [H[s] - B[s] for s in sh]
        print(f"\n    h197 - {nm} on {len(sh)} shared seeds:")
        print(f"      paired {np.mean(d):+7.2f}   se {np.std(d, ddof=1)/np.sqrt(len(d)):5.2f}   "
              f"better on {sum(1 for x in d if x < 0)}/{len(d)}")
        print(f"      per-seed: {[round(x,2) for x in d]}")
    sh = sorted(set(H) & set(C)); d = np.mean([H[s] - C[s] for s in sh])
    v = "P1 -- the spec arm HELPS" if d < -1.26 else ("P3 -- the spec arm HURTS" if d > 1.26 else "P2 -- no effect")
    print(f"\n    VERDICT (threshold 1.26): {v}")
