"""h196 readout, written BEFORE any run finished.

Compares the CORRECTED window (real past actions fed) against h194's CTRL-K1 (11.59)
and h194's defective WINDOW-K8 (16.58). P1: |h196 - 16.58| > 1.26 -> the fix matters.

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
    R  = f"{REPO}/experiments/h196-window-real-actions/results/ckpt"
    R94 = f"{REPO}/experiments/h194-expert-plan-window/results/ckpt"
    W = arm(f"{R}/Borehole_8D__H196-REALACT__seed4[2-6].json")
    C = arm(f"{R94}/Borehole_8D__CTRL-K1__seed4[2-6].json")
    W94 = arm(f"{R94}/Borehole_8D__WINDOW-K8__seed4[2-6].json")
    H = arm(f"{REPO}/experiments/h84-roi-strategy/results/ckpt/Borehole_8D__ROI-Q10__seed4[2-6].json")
    # COMPLETENESS GUARD. arm() reads results/ckpt/, which is written from the FIRST
    # iteration onward -- so a seed-count check passes immediately and the readout will
    # happily report authoritative-looking numbers from 22%-complete runs. (Observed:
    # 31.29 "worse on 5/5" from runs at cost 52/240.) The final results/*.json is written
    # by _atomic ONLY when a run finishes, so require those instead.
    fin = sorted(glob.glob(f"{REPO}/experiments/h196-window-real-actions/results/"
                           f"Borehole_8D__H196-REALACT__seed4[2-6].json"))
    if len(fin) < 5:
        print(f"  h196 has {len(fin)}/5 FINISHED runs -- NOT READ "
              f"(ckpt files exist mid-run and would give a false reading)")
        sys.exit(0)
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
    # ADDED before results (does NOT change the gate, which is WINDOW - CTRL-K1):
    # the human asked whether this beats "our default MF-DRO", which is use_roi=False
    # = 15.82, not the ROI-Q10 the gate is registered against. Report both.
    D = arm(f"{REPO}/experiments/h83-main-comparison/results/ckpt/Borehole_8D__MF-DRO__seed4[2-6].json")
    if D:
        shd = sorted(set(W) & set(D))
        dd = np.array([W[s] - D[s] for s in shd])
        print(f"\n  ANSWERING THE DIRECT QUESTION -- vs DEFAULT MF-DRO (use_roi=False, 15.82):")
        print(f"    WINDOW {np.mean([W[s] for s in shd]):6.2f}   default {np.mean([D[s] for s in shd]):6.2f}"
              f"   paired {dd.mean():+.2f}   better on {int((dd<0).sum())}/{len(shd)}")
        print(f"    (lower is better; ROI alone already gets 11.59, so beating 15.82 is"
              f" not evidence the WINDOW helped)")

    d = np.array([W[s] - C[s] for s in sh])
    print(f"\n  frozen rel%   WINDOW(K=8) {np.mean([W[s] for s in sh]):6.2f}"
          f"   CTRL-K1 {np.mean([C[s] for s in sh]):6.2f}")
    print(f"  paired (WINDOW - CTRL) per seed: {[round(x,2) for x in d]}")
    print(f"  mean {d.mean():+.2f}  se {d.std(ddof=1)/np.sqrt(len(d)):.2f}  window worse on {int((d>0).sum())}/5")
    v = ("P1 window HELPS" if d.mean() < -1.26 else
         "P3 window HURTS" if d.mean() > 1.26 else
         "P2 no effect")
    print(f"\n  REGISTERED VERDICT: {v}")
