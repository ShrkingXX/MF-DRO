"""h184 readout, written BEFORE the CTRL arm landed.

Registered statistic: the NUMBER OF SEEDS (0-5) where LF-forced HEAD is worse than
LF-forced CTRL on the frozen metric.
  P1 Hartmann-like, account SUPPORTED : >= 4
  P2 unchanged,     account REFUSED   : 2 or 3      (Borehole is at 2 unforced)
  P3 inverted                         : <= 1

SC (read FIRST): realised lf_fraction must be ~0.75 on BOTH arms, else the forcing
was asymmetric and the gap is confounded.
"""
import json, glob, sys, os
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments/h83-main-comparison/code"))
from benchmarks import get_benchmark
from analyse import grid                       # FROZEN metric, imported not re-derived

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


def load(tag):
    out = {}
    for f in sorted(glob.glob(f"{REPO}/experiments/h184-lf-forcing/results/ckpt/"
                              f"Borehole_8D__{tag}__seed4[2-6].json")):
        out[int(f.split("seed")[1].split(".")[0])] = rel(f)
    return out


def lf(tag):
    v = {}
    for f in sorted(glob.glob(f"{REPO}/experiments/h184-lf-forcing/results/"
                              f"Borehole_8D__{tag}__seed4[2-6].json")):
        v[int(f.split("seed")[1].split(".")[0])] = float(json.load(open(f))["lf_fraction"])
    return v


if __name__ == "__main__":
    lc, lh = lf("LFF-CTRL"), lf("LFF-HEAD")
    print("  SC FIRST -- realised lf_fraction (unforced Borehole is 0.117):")
    print(f"    LFF-CTRL {[round(v,3) for v in lc.values()]}")
    print(f"    LFF-HEAD {[round(v,3) for v in lh.values()]}")
    if lc and lh:
        ok = abs(np.mean(list(lc.values())) - np.mean(list(lh.values()))) < 0.05
        print(f"    SC symmetric forcing: {'PASS' if ok else 'FAIL -- gap is confounded'}")
    C, H = load("LFF-CTRL"), load("LFF-HEAD")
    sh = sorted(set(C) & set(H))
    if len(sh) < 5:
        print(f"\n  only {len(sh)}/5 paired seeds available -- not read")
        sys.exit(0)
    d = np.array([H[s] - C[s] for s in sh])
    print(f"\n  frozen rel%  CTRL {np.mean([C[s] for s in sh]):6.2f}   "
          f"HEAD {np.mean([H[s] for s in sh]):6.2f}")
    print(f"  paired HEAD-CTRL per seed: {[round(x,2) for x in d]}")
    print(f"  mean {d.mean():+.2f}   seeds where HEAD is worse: {(d>0).sum()}/5")
    n = int((d > 0).sum())
    verdict = ("P1 Hartmann-like -- lf_fraction account SUPPORTED" if n >= 4 else
               "P3 inverted" if n <= 1 else
               "P2 unchanged -- lf_fraction account REFUSED")
    print(f"\n  REGISTERED VERDICT: {verdict}")
    print(f"  (unforced Borehole was 2/5 worse, paired +1.15; Hartmann is 5/5, paired +17.18)")
