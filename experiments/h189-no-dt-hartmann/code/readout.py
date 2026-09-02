"""h189 readout, written BEFORE all runs finished.

Registered statistic: paired (teacher-only - MF-DRO) frozen rel% on Hartmann.
Threshold = 10.9% worst-case harness noise floor on the Hartmann MF-DRO control's
7.99 base = 0.87 rel% points.

  P1 competitive    : |diff| <= 0.87
  P2 teacher BETTER : diff  <  -0.87  -> h187's net-negative result GENERALISES
  P3 teacher WORSE  : diff  >  +0.87  -> h187 is BOREHOLE-SPECIFIC; h31's direction
                                          is confirmed on the frozen metric

Results live under h187's results dir (see h189/protocol.md -- the worker is reused
unchanged, so RES is hardcoded there). lf_fraction is read FIRST.
"""
import json, glob, sys, os
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments/h83-main-comparison/code"))
from benchmarks import get_benchmark
from analyse import grid

OPT = float(get_benchmark("Hartmann_6D_HF")["known_optimal_value"])
G = np.linspace(0, 200, 201)
R187 = os.path.join(REPO, "experiments/h187-no-dt-frozen/results")


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
    lfv = {int(f.split("seed")[1].split(".")[0]): float(json.load(open(f))["lf_fraction"])
           for f in sorted(glob.glob(f"{R187}/Hartmann_6D__NODT__seed4[2-6].json"))}
    print("  lf_fraction FIRST (Hartmann MF-DRO control runs at 0.800):")
    print(f"    teacher-only: {[round(v,3) for v in lfv.values()]}")
    if lfv:
        print(f"    mean {np.mean(list(lfv.values())):.3f}  -- spread is WIDE; note any seed at 0.0")

    T = load(f"{R187}/ckpt/Hartmann_6D__NODT__seed4[2-6].json")
    M = load(f"{REPO}/experiments/h83-main-comparison/results/ckpt/"
             f"Hartmann_6D__MF-DRO__seed4[2-6].json")
    sh = sorted(set(T) & set(M))
    if len(sh) < 5:
        print(f"\n  only {len(sh)}/5 paired seeds available -- NOT READ")
        sys.exit(0)
    d = np.array([T[s] - M[s] for s in sh])
    print(f"\n  frozen rel%   teacher-only {np.mean([T[s] for s in sh]):6.2f}   "
          f"MF-DRO {np.mean([M[s] for s in sh]):6.2f}")
    print(f"  paired (teacher - MF-DRO) per seed: {[round(x,2) for x in d]}")
    print(f"  mean {d.mean():+.2f}  se {d.std(ddof=1)/np.sqrt(len(d)):.2f}  "
          f"teacher better on {int((d<0).sum())}/5")
    v = ("P1 competitive -- Hartmann undecided" if abs(d.mean()) <= 0.87 else
         "P2 teacher BETTER -- h187's net-negative result GENERALISES" if d.mean() < -0.87 else
         "P3 teacher WORSE -- h187 is BOREHOLE-SPECIFIC")
    print(f"\n  REGISTERED VERDICT: {v}")
    print(f"  (Borehole was -2.85, teacher better 5/5. h31 on Hartmann: MF-DRO ahead 7/10,")
    print(f"   but on final simple regret with unmatched fidelity mixes.)")
