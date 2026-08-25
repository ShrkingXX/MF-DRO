"""Reference distribution of f over each benchmark's domain, cached.

Exists because normalised L2 distance to x* is NOT a valid proxy for objective
value, which the distance-based stall classifier had assumed. Measured on 6000
uniform samples:

    benchmark      corr(d*,f)   best f at d*<0.3   best f at d*>1.0
    Currin_2D          -0.799            13.7985             5.5057
    Hartmann_6D        -0.417             2.9196             2.8338   <- no gap
    Borehole_8D        -0.587      (0 samples in 6000)        75.3033

On Hartmann the best value reachable within 0.3 of x* is the same as beyond 1.0
away, and in 8D the near-bucket is empty at any feasible sample size. So "far
from x*" cannot mean "searching badly". A query's PERCENTILE in this reference
distribution can.
"""
import os, sys, json
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
CACHE = os.path.join(REPO, "data", "value_reference.npz")
N = 20000

def reference(bench):
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    store = dict(np.load(CACHE)) if os.path.exists(CACHE) else {}
    if bench in store:
        return store[bench]
    import torch
    from benchmarks import get_benchmark
    b = get_benchmark(f"{bench}_HF")
    lo = np.array(b["domain_min"], float); hi = np.array(b["domain_max"], float)
    S = torch.quasirandom.SobolEngine(dimension=lo.size, scramble=True, seed=7)
    P = lo + (hi - lo) * S.draw(N).numpy()
    with torch.no_grad():
        Y = b["make_objective"]()(torch.tensor(P, dtype=torch.float64)).reshape(-1).numpy()
    store[bench] = np.sort(Y)
    np.savez(CACHE, **store)
    return store[bench]

def pct(bench, y):
    """Percentile of value y in the domain-wide reference distribution."""
    r = reference(bench)
    return float(np.searchsorted(r, y) / r.size)

if __name__ == "__main__":
    for b in ("Currin_2D", "Hartmann_6D", "Borehole_8D"):
        r = reference(b)
        print(f"  {b:<13} n={r.size}  min={r[0]:>10.4f}  median={np.median(r):>10.4f}  "
              f"p90={np.percentile(r,90):>10.4f}  max={r[-1]:>10.4f}")
