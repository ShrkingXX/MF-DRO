#!/usr/bin/env python
"""Per-dimension diagnostics that CANNOT be computed unweighted by accident.

WHY THIS EXISTS
  Five per-dimension averages were examined in one session and all five were
  misleading: a metric choice (h96), a containment diagnostic (h100), a
  boundary-fraction (peer's), Borehole dispersion (h95), Hartmann dispersion.
  Five for five.

  The cause is structural, not carelessness. On these benchmarks some dimensions
  carry almost none of the output variance -- four of Borehole's eight carry 0.4%
  between them -- and those are precisely the dimensions the optimiser wanders
  in, because nothing penalises it for doing so. **The noise is produced by the
  thing being measured.** So an unweighted average of a per-dimension statistic
  is dominated by dimensions that cannot affect the objective, every time.

  "Remember to weight" was not sufficient: it was written down mid-session and
  then violated twice, once by each session. So the weights live in the
  instrument.

API
  w = shares(bench)                  measured first-order shares, cached
  agg(per_dim_values, bench)         sensitivity-weighted aggregate
  agg(v, bench, unweighted=True)     allowed ONLY with the explicit flag, and
                                     prints why it is probably wrong

  CLI:  tools/perdim.py Borehole_8D     -> prints the shares and the dominance
                                          diagnosis for that benchmark
"""
import os, sys, json
import numpy as np

_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".perdim_shares.json")

def shares(bench, N=40000, bins=40, seed=7):
    """First-order variance shares by binned Var(E[Y|X_i])/Var(Y). Cached on disk."""
    try:
        c = json.load(open(_CACHE))
    except Exception:
        c = {}
    if bench in c:
        return np.asarray(c[bench], float)
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from benchmarks import get_benchmark
    torch.set_num_threads(1)
    hf = get_benchmark(f"{bench}_HF"); obj = hf["make_objective"]()
    LO = np.asarray(hf["domain_min"], float); HI = np.asarray(hf["domain_max"], float)
    d = len(LO); rng = np.random.RandomState(seed); X = rng.rand(N, d)
    Y = np.asarray(obj(torch.as_tensor(LO + (HI - LO) * X, dtype=torch.float64)), float).reshape(-1)
    VY = Y.var(); out = []
    for i in range(d):
        b = np.clip((X[:, i] * bins).astype(int), 0, bins - 1)
        m = np.array([Y[b == k].mean() if (b == k).any() else Y.mean() for k in range(bins)])
        w = np.array([(b == k).sum() for k in range(bins)])
        out.append(np.average((m - Y.mean()) ** 2, weights=w) / VY)
    o = np.asarray(out); o = o / o.sum()
    c[bench] = o.tolist()
    try: json.dump(c, open(_CACHE, "w"))
    except Exception: pass
    return o

def agg(per_dim, bench, unweighted=False):
    """Aggregate a per-dimension statistic. Weighted unless explicitly overridden."""
    v = np.asarray(per_dim, float)
    w = shares(bench)
    if v.shape[-1] != w.shape[0]:
        raise ValueError(f"{bench} has {w.shape[0]} dims, got {v.shape[-1]}")
    if unweighted:
        lo = float(w.min()); n_small = int((w < 0.02).sum())
        print(f"  [perdim] UNWEIGHTED aggregate requested for {bench}. "
              f"{n_small} of {w.shape[0]} dims carry <2% of variance "
              f"(smallest {100*lo:.2f}%). Five such averages were checked in this "
              f"project and five were misleading. Report the weighted value too.",
              file=sys.stderr)
        return v.mean(axis=-1)
    return (v * w).sum(axis=-1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(2)
    b = sys.argv[1]; w = shares(b)
    print(f"  {b}: {len(w)} dims")
    for i in np.argsort(-w):
        print(f"    d{i}: {100*w[i]:6.2f}%")
    n = int((w < 0.02).sum())
    print(f"  {n} of {len(w)} dims carry <2% of the variance"
          f" ({100*w[w<0.02].sum():.2f}% between them)")
    print(f"  an unweighted mean gives them {100*n/len(w):.0f}% of the weight")
