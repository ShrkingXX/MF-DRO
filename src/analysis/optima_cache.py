"""x* per benchmark, cached. dro_runner._KNOWN_OPTIMAL_X covers Hartmann and the
Ackleys only; Currin and Borehole have no argmax recorded anywhere (benchmarks.py
computes the optimal VALUE via _find_optimum_bounded but discards the location).
Without x* the stall diagnostic cannot tell "searching away from the optimum"
from "searching near it and missing", so it is computed here by multi-start
L-BFGS-B and cached to data/optima.json.
"""
import os, sys, json
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
CACHE = os.path.join(REPO, "data", "optima.json")

def x_star(bench):
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    c = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    if bench in c:
        return np.array(c[bench]["x"], float), float(c[bench]["f"])
    import torch
    from scipy.optimize import minimize
    from benchmarks import get_benchmark
    try:
        from dro_runner import _KNOWN_OPTIMAL_X
    except Exception:
        _KNOWN_OPTIMAL_X = {}
    b = get_benchmark(f"{bench}_HF")
    lo = np.array(b["domain_min"], float); hi = np.array(b["domain_max"], float)
    f = b["make_objective"]()
    fstar_known = -float(b["known_optimal_value"])
    def nf(x):
        with torch.no_grad():
            return -float(f(torch.tensor(np.clip(x, lo, hi), dtype=torch.float64)
                            .unsqueeze(0)).reshape(-1)[0])
    if bench in _KNOWN_OPTIMAL_X:                 # trust the recorded location
        bx = np.array(_KNOWN_OPTIMAL_X[bench], float); bv = -nf(bx)
    else:
        d = lo.size
        S = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=0)
        P = lo + (hi - lo) * S.draw(4096).numpy()
        vals = np.array([nf(p) for p in P])
        bx, bv = None, -np.inf
        for s in P[np.argsort(vals)[:24]]:        # 24 best starts
            r = minimize(nf, s, method="L-BFGS-B", bounds=list(zip(lo, hi)))
            if -r.fun > bv: bv, bx = -r.fun, np.clip(r.x, lo, hi)
    c[bench] = dict(x=list(map(float, bx)), f=float(bv),
                    f_known=fstar_known, gap=float(fstar_known - bv))
    json.dump(c, open(CACHE, "w"), indent=2)
    return np.asarray(bx, float), float(bv)

if __name__ == "__main__":
    for b in ("Currin_2D", "Hartmann_6D", "Borehole_8D"):
        x, v = x_star(b)
        from benchmarks import get_benchmark
        known = -float(get_benchmark(f"{b}_HF")["known_optimal_value"])
        print(f"  {b:<13} f*={v:>10.4f}  known={known:>10.4f}  gap={known-v:>+9.2e}  "
              f"x*={[round(t,4) for t in x[:4]]}{'...' if x.size>4 else ''}")
