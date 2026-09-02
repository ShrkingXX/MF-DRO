"""h167b -- is the box centre the LOSS-OPTIMAL constant for each teacher?

If the DT sat at the targets' mean, L_loc would match the target-mean constant.
If it sits at the box centre (h167), and the centre is NOT loss-optimal, then
training-time fit and inference-time output disagree.
"""
import os, sys, json
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from benchmarks import get_benchmark

T, N = 8, 20000
hf = get_benchmark("Borehole_8D_HF")
lo = np.array(hf["domain_min"], float); hi = np.array(hf["domain_max"], float); d = len(lo)
f_hf = hf["make_objective"]()
XSTAR = (np.array([0.15, 100.0, 95090.9777, 1110.0, 116.0, 700.0, 1120.0, 12045.0]) - lo) / (hi - lo)
rng = np.random.default_rng(0)
frac = (np.arange(T) / (T - 1))[:, None]

def targets(kind):
    """Teacher action targets in NORMALISED space, exactly as actions_x stores them."""
    if kind == "RANDOM":
        return rng.random((N * T, d))
    if kind == "MES":          # control: not analytically available; skipped
        return None
    A = []
    for _ in range(N):
        s = rng.random(d)
        if kind == "ORACLE":
            e = XSTAR
        else:                  # DIVERSE-GOOD: argmax of 256 true-objective draws
            c = lo + (hi - lo) * rng.random((256, d))
            e = (c[np.asarray(f_hf(__import__("torch").tensor(c)).reshape(-1)).argmax()] - lo) / (hi - lo)
        A.append(s + (e - s) * frac)
    return np.concatenate(A, axis=0)

print(f"\n  MSE of a CONSTANT predictor against each teacher's action targets")
print(f"  (normalised space, mean over {d} dims -- the same convention as L_loc)\n")
print(f"  {'teacher':16s} {'target mean':>12s} {'MSE @ centre':>14s} {'MSE @ tgt mean':>15s} {'observed L_loc':>15s}")
OBS = {"RANDOM": "0.018-0.022", "ORACLE": "0.018-0.022", "DIVERSE-GOOD": "0.018-0.022"}
for k in ("RANDOM", "ORACLE", "DIVERSE-GOOD"):
    A = targets(k)
    m = A.mean(axis=0)
    mse_c = float(((A - 0.5) ** 2).mean())
    mse_m = float(((A - m) ** 2).mean())
    print(f"  {k:16s} {np.round(m,2)[:3]}.. {mse_c:14.4f} {mse_m:15.4f} {OBS[k]:>15s}")
print(f"\n  control (MES) observed L_loc 0.040 -- its targets are model-dependent,")
print(f"  so no analytic constant baseline exists; excluded rather than guessed.")
