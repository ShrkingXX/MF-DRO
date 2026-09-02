"""h161 SMOKE TEST. Verifies the stale-path wrapper actually engages, using a
small LAG and a tiny budget so a broken wrapper is caught in minutes rather
than after 10 worker-hours. Checks SC1 (stale fraction), SC2 (mean lag applied)
and SC4 (pass-2 replays the STALE path exactly, not the current one)."""
import os, sys, importlib.util
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
_s = importlib.util.spec_from_file_location("h161w", os.path.join(H, "worker.py"))
w = importlib.util.module_from_spec(_s); sys.modules["h161w"] = w; _s.loader.exec_module(w)
import src.policy.mf_dro as MF
import numpy as np, torch

w.LAG = 60                      # ~0.3 iterations, so staleness engages immediately
w._B["paths"].clear()
w.h83.BUDGET = 10.0
MF.simulate_mf_trajectory = w._stale_sim
r = w.h83.run("Borehole_8D", "MF-DRO", 42, os.path.join(os.environ["SCRATCH"], "h161_smoke.json"))
B = w._B
sf = B["stale"] / max(B["n"], 1)
ml = B["lag_sum"] / max(B["stale"], 1)
print(f"\n  rollouts wrapped          : {B['n']}")
print(f"  SC1 stale fraction        : {sf:.3f}   -> {'PASS' if sf > 0.5 else 'FAIL (wrapper not engaging)'}")
print(f"  SC2 mean lag applied      : {ml:.0f} rollouts (LAG set to {w.LAG})  -> "
      f"{'PASS' if abs(ml - w.LAG) < 1e-6 else 'FAIL'}")
print(f"  SC4 pass-2 path max |err| : {B['err']:.3e}  -> {'PASS' if B['err'] < 1e-9 else 'FAIL'}")
print(f"  fidelity flip frac        : {B['flips']/max(B['tot'],1):.3f}")
# The decisive check: the replayed path must differ from the CURRENT pass-1 path,
# otherwise "stale" is a no-op and the arm tests nothing.
P = B["paths"]
if len(P) > w.LAG + 5:
    d = [float((P[i] - P[i - w.LAG]).abs().max()) for i in range(w.LAG, min(len(P), w.LAG + 50))]
    print(f"  STALE vs CURRENT path differ: max|diff| median {np.median(d):.4g}  "
          f"zero in {sum(1 for x in d if x < 1e-12)}/{len(d)}  -> "
          f"{'PASS' if np.median(d) > 1e-6 else 'FAIL (stale path == current path)'}")
else:
    print("  STALE vs CURRENT: too few rollouts to compare")
