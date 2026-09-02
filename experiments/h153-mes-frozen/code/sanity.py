"""h153 SC1-SC3 fast check: same worker shim, tiny budget (a few real
iterations) so a broken freeze is caught in a minute, not an hour."""
import os, sys, importlib.util
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
_s = importlib.util.spec_from_file_location("h153w", os.path.join(H, "worker.py"))
w = importlib.util.module_from_spec(_s); sys.modules["h153w"] = w
import numpy as np
_s.loader.exec_module(w)
import src.policy.mf_dro as MF
w.h83.BUDGET = 12.0
MF.simulate_mf_trajectory = w._frozen_sim
r = w.h83.run("Borehole_8D", "MF-DRO", 42, os.path.join(os.environ["SCRATCH"], "sc_ckpt.json"))
D = w._D; c, o = D["rtg0_closed"], D["rtg0_open"]
print(f"\n  rollouts wrapped        : {D['n']}   (no_actions_x={D['no_actions_x']})")
print(f"  SC1 path max |err|      : {D['path_max_abs_err']:.3e}  -> "
      f"{'PASS' if D['path_max_abs_err'] < 1e-9 else 'FAIL'}")
if c:
    pen = float(np.mean(c)) - float(np.mean(o))
    print(f"  SC2 rtg[0] closed       : {np.mean(c):+.4f}")
    print(f"  SC2 rtg[0] open (frozen): {np.mean(o):+.4f}")
    print(f"  SC2 open-loop penalty   : {pen:+.4f}  -> {'PASS' if pen > 0 else 'FAIL (no penalty)'}")
print(f"  SC3 fidelity flip frac  : {D['ell_flips']/max(D['ell_total'],1):.4f}"
      f"   ({D['ell_flips']}/{D['ell_total']})")
