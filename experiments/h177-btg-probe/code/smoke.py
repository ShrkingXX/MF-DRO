"""h177 SC1: does the BTG axis populate? The probe swallows exceptions, so a
scope error looks exactly like a silent no-op. h169 was lost to skipping this."""
import os, sys, importlib.util
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.argv = ["smoke", "Borehole_8D", "42", "random"]
_s = importlib.util.spec_from_file_location("h177w", os.path.join(H, "worker.py"))
w = importlib.util.module_from_spec(_s); sys.modules["h177w"] = w; _s.loader.exec_module(w)
import numpy as np
w.h83.BUDGET = 8.0
r = w.h83.run("Borehole_8D", "MF-DRO", 42, os.path.join(os.environ["SCRATCH"], "h177_smoke.json"))
P = r.get("h168_probe", [])
print(f"\n  probe iterations      : {len(P)}")
if not P: print("  FAIL: no probe data"); raise SystemExit
e = P[-1]
b = e.get("btg_probes", [])
print(f"  BTG probes per iter   : {len(b)}  (expect {len(w.BTG_SWEEP)})")
print(f"  BTG AXIS ACTIVE       : {'PASS' if len(b) == len(w.BTG_SWEEP) else 'FAIL (silent no-op)'}")
if b:
    X = np.array([p["x"] for p in b])
    d = np.linalg.norm(X - 0.5, axis=1)
    print(f"  real btg_now at iter  : {e.get('btg_now'):.3f}")
    print(f"\n  {'BTG':>7s} {'d(box centre)':>14s}")
    for p, dd in zip(b, d): print(f"  {p['btg']:7.1f} {dd:14.4f}")
    print(f"\n  spread across sweep   : {d.max()-d.min():.4f} ({(d.max()-d.min())/d.mean()*100:.1f}% of mean)")
    print(f"  (5 iterations only -- NOT a verdict; h168's RTG sweep gave 8.9% at full length)")
