"""h168 SMOKE: does the probe actually produce data? The probe swallows
exceptions, so a scope error would look exactly like a silent no-op."""
import os, sys, importlib.util
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.argv = ["smoke", "Borehole_8D", "42", "random"]
_s = importlib.util.spec_from_file_location("h168w", os.path.join(H, "worker.py"))
w = importlib.util.module_from_spec(_s); sys.modules["h168w"] = w; _s.loader.exec_module(w)
import numpy as np
w.h83.BUDGET = 8.0
r = w.h83.run("Borehole_8D", "MF-DRO", 42, os.path.join(os.environ["SCRATCH"], "h168_smoke.json"))
P = r.get("h168_probe", [])
print(f"\n  probe iterations recorded : {len(P)}  -> {'PASS' if len(P) > 0 else 'FAIL (silent no-op)'}")
if P:
    e = P[0]
    print(f"  sweep length per iter     : {len(e['probes'])}  (expected {len(w.SWEEP)})")
    print(f"  real rtg_target at iter 0 : {e['rtg_target']:.4f}")
    X = np.array([p["x"] for p in e["probes"]])
    print(f"  emitted x varies with RTG : {not np.allclose(X, X[0])}"
          f"   (max spread across sweep {float(np.abs(X - X.mean(0)).max()):.4f})")
    for p in e["probes"][:3] + e["probes"][-2:]:
        d = float(np.linalg.norm(np.array(p["x"]) - 0.5))
        print(f"    rtg={p['rtg']:.2f}  d(box centre)={d:.4f}  ell={p['ell']}")
