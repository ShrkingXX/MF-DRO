"""h178 SC1: does the emb field populate on the TRAINED modules?"""
import os, sys, importlib.util
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.argv = ["smoke", "Borehole_8D", "42", "random"]
_s = importlib.util.spec_from_file_location("h178w", os.path.join(H, "worker.py"))
w = importlib.util.module_from_spec(_s); sys.modules["h178w"] = w; _s.loader.exec_module(w)
import numpy as np
w.h83.BUDGET = 8.0
r = w.h83.run("Borehole_8D", "MF-DRO", 42, os.path.join(os.environ["SCRATCH"], "h178_smoke.json"))
P = r.get("h168_probe", [])
print(f"\n  probe iterations : {len(P)}")
if not P: print("  FAIL: no probe data"); raise SystemExit
e = P[-1].get("emb", {})
print(f"  emb field        : {e}")
if "error" in e: print(f"  SC1 FAIL -- {e['error']}"); raise SystemExit
ok = "rtg_resp" in e and "btg_resp" in e
print(f"  SC1              : {'PASS' if ok else 'FAIL'}")
if ok:
    rr = [p["emb"]["rtg_resp"] for p in P if "rtg_resp" in p.get("emb", {})]
    bb = [p["emb"]["btg_resp"] for p in P if "btg_resp" in p.get("emb", {})]
    print(f"\n  trained rtg_resp (0.30->1.00) : {np.mean(rr):.4f}   [random-weight estimate 0.4869]")
    print(f"  trained btg_resp (26.1->30.5) : {np.mean(bb):.4f}   [random-weight estimate 0.0056]")
    print(f"  ratio                          : {np.mean(rr)/max(np.mean(bb),1e-12):.1f}x  [predicted ~87x]")
    print(f"  (5 iterations, barely-trained -- NOT a verdict)")
