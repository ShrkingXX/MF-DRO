"""h169 SMOKE: is the STATE axis actually populated? The probe swallows
exceptions, so a None/shape error would look exactly like a silent no-op and
h169 would degrade into a rerun of h168."""
import os, sys, importlib.util
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.argv = ["smoke", "Borehole_8D", "42", "random"]
_s = importlib.util.spec_from_file_location("h169w", os.path.join(H, "worker.py"))
w = importlib.util.module_from_spec(_s); sys.modules["h169w"] = w; _s.loader.exec_module(w)
import numpy as np
w.h83.BUDGET = 8.0
r = w.h83.run("Borehole_8D", "MF-DRO", 42, os.path.join(os.environ["SCRATCH"], "h169_smoke.json"))
P = r.get("h168_probe", [])
print(f"\n  probe iterations         : {len(P)}")
if not P:
    print("  FAIL: no probe data at all"); raise SystemExit
names = sorted({p["state"] for e in P for p in e["probes"]})
print(f"  state labels present     : {names}")
print(f"  STATE AXIS ACTIVE        : {'PASS' if any(n.startswith('train') for n in names) else 'FAIL (silent no-op -- h169 == h168)'}")
e = P[-1]
print(f"  probes per iteration     : {len(e['probes'])}  (expect 9 x n_states)")
import collections
by = collections.defaultdict(list)
for p in e["probes"]:
    by[p["state"]].append(float(np.linalg.norm(np.array(p["x"]) - 0.5)))
print(f"\n  {'state':10s} {'mean d(box centre)':>19s} {'n':>4s}")
for k in sorted(by): print(f"  {k:10s} {np.mean(by[k]):19.4f} {len(by[k]):4d}")
r_, t_ = by.get("real", []), [v for k, vs in by.items() if k.startswith("train") for v in vs]
if r_ and t_:
    print(f"\n  real {np.mean(r_):.4f}  vs  training {np.mean(t_):.4f}   ratio {np.mean(t_)/np.mean(r_):.3f}")
    print(f"  (P1 needs training >= 2x real; 5 iterations only, NOT a verdict)")
