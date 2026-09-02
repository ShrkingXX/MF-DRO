"""h179 SC1: does standardisation actually reach the modules, and does it make
the conditioning responsive? A flag that never fires would look like a null."""
import os, sys, importlib.util
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
import numpy as np, torch
import src.policy.mf_dro as MF
CAP = {}
_OI = MF.DirectMFRegretOptimization.__init__
def _init(self, *a, **k):
    _OI(self, *a, **k); CAP["mf"] = self
MF.DirectMFRegretOptimization.__init__ = _init
sys.argv = ["s", "Borehole_8D", "42"]
_s = importlib.util.spec_from_file_location("h179w", os.path.join(H, "worker.py"))
w = importlib.util.module_from_spec(_s); sys.modules["h179w"] = w; _s.loader.exec_module(w)
w.h83.BUDGET = 8.0
w.h83.run("Borehole_8D", "MF-DRO", 42, os.path.join(os.environ["SCRATCH"], "h179_smoke.json"))
dt = CAP["mf"].dt
print(f"\n  standardize_conditioning flag : {getattr(dt,'standardize_conditioning',None)}")
print(f"  running stats updated (n)     : {float(dt._cond_n):.0f}")
print(f"  _cond_mu  [rtg, btg]          : {[round(float(v),4) for v in dt._cond_mu]}")
print(f"  _cond_sd  [rtg, btg]          : {[round(float(v),4) for v in dt._cond_sd]}")
ok = getattr(dt,'standardize_conditioning',False) and float(dt._cond_n) > 0
print(f"  SC1                           : {'PASS' if ok else 'FAIL (flag never fired)'}")
if ok:
    mu, sd = dt._cond_mu, dt._cond_sd
    def resp(lin, ln, lo, hi, m, s):
        a = ln(lin(torch.tensor([[[(lo-float(m))/float(s)]]], dtype=lin.weight.dtype)))
        b = ln(lin(torch.tensor([[[(hi-float(m))/float(s)]]], dtype=lin.weight.dtype)))
        return float((a-b).norm()/a.norm())
    print(f"\n  trained response over the OBSERVED ranges, now standardised:")
    print(f"    rtg (0.30->1.00): {resp(dt.reward_embedding, dt.reward_ln, 0.30, 1.00, mu[0], sd[0]):.4f}")
    print(f"    btg (26.1->30.5): {resp(dt.btg_embed, dt.btg_ln, 26.1, 30.5, mu[1], sd[1]):.4f}"
          f"   [was 0.0056 raw]")
