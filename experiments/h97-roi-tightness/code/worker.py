"""H97 worker -- a SHIM over h90's worker, so ROI-Q05 differs from h90's ROI-Q10
by exactly one config value and nothing else.

h90's worker produced the confirmed ROI result. Copying it would fork it; this
loads it and overrides two module-level names: the results directory, and the
ARMS table (adding ROI-Q05 and keeping h90's entries byte-identical).
"""
import os, sys, importlib.util   # no numpy/torch before h90's thread-cap env vars
H = os.path.dirname(os.path.abspath(__file__))
H90 = os.path.join(H, "..", "..", "h90-borehole-confirm", "code", "worker.py")
_spec = importlib.util.spec_from_file_location("h90_worker", H90)
h90 = importlib.util.module_from_spec(_spec); sys.modules["h90_worker"] = h90
_spec.loader.exec_module(h90)

RES = os.path.abspath(os.path.join(H, "..", "results"))
h90.RES = RES
# same shape as h90's ROI-Q10, one value changed
h90.ARMS = dict(h90.ARMS)
h90.ARMS["ROI-Q05"] = dict(use_roi=True, roi_beta_mode='quantile', roi_target_accept=0.05)

if __name__ == "__main__":
    bench, arm, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    tag = f"{bench}__{arm}__seed{seed}"
    r = h90.run(bench, arm, seed, os.path.join(RES, "ckpt", tag + ".json"))
    h90._atomic(os.path.join(RES, tag + ".json"), r)
    rs = r.get("roi_summary") or {}
    print(f"[done] {bench} {arm} seed{seed} regret={r['final_regret']:.4f} "
          f"acc={rs.get('accept_frac')} beta={rs.get('beta_sqrt')} "
          f"wall={r['_wall_s']/60:.1f}m", flush=True)
