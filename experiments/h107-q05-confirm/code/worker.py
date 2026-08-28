"""Shim over h97's worker (itself a shim over h90's), so this arm is identical to
h97's ROI-Q05 in everything but the seeds. Overrides only the results directory."""
import os, sys, importlib.util
H = os.path.dirname(os.path.abspath(__file__))
H97 = os.path.join(H, "..", "..", "h97-roi-tightness", "code", "worker.py")
_spec = importlib.util.spec_from_file_location("h97_worker", H97)
h97 = importlib.util.module_from_spec(_spec); sys.modules["h97_worker"] = h97
_spec.loader.exec_module(h97)
RES = os.path.abspath(os.path.join(H, "..", "results"))
h97.h90.RES = RES
if __name__ == "__main__":
    bench, arm, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    tag = f"{bench}__{arm}__seed{seed}"
    r = h97.h90.run(bench, arm, seed, os.path.join(RES, "ckpt", tag + ".json"))
    h97.h90._atomic(os.path.join(RES, tag + ".json"), r)
    rs = r.get("roi_summary") or {}
    print(f"[done] {bench} {arm} seed{seed} regret={r['final_regret']:.4f} "
          f"acc={rs.get('accept_frac')} wall={r['_wall_s']/60:.1f}m", flush=True)
