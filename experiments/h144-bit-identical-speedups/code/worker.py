"""h144 gate worker: Borehole ROI-Q10 on the C1+C2 tree, for bit-identity
against h84's stored trace. Shim over h97/h90; arm copied from h84 line 20."""
import os, sys, importlib.util
H = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location(
    "h97w", os.path.join(H, "..", "..", "h97-roi-tightness", "code", "worker.py"))
h97 = importlib.util.module_from_spec(_s); sys.modules["h97w"] = h97; _s.loader.exec_module(h97)
RES = os.path.abspath(os.path.join(H, "..", "results"))
h97.h90.RES = RES
h97.h90.ARMS = dict(h97.h90.ARMS)
h97.h90.ARMS["ROI-Q10"] = dict(use_roi=True, roi_beta_mode='quantile', roi_target_accept=0.10)
if __name__ == "__main__":
    b, a, sd = sys.argv[1], sys.argv[2], int(sys.argv[3])
    tag = f"{b}__{a}__seed{sd}"
    r = h97.h90.run(b, a, sd, os.path.join(RES, "ckpt", tag + ".json"))
    h97.h90._atomic(os.path.join(RES, tag + ".json"), r)
    print(f"[done] {tag} regret={r['final_regret']:.4f} wall={r['_wall_s']/60:.1f}m", flush=True)
