"""h139 P1: Borehole ROI-FIX2 on the PATCHED tree, so roi_stats records carry
`n_real_iter` and the per-iteration acceptance trajectory is recoverable.

Shim over h97's worker (itself a shim over h90's), identical to the h84 ROI-FIX2
arm in configuration -- roi_beta_mode='fixed', roi_beta_sqrt=2.0 -- and differing
only in the results directory. The ARM DEFINITION IS COPIED VERBATIM from
experiments/h84-roi-strategy/code/worker.py:19 so the run is comparable to the
h84 FIX2 results every h139 claim rests on.

DO NOT LAUNCH BEFORE h136's GATE PASSES. The patch this run exists to exploit is
ungated until then, and a logged array from an unproven build is not evidence.
"""
import os, sys, importlib.util
H = os.path.dirname(os.path.abspath(__file__))
H97 = os.path.join(H, "..", "..", "h97-roi-tightness", "code", "worker.py")
_spec = importlib.util.spec_from_file_location("h97_worker", H97)
h97 = importlib.util.module_from_spec(_spec); sys.modules["h97_worker"] = h97
_spec.loader.exec_module(h97)

RES = os.path.abspath(os.path.join(H, "..", "results"))
h97.h90.RES = RES
h97.h90.ARMS = dict(h97.h90.ARMS)
h97.h90.ARMS["ROI-FIX2"] = dict(use_roi=True, roi_beta_mode='fixed', roi_beta_sqrt=2.0)

if __name__ == "__main__":
    bench, arm, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    tag = f"{bench}__{arm}__seed{seed}"
    r = h97.h90.run(bench, arm, seed, os.path.join(RES, "ckpt", tag + ".json"))
    h97.h90._atomic(os.path.join(RES, tag + ".json"), r)
    rs = r.get("roi_summary") or {}
    print(f"[done] {bench} {arm} seed{seed} regret={r['final_regret']:.4f} "
          f"acc={rs.get('accept_frac')} wall={r['_wall_s']/60:.1f}m", flush=True)
