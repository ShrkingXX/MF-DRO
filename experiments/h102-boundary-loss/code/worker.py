"""H102 worker -- shim over h90's worker, adding one arm and nothing else.

h90's worker produced the confirmed ROI result and the NO-ROI control this
experiment reuses. Loading it (rather than copying) guarantees the treatment and
control differ only in the arm dict.
"""
import os, sys, importlib.util   # no numpy/torch before h90 sets its thread caps
H = os.path.dirname(os.path.abspath(__file__))
H90 = os.path.join(H, "..", "..", "h90-borehole-confirm", "code", "worker.py")
_spec = importlib.util.spec_from_file_location("h90_worker", H90)
h90 = importlib.util.module_from_spec(_spec); sys.modules["h90_worker"] = h90
_spec.loader.exec_module(h90)

RES = os.path.abspath(os.path.join(H, "..", "results"))
h90.RES = RES
h90.ARMS = dict(h90.ARMS)
# ROI off, regression head, L1 instead of MSE. Everything else is h90's NO-ROI.
h90.ARMS["L1-LOSS"] = dict(use_roi=False, loc_loss='l1')

if __name__ == "__main__":
    bench, arm, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    tag = f"{bench}__{arm}__seed{seed}"
    r = h90.run(bench, arm, seed, os.path.join(RES, "ckpt", tag + ".json"))
    h90._atomic(os.path.join(RES, tag + ".json"), r)
    ll = (r.get("L_loc_per_iter") or [None])[-1]
    print(f"[done] {bench} {arm} seed{seed} regret={r['final_regret']:.4f} "
          f"L_loc_final={ll} wall={r['_wall_s']/60:.1f}m", flush=True)
