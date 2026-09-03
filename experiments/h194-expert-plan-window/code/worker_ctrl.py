"""h194 Stage 1a CONTROL -- identical to the WINDOW arm but inference_context_k=1.

Run today, on the same code, so the window comparison does not rest on h84's Aug-27
ROI-Q10 across 17 intervening commits.
"""
import os, sys, importlib.util
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
_s = importlib.util.spec_from_file_location(
    "h83w", os.path.join(REPO, "experiments/h83-main-comparison/code/worker.py"))
h83 = importlib.util.module_from_spec(_s); sys.modules["h83w"] = h83; _s.loader.exec_module(h83)
RES = os.path.abspath(os.path.join(H, "..", "results")); h83.RES = RES
K = 1
_OB = h83._build_mf_dro_config


def _build(*a, **k):
    c = _OB(*a, **k)
    c.use_roi = True
    c.roi_beta_mode = 'quantile'
    c.roi_target_accept = 0.10
    c.inference_context_k = K
    return c


h83._build_mf_dro_config = _build

if __name__ == "__main__":
    bench, seed = sys.argv[1], int(sys.argv[2])
    tag = f"{bench}__CTRL-K1__seed{seed}"
    r = h83.run(bench, "MF-DRO", seed, os.path.join(RES, "ckpt", tag + ".json"))
    r["_h194"] = dict(inference_context_k=K, roi="Q10", role="contemporaneous control")
    h83._atomic(os.path.join(RES, tag + ".json"), r)
    print(f"[done] {tag} regret={r['final_regret']:.4f} lf_frac={r.get('lf_fraction')} "
          f"wall={r['_wall_s']/60:.1f}m", flush=True)
