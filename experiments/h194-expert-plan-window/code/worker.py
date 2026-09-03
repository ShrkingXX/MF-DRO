"""h194 Stage 1a -- the WINDOW alone. usage: worker.py <bench> <seed>

MES teacher + inference_context_k=8 + ROI-Q10. The control is the existing ROI-Q10 run
(11.59), which is this config with K=1, so the pair isolates the sliding window.
"""
import os, sys, importlib.util
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
_s = importlib.util.spec_from_file_location(
    "h83w", os.path.join(REPO, "experiments/h83-main-comparison/code/worker.py"))
h83 = importlib.util.module_from_spec(_s); sys.modules["h83w"] = h83; _s.loader.exec_module(h83)
RES = os.path.abspath(os.path.join(H, "..", "results")); h83.RES = RES
K = 8
_OB = h83._build_mf_dro_config


def _build(*a, **k):
    c = _OB(*a, **k)
    c.use_roi = True
    c.roi_beta_mode = 'quantile'
    c.roi_target_accept = 0.10          # ROI-Q10, matching the control
    c.inference_context_k = K           # the change under test
    return c


h83._build_mf_dro_config = _build

if __name__ == "__main__":
    bench, seed = sys.argv[1], int(sys.argv[2])
    tag = f"{bench}__WINDOW-K{K}__seed{seed}"
    r = h83.run(bench, "MF-DRO", seed, os.path.join(RES, "ckpt", tag + ".json"))
    r["_h194"] = dict(inference_context_k=K, roi="Q10")
    h83._atomic(os.path.join(RES, tag + ".json"), r)
    print(f"[done] {tag} regret={r['final_regret']:.4f} lf_frac={r.get('lf_fraction')} "
          f"wall={r['_wall_s']/60:.1f}m", flush=True)
