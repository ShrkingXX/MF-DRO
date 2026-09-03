"""h200B -- window(K=1) with the fidelity mix HELD FIXED. usage: worker_B.py <bench> <seed>

Both arms carry the SAME HF ceiling. Constraining only the window arm would make it
differ from the control in two ways at once, which is the very error this experiment
exists to correct.
"""
import os, sys, importlib.util
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
_s = importlib.util.spec_from_file_location(
    "h83w", os.path.join(REPO, "experiments/h83-main-comparison/code/worker.py"))
h83 = importlib.util.module_from_spec(_s); sys.modules["h83w"] = h83; _s.loader.exec_module(h83)
RES = os.path.abspath(os.path.join(H, "..", "results")); h83.RES = RES
_OB = h83._build_mf_dro_config

def _build(*a, **k):
    c = _OB(*a, **k)
    c.use_roi = True
    c.roi_beta_mode = 'quantile'
    c.roi_target_accept = 0.10
    c.inference_context_k = 1
    # CTRL-K1 realised lf_fraction 0.261 => HF fraction 0.739. The knob is a
    # realised-fraction controller (forces LF once realised HF >= ceiling), so it
    # drives the mix to the target rather than merely capping it.
    c.max_hf_fraction = 0.739
    return c

h83._build_mf_dro_config = _build

if __name__ == "__main__":
    bench, seed = sys.argv[1], int(sys.argv[2])
    tag = f"{bench}__H200B-CTRL-FIDMATCH__seed{seed}"
    r = h83.run(bench, "MF-DRO", seed, os.path.join(RES, "ckpt", tag + ".json"))
    r["_h200"] = dict(arm="B", inference_context_k=1, max_hf_fraction=0.739, roi="Q10")
    h83._atomic(os.path.join(RES, tag + ".json"), r)
    print(f"[done] {tag} regret={r['final_regret']:.4f} lf_frac={r.get('lf_fraction')} "
          f"wall={r['_wall_s']/60:.1f}m", flush=True)
