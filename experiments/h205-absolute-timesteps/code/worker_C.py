"""h205C -- absolute + prefix. usage: worker_C.py <bench> <seed>

All arms K=8 (the window is what exposes the positional/phase confound; at K=1
the readout is index 0 and there is nothing to fix). Control is h194 CTRL-K1
(11.59), already in hand.
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
    c.use_roi = True                       # ROI-Q10, matching every control
    c.roi_beta_mode = 'quantile'
    c.roi_target_accept = 0.10
    c.inference_context_k = 8
    c.max_seq_length = 256                 # real horizon reaches 159 (Hartmann) + 8
    c.absolute_timesteps = True
    c.real_prefix_training = True
    return c

h83._build_mf_dro_config = _build

if __name__ == "__main__":
    bench, seed = sys.argv[1], int(sys.argv[2])
    tag = f"{bench}__H205C-BOTH__seed{seed}"
    r = h83.run(bench, "MF-DRO", seed, os.path.join(RES, "ckpt", tag + ".json"))
    r["_h205"] = dict(arm="C", absolute_timesteps=True, real_prefix_training=True,
                      inference_context_k=8, max_seq_length=256, roi="Q10")
    h83._atomic(os.path.join(RES, tag + ".json"), r)
    print(f"[done] {tag} regret={r['final_regret']:.4f} lf_frac={r.get('lf_fraction')} "
          f"wall={r['_wall_s']/60:.1f}m", flush=True)
