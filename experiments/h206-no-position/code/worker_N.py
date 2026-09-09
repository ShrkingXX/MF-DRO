"""h206N -- no positional embedding at all. usage: worker_N.py <bench> <seed>

Config is h205's worker verbatim except for disable_position_embedding, so the
only thing separating N, P and h205's B is the positional channel. max_seq_length
stays 256 in BOTH arms: it is inert when only indices 0..7 are labelled, and
holding it fixed keeps the arms differing in one thing.
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
    c.max_seq_length = 256
    c.absolute_timesteps = False           # h205 flags OFF in both h206 arms
    c.real_prefix_training = False
    c.disable_position_embedding = True
    return c

h83._build_mf_dro_config = _build

if __name__ == "__main__":
    bench, seed = sys.argv[1], int(sys.argv[2])
    tag = f"{bench}__H206N-NOPOS__seed{seed}"
    r = h83.run(bench, "MF-DRO", seed, os.path.join(RES, "ckpt", tag + ".json"))
    r["_h206"] = dict(arm="N", disable_position_embedding=True,
                      absolute_timesteps=False, real_prefix_training=False,
                      inference_context_k=8, max_seq_length=256, roi="Q10")
    h83._atomic(os.path.join(RES, tag + ".json"), r)
    print(f"[done] {tag} regret={r['final_regret']:.4f} lf_frac={r.get('lf_fraction')} "
          f"wall={r['_wall_s']/60:.1f}m", flush=True)
