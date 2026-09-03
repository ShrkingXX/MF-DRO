"""h196 -- the window WITH real past actions fed, per DT Algorithm 1.

usage: worker.py <bench> <seed>

Identical to h194's WINDOW-K8 arm (MES teacher, ROI-Q10, inference_context_k=8). The
only difference is in the core: _real_hist now records the action taken and propose_mf
fills the historical action slots with it instead of zeros. See h195 for the audit that
found the defect and h196/protocol.md for the SC that caught a silent no-op in the fix.
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
    c.roi_target_accept = 0.10
    c.inference_context_k = K
    return c


h83._build_mf_dro_config = _build

if __name__ == "__main__":
    bench, seed = sys.argv[1], int(sys.argv[2])
    tag = f"{bench}__H196-REALACT__seed{seed}"
    r = h83.run(bench, "MF-DRO", seed, os.path.join(RES, "ckpt", tag + ".json"))
    r["_h196"] = dict(inference_context_k=K, roi="Q10", real_past_actions=True)
    h83._atomic(os.path.join(RES, tag + ".json"), r)
    print(f"[done] {tag} regret={r['final_regret']:.4f} lf_frac={r.get('lf_fraction')} "
          f"wall={r['_wall_s']/60:.1f}m", flush=True)
