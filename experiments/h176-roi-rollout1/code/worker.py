"""h176 -- ROI-Q10 (h84's config, verbatim) combined with rollout_length=1."""
import os, sys, importlib.util
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
_s = importlib.util.spec_from_file_location(
    "h83w", os.path.join(REPO, "experiments/h83-main-comparison/code/worker.py"))
h83 = importlib.util.module_from_spec(_s); sys.modules["h83w"] = h83; _s.loader.exec_module(h83)

RES = os.path.abspath(os.path.join(H, "..", "results")); h83.RES = RES
_ORIG_BUILD = h83._build_mf_dro_config
L = 1


def _build(*a, **k):
    k["rollout_length"] = L
    cfg = _ORIG_BUILD(*a, **k)
    cfg.rollout_length = L
    cfg.use_roi = True                     # h84 ROI-Q10, verbatim
    cfg.roi_beta_mode = 'quantile'
    cfg.roi_target_accept = 0.10
    return cfg


h83._build_mf_dro_config = _build

if __name__ == "__main__":
    bench, seed = sys.argv[1], int(sys.argv[2])
    tag = f"{bench}__ROI-Q10-L1__seed{seed}"
    r = h83.run(bench, "MF-DRO", seed, os.path.join(RES, "ckpt", tag + ".json"))
    r["_h176"] = dict(rollout_length=L, roi="Q10")
    h83._atomic(os.path.join(RES, tag + ".json"), r)
    print(f"[done] {tag} regret={r['final_regret']:.4f} lf_frac={r.get('lf_fraction')} "
          f"wall={r['_wall_s']/60:.1f}m", flush=True)
