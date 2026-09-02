"""h149 RANDOM-POOL control. Sets rollout_policy='random' -- a PRE-EXISTING code
path (mf_dro.py:1567) that picks a uniformly random candidate from the same
roi_candidates pool the MES teacher argmaxes over. It does NOT touch forced_x.

If this also fails totally, any non-MES teacher does, and the forced_x hook used
by h145/h146 is exonerated. If it improves while the forced arms do not, the
failure tracks my hook and those results must be withdrawn.
"""
import os, sys, importlib.util
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
_s = importlib.util.spec_from_file_location(
    "h83w", os.path.join(REPO, "experiments/h83-main-comparison/code/worker.py"))
h83 = importlib.util.module_from_spec(_s); sys.modules["h83w"] = h83; _s.loader.exec_module(h83)

RES = os.path.abspath(os.path.join(H, "..", "results"))
h83.RES = RES
_ORIG_BUILD = h83._build_mf_dro_config


def _build(*a, **k):
    cfg = _ORIG_BUILD(*a, **k)
    cfg.rollout_policy = "random"      # the only change
    return cfg


h83._build_mf_dro_config = _build

if __name__ == "__main__":
    bench, seed = sys.argv[1], int(sys.argv[2])
    tag = f"{bench}__RANDOM-POOL__seed{seed}"
    r = h83.run(bench, "MF-DRO", seed, os.path.join(RES, "ckpt", tag + ".json"))
    h83._atomic(os.path.join(RES, tag + ".json"), r)
    print(f"[done] {tag} regret={r['final_regret']:.4f} wall={r['_wall_s']/60:.1f}m", flush=True)
