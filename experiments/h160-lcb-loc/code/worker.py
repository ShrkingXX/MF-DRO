"""h160 LCB-LOC. Identical to h155's UCB-LOC except ucb_loc_beta=-2.0, i.e. the
location is the argmax of the LOWER confidence bound (mu - 2*sigma): points that
are confidently mediocre -- low predicted value AND low uncertainty, exactly
where the model expects to learn least. Closed-loop, adaptive and
model-selected in the strictest sense, yet anti-informative by construction.

Injected via a kwarg wrapper so src/policy/mf_dro.py is not modified again and
h155's bit-identity gate stands unchanged.
"""
import os, sys, importlib.util
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
_s = importlib.util.spec_from_file_location(
    "h83w", os.path.join(REPO, "experiments/h83-main-comparison/code/worker.py"))
h83 = importlib.util.module_from_spec(_s); sys.modules["h83w"] = h83; _s.loader.exec_module(h83)
import src.policy.mf_dro as MF

RES = os.path.abspath(os.path.join(H, "..", "results")); h83.RES = RES
_ORIG_BUILD = h83._build_mf_dro_config
_ORIG_SIM = MF.simulate_mf_trajectory


def _build(*a, **k):
    cfg = _ORIG_BUILD(*a, **k)
    cfg.rollout_policy = "ucb_loc"
    return cfg


def _sim(*a, **k):
    k["ucb_loc_beta"] = -2.0           # the only difference from h155
    return _ORIG_SIM(*a, **k)


h83._build_mf_dro_config = _build
MF.simulate_mf_trajectory = _sim

if __name__ == "__main__":
    bench, seed = sys.argv[1], int(sys.argv[2])
    tag = f"{bench}__LCB-LOC__seed{seed}"
    r = h83.run(bench, "MF-DRO", seed, os.path.join(RES, "ckpt", tag + ".json"))
    h83._atomic(os.path.join(RES, tag + ".json"), r)
    print(f"[done] {tag} regret={r['final_regret']:.4f} lf_frac={r.get('lf_fraction')} "
          f"wall={r['_wall_s']/60:.1f}m", flush=True)
