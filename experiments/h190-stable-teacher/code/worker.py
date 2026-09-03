"""h190 -- teacher-only WITH a stable fidelity allocation. usage: worker.py <bench> <seed>

Imports h187's worker, which patches DirectMFRegretOptimization.__init__ to replace
dt.propose_mf with the acquisition rule (teacher decides, DT bypassed). On top of that
it sets max_hf_fraction, the HF ceiling built and identity-gated in h184, so the
teacher's realised lf_fraction is pinned near MF-DRO's 0.800 instead of swinging
between 0.000 and 0.989.

Importing h187's worker is deliberate: the teacher mechanism must be IDENTICAL to
h189's, or the comparison measures two changes at once.
"""
import os, sys, importlib.util
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)

# h187's module applies the teacher patch at import time.
_s = importlib.util.spec_from_file_location(
    "h187w", os.path.join(REPO, "experiments/h187-no-dt-frozen/code/worker.py"))
h187 = importlib.util.module_from_spec(_s); sys.modules["h187w"] = h187; _s.loader.exec_module(h187)

h83 = h187.h83
RES = os.path.abspath(os.path.join(H, "..", "results")); h83.RES = RES
MAX_HF = 0.20                      # -> lf_fraction ~0.80, matching MF-DRO on Hartmann
_ORIG_BUILD = h83._build_mf_dro_config


def _build(*a, **k):
    cfg = _ORIG_BUILD(*a, **k)
    cfg.max_hf_fraction = MAX_HF   # the change under test
    return cfg


h83._build_mf_dro_config = _build

if __name__ == "__main__":
    bench, seed = sys.argv[1], int(sys.argv[2])
    tag = f"{bench}__STABLE-NODT__seed{seed}"
    r = h83.run(bench, "MF-DRO", seed, os.path.join(RES, "ckpt", tag + ".json"))
    h83._atomic(os.path.join(RES, tag + ".json"), r)
    print(f"[done] {tag} regret={r['final_regret']:.4f} lf_frac={r.get('lf_fraction')} "
          f"wall={r['_wall_s']/60:.1f}m", flush=True)
