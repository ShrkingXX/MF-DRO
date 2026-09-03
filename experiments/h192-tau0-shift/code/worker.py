"""h192 -- move the teacher's tau=0 action toward the box centre and see whether
the DT's emitted query follows. usage: worker.py <bench> <seed>

rollout_length=1 so the recorded all-tau teacher mean IS the tau=0 mean and the
imposed shift is directly measurable rather than averaged over eight steps. The
control is h172's ROLLOUT1, which is exactly this config minus the shift.
"""
import os, sys, importlib.util
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
_s = importlib.util.spec_from_file_location(
    "h83w", os.path.join(REPO, "experiments/h83-main-comparison/code/worker.py"))
h83 = importlib.util.module_from_spec(_s); sys.modules["h83w"] = h83; _s.loader.exec_module(h83)

RES = os.path.abspath(os.path.join(H, "..", "results")); h83.RES = RES
L = 1
LAM = 0.5                                  # halfway to the box centre, at tau=0 only
_ORIG_BUILD = h83._build_mf_dro_config


def _build(*a, **k):
    k["rollout_length"] = L
    cfg = _ORIG_BUILD(*a, **k)
    cfg.rollout_length = L                 # matches the ROLLOUT1 control
    cfg.tau0_shift_lambda = LAM            # the change under test
    return cfg


h83._build_mf_dro_config = _build

if __name__ == "__main__":
    bench, seed = sys.argv[1], int(sys.argv[2])
    tag = f"{bench}__TAU0SHIFT__seed{seed}"
    r = h83.run(bench, "MF-DRO", seed, os.path.join(RES, "ckpt", tag + ".json"))
    r["_h192"] = dict(rollout_length=L, tau0_shift_lambda=LAM)
    h83._atomic(os.path.join(RES, tag + ".json"), r)
    print(f"[done] {tag} regret={r['final_regret']:.4f} lf_frac={r.get('lf_fraction')} "
          f"wall={r['_wall_s']/60:.1f}m", flush=True)
