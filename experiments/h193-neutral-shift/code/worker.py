"""h193 -- displace the teacher's tau=0 action by the SAME amount as h192 but in a
DISTANCE-PRESERVING direction. usage: worker.py <bench> <seed>

h192 translated toward the box centre and got two effects at once: the DT followed
one-for-one, AND regret collapsed. The shift direction was not neutral, so the regret
half could not distinguish "the centre is bad" from "any displacement is bad".

This rotates about the centre by 34.0 degrees, reproducing h192's measured chord of
0.4585 while leaving the distance from the centre exactly unchanged.
"""
import os, sys, math, importlib.util
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
_s = importlib.util.spec_from_file_location(
    "h83w", os.path.join(REPO, "experiments/h83-main-comparison/code/worker.py"))
h83 = importlib.util.module_from_spec(_s); sys.modules["h83w"] = h83; _s.loader.exec_module(h83)

RES = os.path.abspath(os.path.join(H, "..", "results")); h83.RES = RES
L = 1
# Plane and angle chosen so the rotation circle FITS INSIDE THE BOX. The mean's
# in-plane radius must be <= 0.5, else rotation pushes a coordinate past the wall
# and clipping destroys the distance preservation the arm depends on. Dims (0,6)
# were tried first at r=0.5817 and the SC caught the clipping.
THETA = math.radians(56.7)          # chord 0.4585 at r=0.4830, matching h192
AXES  = (3, 6)                      # in-plane radius 0.4830 < 0.5, no clipping
_ORIG_BUILD = h83._build_mf_dro_config


def _build(*a, **k):
    k["rollout_length"] = L
    cfg = _ORIG_BUILD(*a, **k)
    cfg.rollout_length = L                 # matches ROLLOUT1 and h192
    cfg.tau0_shift_lambda = THETA          # radians in 'tangent' mode
    cfg.tau0_shift_mode = 'tangent'        # the change vs h192
    cfg.tau0_rot_axes = AXES
    return cfg


h83._build_mf_dro_config = _build

if __name__ == "__main__":
    bench, seed = sys.argv[1], int(sys.argv[2])
    tag = f"{bench}__TAU0ROT__seed{seed}"
    r = h83.run(bench, "MF-DRO", seed, os.path.join(RES, "ckpt", tag + ".json"))
    r["_h193"] = dict(rollout_length=L, theta_deg=56.7, axes=AXES, mode="tangent")
    h83._atomic(os.path.join(RES, tag + ".json"), r)
    print(f"[done] {tag} regret={r['final_regret']:.4f} lf_frac={r.get('lf_fraction')} "
          f"wall={r['_wall_s']/60:.1f}m", flush=True)
