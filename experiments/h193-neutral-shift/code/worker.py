"""h193 -- separate "the DT follows its teacher's mean" from "the centre is bad".

usage: worker.py <bench> <seed> <centre|rot>

Both arms displace the teacher's tau=0 action by the SAME amount (0.2006 in the
centred unit box, matching a lambda=0.5 pull toward the centre on Hartmann's control
mean of 0.4013). They differ ONLY in direction:

  centre : translate toward the box centre     -> distance from centre HALVES
  rot    : rotate about the centre in dims (1,5) by 31.3 deg -> distance PRESERVED

Run on HARTMANN. The Borehole version was abandoned before launch: 80.9% of its
control queries sit within 0.05 of a box wall, so no distance-preserving move exists
there -- three designs failed the SC and the cause was measured, not guessed.
Hartmann has 0 wall-pinned dims.
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
ARM = sys.argv[3] if len(sys.argv) > 3 else "rot"
LAM_CENTRE = 0.5
THETA_ROT = math.radians(31.3)      # chord 0.2006 at in-plane r=0.3713
AXES = (1, 5)                       # Hartmann control mean's largest in-box plane
_ORIG_BUILD = h83._build_mf_dro_config


def _build(*a, **k):
    k["rollout_length"] = L
    cfg = _ORIG_BUILD(*a, **k)
    cfg.rollout_length = L
    if ARM == "centre":
        cfg.tau0_shift_lambda = LAM_CENTRE
        cfg.tau0_shift_mode = "centre"
    else:
        cfg.tau0_shift_lambda = THETA_ROT
        cfg.tau0_shift_mode = "tangent"
        cfg.tau0_rot_axes = AXES
    return cfg


h83._build_mf_dro_config = _build

if __name__ == "__main__":
    bench, seed = sys.argv[1], int(sys.argv[2])
    tag = f"{bench}__H193-{ARM.upper()}__seed{seed}"
    r = h83.run(bench, "MF-DRO", seed, os.path.join(RES, "ckpt", tag + ".json"))
    r["_h193"] = dict(arm=ARM, rollout_length=L,
                      lam=LAM_CENTRE if ARM == "centre" else None,
                      theta_deg=None if ARM == "centre" else 31.3,
                      axes=None if ARM == "centre" else AXES)
    h83._atomic(os.path.join(RES, tag + ".json"), r)
    print(f"[done] {tag} regret={r['final_regret']:.4f} lf_frac={r.get('lf_fraction')} "
          f"wall={r['_wall_s']/60:.1f}m", flush=True)
