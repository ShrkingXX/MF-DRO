"""h153 MES-FROZEN worker. Shim over h83's worker so config, initial design,
cost budget and metric are IDENTICAL to the MF-DRO control. The ONE difference:
the rollout teacher cannot adapt.

Pass 1 runs the ordinary closed-loop MES rollout and yields its path. Pass 2
re-runs the same rollout with that path frozen through forced_x. The frozen
teacher's trajectories therefore have exactly the control's quality -- same
rule, same state, same pool -- and differ only in being unable to react to the
fantasy draws of the run they are fed into.

UNIT CONVERSION: traj['actions_x'] is NORMALISED to [0,1]^d (mf_dro.py:1883);
forced_x is consumed as RAW domain (mf_dro.py:1619). De-normalising is
mandatory -- h145 was unaffected only because it built its path in raw
coordinates from the start.
"""
import os, sys, json, importlib.util
import numpy as np
import torch

H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
_s = importlib.util.spec_from_file_location(
    "h83w", os.path.join(REPO, "experiments/h83-main-comparison/code/worker.py"))
h83 = importlib.util.module_from_spec(_s); sys.modules["h83w"] = h83; _s.loader.exec_module(h83)
import src.policy.mf_dro as MF

RES = os.path.abspath(os.path.join(H, "..", "results"))
h83.RES = RES

_ORIG_SIM = MF.simulate_mf_trajectory
# SC diagnostics accumulated across every rollout of the run.
_D = {"n": 0, "path_max_abs_err": 0.0, "rtg0_open": [], "rtg0_closed": [],
      "ell_flips": 0, "ell_total": 0, "no_actions_x": 0}


def _frozen_sim(*args, **kw):
    traj = _ORIG_SIM(*args, **kw)                      # pass 1: closed-loop MES
    if "actions_x" not in traj:                        # candidate-scoring mode
        _D["no_actions_x"] += 1
        return traj

    bounds = kw.get("bounds", args[6] if len(args) > 6 else None)
    lo, hi = bounds[0], bounds[1]
    a = traj["actions_x"].to(dtype=bounds.dtype)
    path = lo + a * (hi - lo)                          # NORMALISED -> RAW

    kw2 = dict(kw); kw2["forced_x"] = path
    out = _ORIG_SIM(*args, **kw2)                      # pass 2: same path, frozen

    _D["n"] += 1
    if "rtg" in traj and len(traj["rtg"]) and "rtg" in out and len(out["rtg"]):
        _D["rtg0_closed"].append(float(traj["rtg"][0]))
        _D["rtg0_open"].append(float(out["rtg"][0]))
    if "actions_x" in out:                             # SC1
        m = min(out["actions_x"].shape[0], a.shape[0])
        _D["path_max_abs_err"] = max(
            _D["path_max_abs_err"],
            float((out["actions_x"][:m].to(a.dtype) - a[:m]).abs().max()))
    e1, e2 = traj.get("actions_ell"), out.get("actions_ell")               # SC3
    if e1 is not None and e2 is not None:
        m = min(len(e1), len(e2))
        _D["ell_flips"] += int((e1[:m] != e2[:m]).sum()); _D["ell_total"] += m
    return out


if __name__ == "__main__":
    bench, seed = sys.argv[1], int(sys.argv[2])
    MF.simulate_mf_trajectory = _frozen_sim
    tag = f"{bench}__MES-FROZEN__seed{seed}"
    r = h83.run(bench, "MF-DRO", seed, os.path.join(RES, "ckpt", tag + ".json"))
    c, o = _D["rtg0_closed"], _D["rtg0_open"]
    r["_h153"] = dict(
        n_rollouts=_D["n"], no_actions_x=_D["no_actions_x"],
        sc1_path_max_abs_err=_D["path_max_abs_err"],
        sc2_rtg0_closed=float(np.mean(c)) if c else None,
        sc2_rtg0_open=float(np.mean(o)) if o else None,
        sc2_open_loop_penalty=(float(np.mean(c)) - float(np.mean(o))) if c else None,
        sc3_ell_flip_frac=(_D["ell_flips"] / _D["ell_total"]) if _D["ell_total"] else None)
    h83._atomic(os.path.join(RES, tag + ".json"), r)
    print(f"[done] {tag} regret={r['final_regret']:.4f} SC1={_D['path_max_abs_err']:.2e} "
          f"SC2pen={r['_h153']['sc2_open_loop_penalty']} SC3flip={r['_h153']['sc3_ell_flip_frac']} "
          f"wall={r['_wall_s']/60:.1f}m", flush=True)
