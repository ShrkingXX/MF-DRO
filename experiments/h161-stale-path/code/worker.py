"""h161 STALE-PATH. Identical to h153's MES-FROZEN two-pass wrapper except that
pass 2 replays a path taken LAG rollouts earlier -- model-selected, but against
a model that has since seen ~10 real iterations of new data.

Tests whether the teacher's locations must be model-selected FOR THE CURRENT
STATE (h161 fails) or merely model-selected at all (h161 works). h159 and h160
could not separate these; both accounts predicted success for both.
"""
import os, sys, importlib.util
import numpy as np, torch
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
_s = importlib.util.spec_from_file_location(
    "h83w", os.path.join(REPO, "experiments/h83-main-comparison/code/worker.py"))
h83 = importlib.util.module_from_spec(_s); sys.modules["h83w"] = h83; _s.loader.exec_module(h83)
import src.policy.mf_dro as MF

RES = os.path.abspath(os.path.join(H, "..", "results")); h83.RES = RES
_ORIG_SIM = MF.simulate_mf_trajectory
LAG = 600                       # 10 real iterations: the batch is 60 trajectories
                                # per iteration (STATE-DIAG n_traj=60), NOT the 200 of
                                # rollouts_per_iter. LAG=2000 would have been ~33 of the
                                # ~60 iterations, leaving over half the run in warmup and
                                # identical to h153. Caught by the smoke test before the
                                # arm was allowed to run.
_B = {"paths": [], "n": 0, "stale": 0, "lag_sum": 0, "err": 0.0, "flips": 0, "tot": 0}


def _stale_sim(*args, **kw):
    traj = _ORIG_SIM(*args, **kw)                       # pass 1: ordinary MES rollout
    if "actions_x" not in traj:
        return traj
    bounds = kw.get("bounds", args[6] if len(args) > 6 else None)
    lo, hi = bounds[0], bounds[1]
    cur = lo + traj["actions_x"].to(bounds.dtype) * (hi - lo)   # NORMALISED -> RAW
    _B["paths"].append(cur)

    i = len(_B["paths"]) - 1
    j = i - LAG
    if j >= 0:
        path = _B["paths"][j]; _B["stale"] += 1; _B["lag_sum"] += i - j
    else:
        path = cur                                      # warmup: no history yet
    _B["n"] += 1

    kw2 = dict(kw); kw2["forced_x"] = path
    out = _ORIG_SIM(*args, **kw2)                       # pass 2: replay the stale path

    if "actions_x" in out:                              # SC4
        a = ((path - lo) / (hi - lo))
        m = min(out["actions_x"].shape[0], a.shape[0])
        _B["err"] = max(_B["err"],
                        float((out["actions_x"][:m].to(a.dtype) - a[:m]).abs().max()))
    e1, e2 = traj.get("actions_ell"), out.get("actions_ell")
    if e1 is not None and e2 is not None:
        m = min(len(e1), len(e2))
        _B["flips"] += int((e1[:m] != e2[:m]).sum()); _B["tot"] += m
    return out


MF.simulate_mf_trajectory = _stale_sim

if __name__ == "__main__":
    bench, seed = sys.argv[1], int(sys.argv[2])
    tag = f"{bench}__STALE-PATH__seed{seed}"
    r = h83.run(bench, "MF-DRO", seed, os.path.join(RES, "ckpt", tag + ".json"))
    r["_h161"] = dict(LAG=LAG, n_rollouts=_B["n"],
                      sc1_stale_frac=_B["stale"] / max(_B["n"], 1),
                      sc2_mean_lag=_B["lag_sum"] / max(_B["stale"], 1),
                      sc4_path_max_abs_err=_B["err"],
                      fidelity_flip_frac=_B["flips"] / max(_B["tot"], 1))
    h83._atomic(os.path.join(RES, tag + ".json"), r)
    print(f"[done] {tag} regret={r['final_regret']:.4f} stale={r['_h161']['sc1_stale_frac']:.3f} "
          f"lag={r['_h161']['sc2_mean_lag']:.0f} SC4={r['_h161']['sc4_path_max_abs_err']:.1e} "
          f"wall={r['_wall_s']/60:.1f}m", flush=True)
