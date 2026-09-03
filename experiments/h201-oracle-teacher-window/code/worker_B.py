"""h201B -- h145's interpolating oracle teacher with inference_context_k=1.

CEILING/DIAGNOSTIC, NOT A METHOD: x* is unavailable at run time. The teacher is
installed exactly as h145 does (wrapping simulate_mf_trajectory with forced_x), which
is orthogonal to the window config, so the two arms differ ONLY by inference_context_k.
"""
import os, sys, importlib.util
import torch
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
_s = importlib.util.spec_from_file_location(
    "h83w", os.path.join(REPO, "experiments/h83-main-comparison/code/worker.py"))
h83 = importlib.util.module_from_spec(_s); sys.modules["h83w"] = h83; _s.loader.exec_module(h83)
RES = os.path.abspath(os.path.join(H, "..", "results")); h83.RES = RES
import src.policy.mf_dro as MF

XSTAR = {"Hartmann_6D": [0.2017, 0.1500, 0.4769, 0.2753, 0.3116, 0.6573],
         "Borehole_8D": [0.15, 100.0, 95090.9777, 1110.0, 116.0, 700.0, 1120.0, 12045.0]}
_ORIG_SIM = MF.simulate_mf_trajectory
_EXPERT = {"x_star": None, "rng": None, "n": 0, "tau7_max_dev": 0.0, "tau0": []}

def _expert_path(bounds, T, d):
    lo, hi = bounds[0], bounds[1]
    x_start = lo + (hi - lo) * torch.rand(d, generator=_EXPERT["rng"], dtype=bounds.dtype)
    xs = _EXPERT["x_star"].to(dtype=bounds.dtype)
    return torch.stack([x_start + (xs - x_start) * (t / max(T - 1, 1)) for t in range(T)])

def _expert_sim(*args, **kw):
    bounds = kw.get("bounds")
    if bounds is None:
        bounds = args[6]
    T = kw.get("rollout_length", args[3] if len(args) > 3 else 8)
    d = int(bounds.shape[1])
    path = _expert_path(bounds, int(T), d)
    # SC2, accumulated in-run: the LAST step must be x* exactly, and tau=0 must vary.
    _xs = _EXPERT["x_star"].to(dtype=bounds.dtype)
    _EXPERT["tau7_max_dev"] = max(_EXPERT["tau7_max_dev"],
                                  float((path[-1] - _xs).abs().max()))
    _EXPERT["tau0"].append(((path[0]-bounds[0])/(bounds[1]-bounds[0])).reshape(-1).clone())
    _EXPERT["n"] += 1
    kw["forced_x"] = path
    return _ORIG_SIM(*args, **kw)

_OB = h83._build_mf_dro_config
def _build(*a, **k):
    c = _OB(*a, **k)
    c.use_roi = True; c.roi_beta_mode = 'quantile'; c.roi_target_accept = 0.10
    c.inference_context_k = 1
    return c
h83._build_mf_dro_config = _build

if __name__ == "__main__":
    bench, seed = sys.argv[1], int(sys.argv[2])
    _EXPERT["x_star"] = torch.tensor(XSTAR[bench], dtype=torch.float64)
    _EXPERT["rng"] = torch.Generator().manual_seed(seed * 7919 + 145)
    MF.simulate_mf_trajectory = _expert_sim
    tag = f"{bench}__H201B-ORACLE-K1__seed{seed}"
    r = h83.run(bench, "MF-DRO", seed, os.path.join(RES, "ckpt", tag + ".json"))
    _t0 = torch.stack(_EXPERT["tau0"]) if _EXPERT["tau0"] else None
    r["_h201"] = dict(arm="B", inference_context_k=1, ceiling=True, not_a_method=True,
                      n_paths=_EXPERT["n"], tau7_max_dev_from_xstar=_EXPERT["tau7_max_dev"],
                      tau0_sd=float(_t0.std(dim=0).mean()) if _t0 is not None else None)
    h83._atomic(os.path.join(RES, tag + ".json"), r)
    print(f"[done] {tag} regret={r['final_regret']:.4f} lf_frac={r.get('lf_fraction')} "
          f"tau7_dev={_EXPERT['tau7_max_dev']:.2e} wall={r['_wall_s']/60:.1f}m", flush=True)
