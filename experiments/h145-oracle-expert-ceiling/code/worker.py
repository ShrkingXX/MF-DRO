"""h145 oracle-expert worker. Shim over h83's worker so the config, initial
design, cost budget and metric are IDENTICAL to the MF-DRO control -- the only
difference is that the rollout teacher's LOCATIONS come from an oracle path.

Mechanism: wrap the module-level `simulate_mf_trajectory` so each call receives
its own freshly-drawn expert path via the `forced_x` argument added for h145.
`_generate_rollout_batch` calls it once per trajectory, so every trajectory gets
an independent x_start, and all the batch-level machinery above it is untouched.

Fidelity is NOT forced: `forced_x` replaces the location only, and ell_tau is
still chosen by the same cost-normalised MES criterion at that point.
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

# x* per benchmark, RAW domain scale, from data/optima.json (the cached solver
# result, whose own gap is recorded alongside and used by SC3).
XSTAR = {
    "Hartmann_6D": [0.2017, 0.1500, 0.4769, 0.2753, 0.3116, 0.6573],
    "Borehole_8D": [0.15, 100.0, 95090.9777, 1110.0, 116.0, 700.0, 1120.0, 12045.0],
}

_ORIG_SIM = MF.simulate_mf_trajectory
_EXPERT = {"x_star": None, "rng": None, "paths": []}


def _expert_path(bounds, T, d):
    """x_start ~ Uniform(domain); linear interpolation to x*, arriving exactly."""
    lo, hi = bounds[0], bounds[1]
    x_start = lo + (hi - lo) * torch.rand(d, generator=_EXPERT["rng"], dtype=bounds.dtype)
    xs = _EXPERT["x_star"].to(dtype=bounds.dtype)
    return torch.stack([x_start + (xs - x_start) * (t / max(T - 1, 1)) for t in range(T)])


def _expert_sim(*args, **kw):
    bounds = kw.get("bounds")
    if bounds is None:                      # bounds is positional in the real signature
        bounds = args[6]
    T = kw.get("rollout_length", args[3] if len(args) > 3 else 8)
    d = int(bounds.shape[1])
    path = _expert_path(bounds, int(T), d)
    _EXPERT["paths"].append(path)
    kw["forced_x"] = path
    return _ORIG_SIM(*args, **kw)


if __name__ == "__main__":
    bench, seed = sys.argv[1], int(sys.argv[2])
    _EXPERT["x_star"] = torch.tensor(XSTAR[bench], dtype=torch.float64)
    _EXPERT["rng"] = torch.Generator().manual_seed(seed * 7919 + 145)
    MF.simulate_mf_trajectory = _expert_sim          # install the oracle teacher

    tag = f"{bench}__ORACLE-EXPERT__seed{seed}"
    r = h83.run(bench, "MF-DRO", seed, os.path.join(RES, "ckpt", tag + ".json"))
    r["_h145"] = dict(n_expert_paths=len(_EXPERT["paths"]), x_star=XSTAR[bench])
    h83._atomic(os.path.join(RES, tag + ".json"), r)
    print(f"[done] {tag} regret={r['final_regret']:.4f} paths={len(_EXPERT['paths'])} "
          f"wall={r['_wall_s']/60:.1f}m", flush=True)
