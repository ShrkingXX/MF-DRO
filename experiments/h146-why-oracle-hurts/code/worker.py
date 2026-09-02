"""h146 DIVERSE-GOOD worker. Identical to h145's oracle worker in every respect
EXCEPT the destination: each trajectory interpolates toward its own high-quality
endpoint instead of the single global x*.

Endpoint = best of a fresh random pool of POOL points under the TRUE objective.
Quality stays high (it is the max of POOL draws); endpoint diversity is restored
(a different endpoint per trajectory). That is the one factor h145 confounded.
"""
import os, sys, importlib.util
import torch

H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
_s = importlib.util.spec_from_file_location(
    "h83w", os.path.join(REPO, "experiments/h83-main-comparison/code/worker.py"))
h83 = importlib.util.module_from_spec(_s); sys.modules["h83w"] = h83; _s.loader.exec_module(h83)
import src.policy.mf_dro as MF
from benchmarks import get_benchmark

RES = os.path.abspath(os.path.join(H, "..", "results"))
h83.RES = RES
POOL = 256                      # endpoint = argmax of POOL true-objective draws

_ORIG = MF.simulate_mf_trajectory
_S = {"rng": None, "f": None, "endpoints": []}


def _diverse_good_path(bounds, T, d):
    lo, hi = bounds[0], bounds[1]
    # high-quality but DIFFERENT endpoint per trajectory
    cand = lo + (hi - lo) * torch.rand(POOL, d, generator=_S["rng"], dtype=bounds.dtype)
    y = _S["f"](cand).reshape(-1)
    x_end = cand[int(torch.argmax(y).item())]
    x_start = lo + (hi - lo) * torch.rand(d, generator=_S["rng"], dtype=bounds.dtype)
    _S["endpoints"].append(x_end)
    return torch.stack([x_start + (x_end - x_start) * (t / max(T - 1, 1)) for t in range(T)])


def _sim(*a, **k):
    bounds = k.get("bounds", a[6] if len(a) > 6 else None)
    T = int(k.get("rollout_length", a[3] if len(a) > 3 else 8))
    k["forced_x"] = _diverse_good_path(bounds, T, int(bounds.shape[1]))
    return _ORIG(*a, **k)


if __name__ == "__main__":
    bench, seed = sys.argv[1], int(sys.argv[2])
    _S["rng"] = torch.Generator().manual_seed(seed * 7919 + 146)
    _S["f"] = get_benchmark(bench + "_HF")["make_objective"]()
    MF.simulate_mf_trajectory = _sim
    tag = f"{bench}__DIVERSE-GOOD__seed{seed}"
    r = h83.run(bench, "MF-DRO", seed, os.path.join(RES, "ckpt", tag + ".json"))
    E = torch.stack(_S["endpoints"]) if _S["endpoints"] else torch.zeros(0)
    r["_h146"] = dict(n_paths=int(E.shape[0]), pool=POOL,
                      endpoint_spread=float(E.std(0).mean()) if E.numel() else None)
    h83._atomic(os.path.join(RES, tag + ".json"), r)
    print(f"[done] {tag} regret={r['final_regret']:.4f} paths={E.shape[0]} "
          f"wall={r['_wall_s']/60:.1f}m", flush=True)
