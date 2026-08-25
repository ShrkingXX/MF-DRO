"""H50 worker. See ../protocol.md. No mf_dro.py edit -- instrumentation wraps
mf.dt.propose_mf and returns the original's output unmodified, so the
trajectory is bit-identical to h45's for the same seed."""
import os, sys, json, time
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization, compute_joint_mf_mes

N_POOL, N_DRAWS, CLUST_T = 200, 5, 0.15


def mode_structure(S, thresh=CLUST_T):
    """S: [N,d] normalised argmax samples -> modes, weights, separation."""
    N = S.shape[0]
    if N == 1:
        return np.array([S[0]]), np.array([1.0]), 0.0
    if np.allclose(S, S[0]):
        return np.array([S[0]]), np.array([1.0]), 0.0
    lab = fcluster(linkage(pdist(S), method="average"), t=thresh, criterion="distance")
    cs, ws = [], []
    for k in np.unique(lab):
        m = lab == k
        cs.append(S[m].mean(axis=0)); ws.append(m.sum() / N)
    cs, ws = np.array(cs), np.array(ws)
    if len(cs) == 1:
        return cs, ws, 0.0
    # weight-weighted mean pairwise centroid distance
    num = den = 0.0
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            w = ws[i] * ws[j]
            num += w * np.linalg.norm(cs[i] - cs[j]); den += w
    return cs, ws, (num / den if den > 0 else 0.0)


def run(seed):
    hf = get_benchmark("Hartmann_6D_HF"); lf = get_benchmark("Hartmann_6D_LF")
    bounds = torch.tensor([hf["domain_min"], hf["domain_max"]], dtype=torch.float64)
    torch.manual_seed(seed); np.random.seed(seed)
    cfg = _build_mf_dro_config("h50", "Hartmann_6D", "mc", seed, bo_iterations=2000,
        num_epochs=10, minimum_hf_fraction=0.25, real_hf_warmup=2, cost_budget=200.0,
        initial_hf=36, initial_lf=60, dkl_threshold=9999, bes_delta=0.0, rollout_length=8)
    cfg.seed = seed; cfg.rollout_reward = "mes_entropy"
    cfg.use_candidate_scoring = False        # h45's config verbatim
    mf = DirectMFRegretOptimization(cfg, hf["make_objective"](), lf["make_objective"](), bounds)

    lo, hi = mf.bounds[0], mf.bounds[1]
    probe, orig, it = [], mf.dt.propose_mf, {"n": 0}

    def wrapped(state, rtg, btg, timestep=0, use_candidate_scoring=False,
                candidate_features=None, fidelity_sampling=True, hist=None):
        x_norm, ell = orig(state, rtg, btg, timestep=timestep,
                           use_candidate_scoring=use_candidate_scoring,
                           candidate_features=candidate_features,
                           fidelity_sampling=fidelity_sampling, hist=hist)
        try:
            t = it["n"]
            g = torch.Generator().manual_seed(90210 + 7919 * t)
            X = lo + (hi - lo) * torch.rand(N_POOL, mf.d, dtype=torch.float64, generator=g)
            S, fids = [], []
            for m in range(len(mf.ko_ensemble)):
                for _ in range(N_DRAWS):
                    xr, el, _ = compute_joint_mf_mes(mf.ko_ensemble[m], X, mf.c_H, mf.c_L)
                    S.append(((xr - lo) / (hi - lo)).clamp(0, 1).cpu().numpy())
                    fids.append(int(el))
            S = np.asarray(S, dtype=float)
            cs, ws, sep = mode_structure(S)
            xd = np.asarray(x_norm.detach().cpu().numpy(), dtype=float).reshape(-1)
            d_near = float(min(np.linalg.norm(xd - c) for c in cs))
            tmean = (ws[:, None] * cs).sum(axis=0)
            probe.append(dict(
                iter=t, n_modes=int(len(cs)),
                mode_weights=[float(w) for w in ws],
                mode_sep=float(sep),
                d_nearest_mode=d_near,
                d_teacher_mean=float(np.linalg.norm(xd - tmean)),
                between_ratio=(float(d_near / sep) if sep > 1e-9 else None),
                teacher_spread=float(pdist(S).mean()) if len(S) > 1 else 0.0,
                n_unique_argmax=int(len({tuple(np.round(s, 6)) for s in S})),
                teacher_hf_frac=float(np.mean(fids)),
                x_dt=[float(v) for v in xd], ell_dt=int(ell)))
        except Exception as e:
            probe.append(dict(iter=it["n"], error=f"{type(e).__name__}: {e}"))
        it["n"] += 1
        return x_norm, ell

    mf.dt.propose_mf = wrapped
    t0 = time.time(); r = mf.run()
    sr = [float(v) for v in r["hf_regret_curve"]]
    ir = [float(v) for v in r.get("inference_regret_curve", [])]
    return dict(seed=seed, group=("FAIL" if seed in (49, 50) else "PASS"),
                final_regret=sr[-1], n_iters=len(sr),
                n_improvements=int(sum(1 for i in range(1, len(sr)) if sr[i] < sr[i-1]-1e-12)),
                sr_curve=sr, ir_curve=ir,
                cost_curve=[float(v) for v in r["cost_curve"]],
                fidelity_trace=[int(v) for v in r["fidelity_trace"]],
                x_t_trace=[[float(v) for v in x] for x in np.asarray(r["x_t_trace"])],
                y_t_trace=[float(v) for v in r["y_t_trace"]],
                probe=probe, wall_min=(time.time()-t0)/60.0)


if __name__ == "__main__":
    s = int(sys.argv[1])
    d = os.path.join(os.path.dirname(__file__), "..", "results")
    o = run(s)
    tmp = os.path.join(d, f"mc__seed{s}.json.tmp")
    with open(tmp, "w") as f: json.dump(o, f)
    os.replace(tmp, os.path.join(d, f"mc__seed{s}.json"))
    print(f"[done] seed{s} {o['group']} regret={o['final_regret']:.4f} "
          f"impr={o['n_improvements']} iters={o['n_iters']} wall={o['wall_min']:.1f}m", flush=True)
