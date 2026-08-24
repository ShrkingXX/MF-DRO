"""
H15 (see ../protocol.md): re-run the reward gate on a FAIR yardstick.

M3 (reproduction, run FIRST): Spearman(rtg[0], true f_hf(x_0))  <- original axis
M1 (primary):                 Spearman(rtg[0], max_tau f_hf(x_tau) over HF steps)
M1b (robustness):             same, over ALL steps
M2:                           simple regret at end of rollout (monotone in M1)

10 groups = 10 ensemble models x rollouts_per_model, matching the original.
Single process, thread-capped.
"""
import os, sys, json
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

import numpy as np
import torch
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
from scipy.stats import spearmanr

from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization

SEED = 44


def build(reward):
    hf = get_benchmark("Hartmann_6D_HF"); lf = get_benchmark("Hartmann_6D_LF")
    bounds = torch.tensor([hf["domain_min"], hf["domain_max"]], dtype=torch.float64)
    torch.manual_seed(SEED); np.random.seed(SEED)
    cfg = _build_mf_dro_config("h15", "Hartmann_6D", reward, SEED,
                                bo_iterations=1, num_epochs=1,
                                minimum_hf_fraction=0.25, real_hf_warmup=2,
                                cost_budget=1e9, initial_hf=36, initial_lf=60,
                                dkl_threshold=9999, bes_delta=0.0, rollout_length=8)
    cfg.seed = SEED
    cfg.rollout_reward = reward
    mf = DirectMFRegretOptimization(cfg, hf["make_objective"](),
                                     lf["make_objective"](), bounds)
    mf._sample_initial_points(); mf._update_ko_ensemble()
    batch = mf._generate_rollout_batch()
    return mf, batch, bounds, hf


def extract(mf, batch, bounds, hf):
    """Per-trajectory: rtg[0], f_hf(x_0), max f_hf over HF steps / all steps."""
    f_hf = hf["make_objective"]()
    opt = hf.get("known_optimal_value", None)
    rows = []
    for i, t in enumerate(batch):
        r = t["rtg"].flatten()
        if r.numel() == 0:
            continue
        # Under use_candidate_scoring the trajectory stores candidates +
        # chosen_idx instead of actions_x; the first d candidate columns are
        # the [0,1]^d coordinates (build_candidate_features' own layout).
        if "actions_x" in t:
            ax = t["actions_x"].reshape(-1, mf.d).double()
        else:
            cand = t["candidates"].double()                     # [T, K, F]
            idx = t["chosen_idx"].long().reshape(-1)            # [T]
            ax = cand[torch.arange(cand.shape[0]), idx][:, : mf.d]
        ell = t["actions_ell"].flatten()
        m = t.get("valid_mask")
        if m is not None:
            m = m.flatten().bool()[: ax.shape[0]]
            ax, ell = ax[m], ell[: m.numel()][m]
        if ax.shape[0] == 0:
            continue
        raw = bounds[0] + (bounds[1] - bounds[0]) * ax          # raw domain
        with torch.no_grad():
            y = f_hf(raw).reshape(-1).double()
        hf_mask = (ell.reshape(-1) == 1)
        rows.append({
            "group": i // mf.config.rollouts_per_model,
            "rtg0": float(r[0]),
            "f_x0": float(y[0]),
            "best_hf": float(y[hf_mask].max()) if bool(hf_mask.any()) else np.nan,
            "best_all": float(y.max()),
            "n_hf": int(hf_mask.sum()), "n_lf": int((~hf_mask).sum()),
            "opt": opt,
        })
    return rows


def grouped_spearman(rows, ykey):
    """Within-group Spearman + the original gate's aggregation (mean, SE, z)."""
    cors = []
    for g in sorted({r["group"] for r in rows}):
        sub = [r for r in rows if r["group"] == g and not np.isnan(r[ykey])]
        if len(sub) < 4:
            continue
        a = [r["rtg0"] for r in sub]; b = [r[ykey] for r in sub]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        cors.append(spearmanr(a, b).correlation)
    cors = [c for c in cors if np.isfinite(c)]
    if not cors:
        return None
    mean = float(np.mean(cors)); se = float(np.std(cors, ddof=1) / np.sqrt(len(cors)))
    z = mean / se if se > 0 else float("nan")
    from scipy.stats import norm
    return {"n_groups": len(cors), "mean": mean, "se": se, "z": z,
            "p": float(2 * (1 - norm.cdf(abs(z)))),
            "n_negative": int(sum(1 for c in cors if c < 0))}


def main():
    out = {}
    for reward in ("improvement", "mes_entropy"):
        mf, batch, bounds, hf = build(reward)
        rows = extract(mf, batch, bounds, hf)
        res = {k: grouped_spearman(rows, k)
               for k in ("f_x0", "best_hf", "best_all")}
        res["n_traj"] = len(rows)
        res["mean_n_hf"] = float(np.mean([r["n_hf"] for r in rows]))
        res["mean_n_lf"] = float(np.mean([r["n_lf"] for r in rows]))
        out[reward] = res
        print(f"\n[{reward}]  {len(rows)} trajectories, "
              f"mean n_HF={res['mean_n_hf']:.2f} n_LF={res['mean_n_lf']:.2f}")
        for k, lab in [("f_x0", "M3 f_hf(x_0)   [ORIGINAL AXIS]"),
                       ("best_hf", "M1 best HF pt  [PRIMARY]"),
                       ("best_all", "M1b best any   [robustness]")]:
            s = res[k]
            if s is None:
                print(f"  {lab}: undefined"); continue
            print(f"  {lab}: mean {s['mean']:+.4f}  SE {s['se']:.4f}  "
                  f"z={s['z']:+.2f}  p={s['p']:.4f}  "
                  f"({s['n_negative']}/{s['n_groups']} negative)")

    print("\n" + "=" * 74)
    imp, mes = out["improvement"], out["mes_entropy"]
    # Prediction 2 FIRST: does the original result reproduce?
    repro = (imp["f_x0"]["mean"] > mes["f_x0"]["mean"])
    print(f"PRED 2 REPRODUCTION (improvement > mes_entropy on the ORIGINAL axis):")
    print(f"   improvement {imp['f_x0']['mean']:+.4f}  vs  "
          f"mes_entropy {mes['f_x0']['mean']:+.4f}  -> {'PASS' if repro else 'FAIL'}")
    if not repro:
        print("\n   Original gate does NOT reproduce. Per protocol, BOTH the old and")
        print("   the new conclusions are suspect. STOPPING -- no M1 interpretation.")
        print("=" * 74)
    else:
        p1 = (mes["best_hf"]["mean"] >= imp["best_hf"]["mean"]
              and mes["best_hf"]["p"] < 0.05)
        print(f"\nPRED 1 PRIMARY (mes_entropy >= improvement on M1, and p<0.05):")
        print(f"   improvement {imp['best_hf']['mean']:+.4f} (p={imp['best_hf']['p']:.4f})")
        print(f"   mes_entropy {mes['best_hf']['mean']:+.4f} (p={mes['best_hf']['p']:.4f})")
        print(f"   -> {'PASS -- the gate was the problem' if p1 else 'FAIL'}")
        if not p1:
            print("\n   PRED 3 NULL: mes_entropy is the weaker conditioning signal even")
            print("   on a fair yardstick, despite its healthier distribution.")
        print("=" * 74)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..",
                                      "results", "gate15.json"), "w"),
              indent=2, default=float)


if __name__ == "__main__":
    main()
