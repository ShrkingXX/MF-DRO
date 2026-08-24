"""
H12 GATE (see ../protocol.md) -- run FIRST, standalone.

G1: frac of trajectories with rtg[0]==0  drops from 63.0% to < 20%
G2: frac of LF steps earning nonzero reward > 50%

If either fails, H12 stops. Single process, thread-capped.
"""
import os, sys, json, time
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

import numpy as np
import torch
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization

SEED = 44


def batch_for(reward):
    hf = get_benchmark("Hartmann_6D_HF"); lf = get_benchmark("Hartmann_6D_LF")
    bounds = torch.tensor([hf["domain_min"], hf["domain_max"]], dtype=torch.float64)
    torch.manual_seed(SEED); np.random.seed(SEED)
    cfg = _build_mf_dro_config("h12_gate", "Hartmann_6D", reward, SEED,
                                bo_iterations=1, num_epochs=1,
                                minimum_hf_fraction=0.25, real_hf_warmup=2,
                                cost_budget=1e9, initial_hf=36, initial_lf=60,
                                dkl_threshold=9999, bes_delta=0.0, rollout_length=8)
    cfg.seed = SEED
    cfg.rollout_reward = reward
    mf = DirectMFRegretOptimization(cfg, hf["make_objective"](),
                                     lf["make_objective"](), bounds)
    mf._sample_initial_points(); mf._update_ko_ensemble()
    t0 = time.time()
    batch = mf._generate_rollout_batch()
    return batch, time.time() - t0


def stats_for(batch):
    dead, lf_steps, lf_nonzero, all_r, rtg0 = 0, 0, 0, [], []
    for t in batch:
        r = t["rtg"].flatten()
        if r.numel() == 0:
            continue
        rtg0.append(float(r[0]))
        if float(r[0]) <= 1e-12:
            dead += 1
        ell = t["actions_ell"].flatten()
        m = t.get("valid_mask")
        if m is not None:
            m = m.flatten().bool()
            ell = ell[: m.numel()][m]
        # per-step reward recovered from the RTG forward-sum: r_j = rtg_j - rtg_{j+1}
        rr = r[: ell.numel()]
        per = (rr[:-1] - rr[1:]).tolist() + ([float(rr[-1])] if rr.numel() else [])
        for e, rv in zip(ell.tolist(), per):
            all_r.append(rv)
            if int(e) == 0:
                lf_steps += 1
                if abs(rv) > 1e-12:
                    lf_nonzero += 1
    n = len(rtg0)
    return {"n_traj": n,
            "dead_frac": dead / max(n, 1),
            "lf_steps": lf_steps,
            "lf_nonzero_frac": lf_nonzero / max(lf_steps, 1),
            "rtg0_mean": float(np.mean(rtg0)) if rtg0 else 0.0,
            "rtg0_cv": float(np.std(rtg0) / (np.mean(rtg0) + 1e-12)) if rtg0 else 0.0}


def main():
    out = {}
    for reward in ("improvement", "kg_incumbent"):
        b, secs = batch_for(reward)
        s = stats_for(b)
        s["batch_seconds"] = round(secs, 1)
        out[reward] = s
        print(f"\n[{reward}]  ({secs:.1f}s for {s['n_traj']} trajectories)")
        print(f"  trajectories with rtg[0]==0 : {s['dead_frac']:.1%}")
        print(f"  LF steps                     : {s['lf_steps']}")
        print(f"  LF steps with nonzero reward : {s['lf_nonzero_frac']:.1%}")
        print(f"  rtg[0] mean / CV             : {s['rtg0_mean']:.4f} / {s['rtg0_cv']:.3f}")

    kg = out["kg_incumbent"]
    g1 = kg["dead_frac"] < 0.20
    g2 = kg["lf_nonzero_frac"] > 0.50
    print("\n" + "=" * 68)
    print(f"G1 dead-signal < 20%      : {kg['dead_frac']:.1%}  -> {'PASS' if g1 else 'FAIL'}")
    print(f"G2 LF nonzero > 50%       : {kg['lf_nonzero_frac']:.1%}  -> {'PASS' if g2 else 'FAIL'}")
    print(f"\nGATE: {'PASS -- proceed to locked predictions' if (g1 and g2) else 'FAIL -- H12 STOPS HERE, no comparison run'}")
    print("=" * 68)
    out["gate"] = {"G1": g1, "G2": g2, "pass": bool(g1 and g2)}
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..",
                                      "results", "gate.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
