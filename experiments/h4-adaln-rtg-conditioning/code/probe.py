"""
H4 primary measurement (see ../protocol.md): does RTG conditioning causally
affect the emitted decision?

Locked prediction: under rtg_conditioning="adaln", sweeping the RTG target over
{0.1,0.5,1,2,5,10}x changes the proposed argmax on >30% of sweeps, vs a
measured 0% under "token".

Single process, thread-capped -- safe to run alongside the h1 grid.
"""
import os
import sys

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
from src.policy.mf_dro import (DirectMFRegretOptimization,
                                build_candidate_features, _y_star_for_model)

MULTS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
N_POOLS = 12
SEED = 44
FD = torch.float32


def build_trained(mode):
    hf = get_benchmark("Hartmann_6D_HF")
    lf = get_benchmark("Hartmann_6D_LF")
    bounds = torch.tensor([hf["domain_min"], hf["domain_max"]], dtype=torch.float64)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    cfg = _build_mf_dro_config(
        "h4_probe", "Hartmann_6D", mode, SEED,
        bo_iterations=3, num_epochs=10, minimum_hf_fraction=0.25,
        real_hf_warmup=2, cost_budget=1e9, initial_hf=36, initial_lf=60,
        dkl_threshold=9999, bes_delta=0.0, rollout_length=8,
    )
    cfg.seed = SEED
    cfg.rtg_conditioning = mode
    mf = DirectMFRegretOptimization(cfg, hf["make_objective"](), lf["make_objective"](), bounds)
    mf._sample_initial_points()
    mf._update_ko_ensemble()
    batch = mf._generate_rollout_batch()
    mf._train_dt(batch)          # train so the conditioning has a chance to matter
    return mf, batch, bounds


def sweep(mf, batch, bounds, pool_seed):
    """Return (argmaxes across MULTS, score vectors, mu_H column)."""
    st = batch[0]["states"][0].double()
    base_rtg = float(mf._last_rtg_target) if mf._last_rtg_target else 1.0
    base_btg = float(mf.btg_target_base or 22.0)

    Xc = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(
        200, mf.d, dtype=torch.float64,
        generator=torch.Generator().manual_seed(pool_seed))
    ysa = _y_star_for_model(mf.ko_ensemble[0], mf.y_star_pool, seed=SEED)
    cf = build_candidate_features(mf.ko_ensemble[0], Xc, bounds, mf.c_H, mf.c_L,
                                   torch.zeros(mf.d, dtype=torch.float64),
                                   y_star_arr=ysa)
    mf.dt.eval()
    vecs = []
    with torch.no_grad():
        for m in MULTS:
            x_t, _ = mf.dt.propose_mf(
                st.float(), base_rtg * m, base_btg, timestep=0,
                use_candidate_scoring=True,
                candidate_features=cf.float(),
                fidelity_sampling=False)
            # recover which candidate was chosen (propose_mf returns its coords)
            d = (cf[:, :mf.d].float() - x_t.unsqueeze(0)).norm(dim=-1)
            vecs.append(int(d.argmin().item()))
    return vecs, cf[:, mf.d]      # argmaxes, mu_H column


def main():
    print(f"H4 probe: RTG sweep {MULTS} over {N_POOLS} resampled candidate pools\n")
    summary = {}
    for mode in ["token", "adaln"]:
        mf, batch, bounds = build_trained(mode)
        moved, distinct_counts, muH_agree = 0, [], 0
        for p in range(N_POOLS):
            argmaxes, muH = sweep(mf, batch, bounds, 100 + p)
            nd = len(set(argmaxes))
            distinct_counts.append(nd)
            if nd > 1:
                moved += 1
            if argmaxes[MULTS.index(1.0)] == int(muH.argmax().item()):
                muH_agree += 1
        frac = moved / N_POOLS
        summary[mode] = frac
        print(f"[{mode}]")
        print(f"  sweeps where argmax MOVED : {moved}/{N_POOLS} = {frac:.1%}")
        print(f"  distinct argmaxes per sweep: mean={np.mean(distinct_counts):.2f} "
              f"max={max(distinct_counts)} (1 = fully insensitive)")
        print(f"  argmax==argmax(mu_H) at 1x : {muH_agree}/{N_POOLS} "
              f"({muH_agree/N_POOLS:.0%}) -- still a fixed acquisition?\n")

    print("=" * 66)
    print(f"LOCKED PREDICTION: adaln moves argmax on >30% of sweeps")
    print(f"  token = {summary['token']:.1%}   adaln = {summary['adaln']:.1%}")
    verdict = ("H4 SUPPORTED" if summary["adaln"] > 0.30
               else "H4 REFUTED -- insensitivity is NOT attention under-allocation")
    print(f"  RESULT: {verdict}")
    print("=" * 66)


if __name__ == "__main__":
    main()
