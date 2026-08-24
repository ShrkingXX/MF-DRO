"""
H5 primary measurement (see ../protocol.md).

Arm A: use_candidate_features=True   -> candidates carry [x_norm(d), mu_H, sigma_H, mu_L, sigma_L, dist_inc]
Arm B: use_candidate_features=False  -> candidates carry [x_norm(d)] only

Locked predictions (B vs A):
  1. h-shuffle sensitivity rises from ~8% to >30%      <- REAL TEST
  2. argmax(score)==argmax(mu_H) falls to <30%          <- manipulation check only
  3. RTG-sweep argmax movement rises above A's 25%      <- REAL TEST

Single process, thread-capped.
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
                                build_candidate_features, _y_star_for_model,
                                _gp_candidate_features)

MULTS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
N_POOLS = 12
SEED = 44
FD = torch.float32


def build(use_feats):
    hf = get_benchmark("Hartmann_6D_HF")
    lf = get_benchmark("Hartmann_6D_LF")
    bounds = torch.tensor([hf["domain_min"], hf["domain_max"]], dtype=torch.float64)
    torch.manual_seed(SEED); np.random.seed(SEED)
    cfg = _build_mf_dro_config(
        "h5_probe", "Hartmann_6D", f"feats{use_feats}", SEED,
        bo_iterations=3, num_epochs=10, minimum_hf_fraction=0.25,
        real_hf_warmup=2, cost_budget=1e9, initial_hf=36, initial_lf=60,
        dkl_threshold=9999, bes_delta=0.0, rollout_length=8,
    )
    cfg.seed = SEED
    cfg.use_candidate_features = use_feats
    cfg.rtg_conditioning = "token"     # H4 showed adaln is not the lever; vary ONE thing
    mf = DirectMFRegretOptimization(cfg, hf["make_objective"](), lf["make_objective"](), bounds)
    mf._sample_initial_points(); mf._update_ko_ensemble()
    batch = mf._generate_rollout_batch()
    mf._train_dt(batch)
    return mf, batch, bounds


def hidden_and_scores(mf, st, rtg, btg, cf):
    """Return (h_state, score_vector) for one (state, rtg, btg, candidate-set)."""
    H = mf.dt.hidden_size
    s = st.unsqueeze(0).unsqueeze(0).to(FD)
    r = torch.tensor([[[rtg]]], dtype=FD); b_ = torch.tensor([[[btg]]], dtype=FD)
    ax = torch.zeros(1, 1, mf.d, dtype=FD); ae0 = torch.zeros(1, 1, dtype=torch.long)
    rtg_e = mf.dt.reward_ln(mf.dt.reward_embedding(r))
    btg_e = mf.dt.btg_ln(mf.dt.btg_embed(b_))
    s_e = mf.dt.state_ln(mf.dt.state_embedding(s))
    a_e = mf.dt.action_ln(mf.dt.action_embed_mf(
        torch.cat([ax, ae0.to(FD).unsqueeze(-1)], dim=-1)))
    ts0 = torch.tensor([[0]], dtype=torch.long)
    pos = mf.dt.position_embedding(ts0).repeat_interleave(4, dim=1)
    seq = torch.stack([rtg_e, btg_e, s_e, a_e], dim=2).reshape(1, 4, H) + pos
    cm = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    h = mf.dt.transformer(seq, mask=cm)[0, 2::4, :][0]
    w = mf.dt.coef_head(h); b2 = mf.dt.bias_head(h)
    sc = (cf.to(FD) * w.unsqueeze(0)).sum(-1) + b2
    return h, sc.detach()


def main():
    print(f"H5 probe: {N_POOLS} resampled candidate pools, RTG sweep {MULTS}\n")
    summary = {}
    for use_feats, label in [(True, "A: WITH GP features (d+5)"),
                             (False, "B: coords only (d)")]:
        mf, batch, bounds = build(use_feats)
        st = batch[0]["states"][0].double()
        base_rtg = float(mf._last_rtg_target) if mf._last_rtg_target else 1.0
        base_btg = float(mf.btg_target_base or 22.0)

        rtg_moved, h_changed, muH_agree = 0, 0, 0
        mf.dt.eval()
        with torch.no_grad():
            for p in range(N_POOLS):
                Xc = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(
                    200, mf.d, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(300 + p))
                ysa = _y_star_for_model(mf.ko_ensemble[0], mf.y_star_pool, seed=SEED)
                if use_feats:
                    cf = build_candidate_features(
                        mf.ko_ensemble[0], Xc, bounds, mf.c_H, mf.c_L,
                        torch.zeros(mf.d, dtype=torch.float64), y_star_arr=ysa)
                else:
                    cf = ((Xc - bounds[0]) / (bounds[1] - bounds[0])).clamp(0, 1)
                # mu_H always computed from the GP for the comparison, even in
                # Arm B where it is NOT a feature.
                muH = _gp_candidate_features(mf.ko_ensemble[0], Xc, mf.c_H, mf.c_L, ysa)[0]

                # (3) RTG sweep
                argmaxes = []
                for m in MULTS:
                    _, sc = hidden_and_scores(mf, st, base_rtg * m, base_btg, cf)
                    argmaxes.append(int(sc.argmax()))
                if len(set(argmaxes)) > 1:
                    rtg_moved += 1
                base_arg = argmaxes[MULTS.index(1.0)]

                # (2) manipulation check
                if base_arg == int(muH.argmax()):
                    muH_agree += 1

                # (1) h-shuffle sensitivity: same candidates, DIFFERENT h
                #     (h from a different state in the batch)
                st_other = batch[(p + 7) % len(batch)]["states"][0].double()
                _, sc_other = hidden_and_scores(mf, st_other, base_rtg, base_btg, cf)
                if int(sc_other.argmax()) != base_arg:
                    h_changed += 1

        fr_rtg, fr_h, fr_mu = (rtg_moved / N_POOLS, h_changed / N_POOLS,
                                muH_agree / N_POOLS)
        summary[label] = (fr_h, fr_mu, fr_rtg)
        print(f"[{label}]  cand_feature_dim={cf.shape[-1]}")
        print(f"  (1) h swapped changes argmax : {h_changed}/{N_POOLS} = {fr_h:.1%}   <- REAL TEST")
        print(f"  (2) argmax == argmax(mu_H)   : {muH_agree}/{N_POOLS} = {fr_mu:.1%}   (manipulation check)")
        print(f"  (3) RTG sweep moves argmax   : {rtg_moved}/{N_POOLS} = {fr_rtg:.1%}   <- REAL TEST\n")

    A = summary["A: WITH GP features (d+5)"]; B = summary["B: coords only (d)"]
    print("=" * 70)
    print("LOCKED PREDICTIONS (B vs A)")
    print(f"  1. h-sensitivity  >30% in B : A={A[0]:.1%} -> B={B[0]:.1%}  "
          f"{'PASS' if B[0] > 0.30 else 'FAIL'}")
    print(f"  2. mu_H agreement <30% in B : A={A[1]:.1%} -> B={B[1]:.1%}  "
          f"{'(check ok)' if B[1] < 0.30 else '(check FAILED)'}")
    print(f"  3. RTG movement rises       : A={A[2]:.1%} -> B={B[2]:.1%}  "
          f"{'PASS' if B[2] > A[2] else 'FAIL'}")
    real = (B[0] > 0.30) and (B[2] > A[2])
    print(f"\n  VERDICT: {'H5 SUPPORTED' if real else 'H5 NOT SUPPORTED on the real tests'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
