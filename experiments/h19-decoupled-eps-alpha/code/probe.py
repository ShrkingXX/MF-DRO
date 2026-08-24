"""H18 gate + primary probe. See ../protocol.md. Single process, thread-capped."""
import os, sys, json
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)

from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import (DirectMFRegretOptimization,
                                build_candidate_features, _y_star_for_model)
IN_BAND = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]   # H8's realised support
N_POOLS, SEED, FD = 12, 44, torch.float32


def build(mode):
    hf = get_benchmark("Hartmann_6D_HF"); lf = get_benchmark("Hartmann_6D_LF")
    bounds = torch.tensor([hf["domain_min"], hf["domain_max"]], dtype=torch.float64)
    torch.manual_seed(SEED); np.random.seed(SEED)
    cfg = _build_mf_dro_config("h18", "Hartmann_6D", mode, SEED,
                                bo_iterations=3, num_epochs=10,
                                minimum_hf_fraction=0.25, real_hf_warmup=2,
                                cost_budget=1e9, initial_hf=36, initial_lf=60,
                                dkl_threshold=9999, bes_delta=0.0, rollout_length=8)
    cfg.seed = SEED
    cfg.fantasy_mode, cfg.rollout_policy = mode.split("+")
    mf = DirectMFRegretOptimization(cfg, hf["make_objective"](),
                                     lf["make_objective"](), bounds)
    mf._sample_initial_points(); mf._update_ko_ensemble()
    batch = mf._generate_rollout_batch()
    mf._train_dt(batch)
    return mf, batch, bounds


def score_vec(mf, st, rtg, btg, cf):
    H = mf.dt.hidden_size
    s = st.unsqueeze(0).unsqueeze(0).to(FD)
    r = torch.tensor([[[rtg]]], dtype=FD); b_ = torch.tensor([[[btg]]], dtype=FD)
    ax = torch.zeros(1, 1, mf.d, dtype=FD); ae0 = torch.zeros(1, 1, dtype=torch.long)
    rtg_e = mf.dt.reward_ln(mf.dt.reward_embedding(r))
    btg_e = mf.dt.btg_ln(mf.dt.btg_embed(b_))
    s_e = mf.dt.state_ln(mf.dt.state_embedding(s))
    a_e = mf.dt.action_ln(mf.dt.action_embed_mf(
        torch.cat([ax, ae0.to(FD).unsqueeze(-1)], dim=-1)))
    pos = mf.dt.position_embedding(torch.tensor([[0]], dtype=torch.long)).repeat_interleave(4, dim=1)
    seq = torch.stack([rtg_e, btg_e, s_e, a_e], dim=2).reshape(1, 4, H) + pos
    cm = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    h = mf.dt.transformer(seq, mask=cm)[0, 2::4, :][0]
    return ((cf.to(FD) * mf.dt.coef_head(h).unsqueeze(0)).sum(-1)
            + mf.dt.bias_head(h)).detach()


def main():
    res = {}
    # ---- G1 determinism -------------------------------------------------
    hf = get_benchmark("Hartmann_6D_HF"); lf = get_benchmark("Hartmann_6D_LF")
    bounds = torch.tensor([hf["domain_min"], hf["domain_max"]], dtype=torch.float64)
    torch.manual_seed(SEED); np.random.seed(SEED)
    cfg = _build_mf_dro_config("h18g", "Hartmann_6D", "g", SEED, bo_iterations=1,
                                num_epochs=1, minimum_hf_fraction=0.25,
                                real_hf_warmup=2, cost_budget=1e9, initial_hf=36,
                                initial_lf=60, dkl_threshold=9999, bes_delta=0.0,
                                rollout_length=8)
    cfg.seed = SEED
    mfg = DirectMFRegretOptimization(cfg, hf["make_objective"](), lf["make_objective"](), bounds)
    mfg._sample_initial_points(); mfg._update_ko_ensemble()
    ko = mfg.ko_ensemble[0]
    xs = bounds[0] + (bounds[1]-bounds[0]) * torch.rand(
        6, mfg.d, dtype=torch.float64, generator=torch.Generator().manual_seed(7))
    d_mean = max(abs(ko.sample_fantasy(x, f, mode='mean')
                     - ko.sample_fantasy(x, f, mode='mean'))
                 for x in xs for f in "LH")
    d_samp = max(abs(ko.sample_fantasy(x, f, mode='sample')
                     - ko.sample_fantasy(x, f, mode='sample'))
                 for x in xs for f in "LH")
    g1 = d_mean < 1e-12
    print(f"G1 determinism: mean-mode max repeat-diff {d_mean:.3e} "
          f"(sample-mode {d_samp:.3e})  -> {'PASS' if g1 else 'FAIL'}")

    # ---- build both arms; G2 diversity ---------------------------------
    for mode in ("sample+mes", "sample+thompson", "mean+mes", "mean+thompson"):
        mf, batch, bnds = build(mode)
        # INSTRUMENT FIX: the previous signature was (rtg, ell) and omitted
        # the QUERY LOCATIONS entirely, so two rollouts that visited totally
        # different points counted as identical whenever their reward was
        # dead (63% of trajectories) and their fidelity pattern matched. That
        # measures reward degeneracy, not trajectory diversity. Signature is
        # now the actual chosen coordinates plus the fidelity pattern.
        sigs = set()
        for t in batch:
            if "actions_x" in t:
                xs = t["actions_x"].reshape(-1, mf.d)
            else:
                cnd = t["candidates"].double()
                ix = t["chosen_idx"].long().reshape(-1)
                xs = cnd[torch.arange(cnd.shape[0]), ix][:, : mf.d]
            sigs.add(tuple(np.round(xs.flatten().tolist(), 6))
                     + tuple(t["actions_ell"].flatten().tolist()))
        ysa = _y_star_for_model(mf.ko_ensemble[0], mf.y_star_pool, seed=SEED)
        st = batch[0]["states"][0].double()
        base_btg = float(mf.btg_target_base or 22.0)
        moved = 0
        mf.dt.eval()
        with torch.no_grad():
            for p in range(N_POOLS):
                Xc = bnds[0] + (bnds[1]-bnds[0]) * torch.rand(
                    200, mf.d, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(500 + p))
                cf = build_candidate_features(mf.ko_ensemble[0], Xc, bnds, mf.c_H,
                                               mf.c_L, torch.zeros(mf.d, dtype=torch.float64),
                                               y_star_arr=ysa)
                am = [int(score_vec(mf, st, t, base_btg, cf).argmax()) for t in IN_BAND]
                if len(set(am)) > 1:
                    moved += 1
        res[mode] = {"distinct_traj": len(sigs), "n_traj": len(batch),
                     "moved": moved, "frac": moved / N_POOLS}
        print(f"[{mode:>15}] distinct trajectories {len(sigs)}/{len(batch)}   "
              f"argmax moved {moved}/{N_POOLS} = {moved/N_POOLS:.1%}")

    g2 = res["mean+thompson"]["distinct_traj"] > 150
    print(f"\nG2 diversity preserved (>150 distinct): "
          f"{res['mean+thompson']['distinct_traj']}  -> {'PASS' if g2 else 'FAIL'}")
    print("=" * 70)
    if not (g1 and g2):
        print("GATE FAIL -- H18 stops here, primary prediction not interpreted.")
    else:
        p1 = res['mean+thompson']['frac'] > 0.30
        print(f"PRED 1 (>30% argmax movement under deterministic dynamics): "
              f"{res['mean']['frac']:.1%} -> {'PASS' if p1 else 'FAIL'}")
        if not p1:
            print("\nPRED 2 NULL: near-determinism is NOT the binding constraint.")
            print("  The DT fails to condition even where RCSL theory says it")
            print("  should succeed -> cause is the score-head bottleneck (H5),")
            print("  not RCSL's preconditions. Stronger negative result.")
    print("=" * 70)
    json.dump({"g1": bool(g1), "g2": bool(g2), **res},
              open(os.path.join(os.path.dirname(__file__), "..", "results",
                                "h19_fixed.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
