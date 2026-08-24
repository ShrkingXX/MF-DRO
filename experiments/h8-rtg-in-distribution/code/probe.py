"""
H8: does RTG move the decision WITHIN its realised band [0.5, 1.0]?
See ../protocol.md. Single process, thread-capped.
"""
import os, sys
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

# The realised support, measured across all 10 post-fix seeds: rtg_target
# spans exactly [0.500, 1.000] in every run.
IN_BAND = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
OOD_MULT = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]      # the original (flawed) design
N_POOLS = 12
SEED = 44
FD = torch.float32


def build():
    hf = get_benchmark("Hartmann_6D_HF"); lf = get_benchmark("Hartmann_6D_LF")
    bounds = torch.tensor([hf["domain_min"], hf["domain_max"]], dtype=torch.float64)
    torch.manual_seed(SEED); np.random.seed(SEED)
    cfg = _build_mf_dro_config(
        "h8_probe", "Hartmann_6D", "v", SEED,
        bo_iterations=3, num_epochs=10, minimum_hf_fraction=0.25,
        real_hf_warmup=2, cost_budget=1e9, initial_hf=36, initial_lf=60,
        dkl_threshold=9999, bes_delta=0.0, rollout_length=8)
    cfg.seed = SEED
    mf = DirectMFRegretOptimization(cfg, hf["make_objective"](), lf["make_objective"](), bounds)
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
    ts0 = torch.tensor([[0]], dtype=torch.long)
    pos = mf.dt.position_embedding(ts0).repeat_interleave(4, dim=1)
    seq = torch.stack([rtg_e, btg_e, s_e, a_e], dim=2).reshape(1, 4, H) + pos
    cm = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    h = mf.dt.transformer(seq, mask=cm)[0, 2::4, :][0]
    w = mf.dt.coef_head(h); b2 = mf.dt.bias_head(h)
    return ((cf.to(FD) * w.unsqueeze(0)).sum(-1) + b2).detach()


def run(mf, batch, bounds, targets, label):
    st = batch[0]["states"][0].double()
    base_btg = float(mf.btg_target_base or 22.0)
    moved, dcounts, cors = 0, [], []
    mf.dt.eval()
    with torch.no_grad():
        for p in range(N_POOLS):
            Xc = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(
                200, mf.d, dtype=torch.float64,
                generator=torch.Generator().manual_seed(500 + p))
            ysa = _y_star_for_model(mf.ko_ensemble[0], mf.y_star_pool, seed=SEED)
            cf = build_candidate_features(mf.ko_ensemble[0], Xc, bounds, mf.c_H, mf.c_L,
                                           torch.zeros(mf.d, dtype=torch.float64), y_star_arr=ysa)
            vecs = [score_vec(mf, st, t, base_btg, cf) for t in targets]
            am = [int(v.argmax()) for v in vecs]
            dcounts.append(len(set(am)))
            if len(set(am)) > 1:
                moved += 1
            cs = [np.corrcoef(vecs[i].numpy(), vecs[j].numpy())[0, 1]
                  for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
            cors.append(np.mean(cs))
    frac = moved / N_POOLS
    print(f"[{label}] targets={[round(t,2) for t in targets]}")
    print(f"  argmax moved      : {moved}/{N_POOLS} = {frac:.1%}")
    print(f"  distinct argmaxes : mean {np.mean(dcounts):.2f}, max {max(dcounts)}")
    print(f"  mean pairwise corr: {np.mean(cors):.6f}\n")
    return frac, np.mean(cors)


def main():
    mf, batch, bounds = build()
    base = float(mf._last_rtg_target) if mf._last_rtg_target else 0.62
    print(f"model's own rtg_target = {base:.4f}  (realised support across 10 seeds: [0.500, 1.000])\n")
    f_in, c_in = run(mf, batch, bounds, IN_BAND, "IN-BAND [0.5,1.0]")
    f_ood, c_ood = run(mf, batch, bounds, [base * m for m in OOD_MULT], "OOD 0.1x-10x (original design)")
    print("=" * 68)
    print(f"LOCKED PREDICTION 1: IN-BAND argmax movement < 20%  ->  "
          f"{'PASS' if f_in < 0.20 else 'FAIL'} ({f_in:.1%})")
    print(f"LOCKED PREDICTION 2: IN-BAND corr > OOD corr        ->  "
          f"{'PASS' if c_in > c_ood else 'FAIL'} ({c_in:.6f} vs {c_ood:.6f})")
    print()
    if f_in < 0.20:
        print("  VERDICT: the 'RTG does not drive decisions' finding SURVIVES once")
        print("  the OOD confound is removed. Restate as a claim about the realised")
        print("  support [0.5,1.0]. H4/H5 conclusions stand.")
    else:
        print("  VERDICT: the RTG findings were an OOD ARTEFACT. H4's refutation is")
        print("  unsafe and the schema (alpha_rtg, normalisation) becomes the prime")
        print("  suspect rather than the network.")
    print("=" * 68)


if __name__ == "__main__":
    main()
