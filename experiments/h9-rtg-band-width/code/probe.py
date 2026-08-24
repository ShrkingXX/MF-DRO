"""
H9: is RTG inert, or merely starved? See ../protocol.md.
One variable: alpha_rtg (0.5 -> 0.1). Each arm swept on its OWN realised band.
"""
import os, sys
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import (DirectMFRegretOptimization,
                                build_candidate_features, _y_star_for_model)

N_POOLS, SEED, FD = 12, 44, torch.float32
N_ITERS = 12          # enough for the rtg schema's running-max to settle


def build(alpha_rtg):
    hf = get_benchmark("Hartmann_6D_HF"); lf = get_benchmark("Hartmann_6D_LF")
    bounds = torch.tensor([hf["domain_min"], hf["domain_max"]], dtype=torch.float64)
    torch.manual_seed(SEED); np.random.seed(SEED)
    cfg = _build_mf_dro_config(
        "h9_probe", "Hartmann_6D", f"a{alpha_rtg}", SEED,
        bo_iterations=N_ITERS, num_epochs=10, minimum_hf_fraction=0.25,
        real_hf_warmup=2, cost_budget=1e9, initial_hf=36, initial_lf=60,
        dkl_threshold=9999, bes_delta=0.0, rollout_length=8,
        alpha_rtg=alpha_rtg)
    cfg.seed = SEED
    mf = DirectMFRegretOptimization(cfg, hf["make_objective"](), lf["make_objective"](), bounds)
    res = mf.run()
    return mf, res, bounds


def score_vec(mf, st, rtg, btg, cf):
    H = mf.dt.hidden_size
    s = st.unsqueeze(0).unsqueeze(0).to(FD)
    r = torch.tensor([[[rtg]]], dtype=FD); b_ = torch.tensor([[[btg]]], dtype=FD)
    ax = torch.zeros(1,1,mf.d,dtype=FD); ae0 = torch.zeros(1,1,dtype=torch.long)
    rtg_e = mf.dt.reward_ln(mf.dt.reward_embedding(r))
    btg_e = mf.dt.btg_ln(mf.dt.btg_embed(b_))
    s_e = mf.dt.state_ln(mf.dt.state_embedding(s))
    a_e = mf.dt.action_ln(mf.dt.action_embed_mf(
        torch.cat([ax, ae0.to(FD).unsqueeze(-1)], dim=-1)))
    ts0 = torch.tensor([[0]], dtype=torch.long)
    pos = mf.dt.position_embedding(ts0).repeat_interleave(4, dim=1)
    seq = torch.stack([rtg_e, btg_e, s_e, a_e], dim=2).reshape(1,4,H) + pos
    cm = torch.triu(torch.ones(4,4,dtype=torch.bool), diagonal=1)
    h = mf.dt.transformer(seq, mask=cm)[0, 2::4, :][0]
    w = mf.dt.coef_head(h); b2 = mf.dt.bias_head(h)
    return ((cf.to(FD) * w.unsqueeze(0)).sum(-1) + b2).detach()


def sweep(mf, res, bounds, label):
    r = np.array(res["rtg_target"])
    lo, hi = r.min(), r.max()
    targets = list(np.linspace(lo, hi, 6))
    st = torch.tensor(res["x_t_trace"][0], dtype=torch.float64) * 0  # placeholder replaced below
    # use a real state from the last rollout batch
    batch = mf._generate_rollout_batch()
    st = batch[0]["states"][0].double()
    base_btg = float(mf.btg_target_base or 22.0)
    moved, cors = 0, []
    mf.dt.eval()
    with torch.no_grad():
        for p in range(N_POOLS):
            Xc = bounds[0] + (bounds[1]-bounds[0]) * torch.rand(
                200, mf.d, dtype=torch.float64,
                generator=torch.Generator().manual_seed(700+p))
            ysa = _y_star_for_model(mf.ko_ensemble[0], mf.y_star_pool, seed=SEED)
            cf = build_candidate_features(mf.ko_ensemble[0], Xc, bounds, mf.c_H, mf.c_L,
                                           torch.zeros(mf.d,dtype=torch.float64), y_star_arr=ysa)
            vecs = [score_vec(mf, st, float(t), base_btg, cf) for t in targets]
            am = [int(v.argmax()) for v in vecs]
            if len(set(am)) > 1:
                moved += 1
            cors.append(np.mean([np.corrcoef(vecs[i].numpy(), vecs[j].numpy())[0,1]
                                 for i in range(len(vecs)) for j in range(i+1,len(vecs))]))
    frac = moved/N_POOLS
    width = hi/lo if lo > 0 else float('inf')
    print(f"[{label}]")
    print(f"  realised rtg_target band : [{lo:.4f}, {hi:.4f}]  = {width:.2f}x wide  (CV {r.std()/r.mean():.3f})")
    print(f"  swept                    : {[round(float(t),3) for t in targets]}")
    print(f"  argmax moved             : {moved}/{N_POOLS} = {frac:.1%}")
    print(f"  mean pairwise corr       : {np.mean(cors):.6f}\n")
    return frac, width


def main():
    print("H9: does widening the RTG band rescue RTG sensitivity?\n")
    mfN, resN, b = build(0.5); fN, wN = sweep(mfN, resN, b, "NARROW alpha_rtg=0.5 (control)")
    mfW, resW, b = build(0.1); fW, wW = sweep(mfW, resW, b, "WIDE   alpha_rtg=0.1")
    print("="*70)
    ok = wW > wN * 1.5
    print(f"MANIPULATION CHECK: WIDE band materially wider  ->  "
          f"{'PASS' if ok else 'FAIL -- EXPERIMENT VOID'} ({wN:.2f}x -> {wW:.2f}x)")
    if not ok:
        print("  alpha_rtg is not the lever assumed; do NOT interpret prediction 1.")
    else:
        print(f"LOCKED PREDICTION 1: WIDE argmax movement < 20%  ->  "
              f"{'PASS' if fW < 0.20 else 'FAIL'} ({fW:.1%}, control {fN:.1%})")
        print()
        if fW < 0.20:
            print("  VERDICT: H9 SUPPORTED. Widening the band does not rescue RTG, so the")
            print("  insensitivity belongs to the NETWORK, not the schema. The confound is")
            print("  closed and 'MF-DRO re-fits rather than conditions' is defensible.")
        else:
            print("  VERDICT: H9 REFUTED -- RTG was STARVED, not ignored. The fix is a")
            print("  schema change, not an architecture change, and H4/H5 must be re-read")
            print("  as having been tested under a starved signal.")
    print("="*70)


if __name__ == "__main__":
    main()
