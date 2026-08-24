"""
H11 (see ../protocol.md): is the RTG channel ABSENT at inference rather than inert?

Arm A  T=1, timestep=0, one RTG token            <- what every run has always done
Arm B  T=k real history, RTG target repeated     <- isolates context length
Arm C  T=k real history, DT-style RTG decrement  <- Chen et al. 2021, Alg. 1

Same trained weights in all three arms. Only the inference context differs.
Single process, thread-capped.
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

import json
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import (DirectMFRegretOptimization,
                                build_candidate_features, _y_star_for_model)

K = 8              # trained position range (rollout_length)
N_POOLS = 12
SEED = 44
FD = torch.float32
MULTS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]     # the realised in-band support (H8)


def build_with_history(n_iters=12):
    """Run a real MF-DRO run, recording the real state fed at every iteration."""
    hf = get_benchmark("Hartmann_6D_HF"); lf = get_benchmark("Hartmann_6D_LF")
    bounds = torch.tensor([hf["domain_min"], hf["domain_max"]], dtype=torch.float64)
    torch.manual_seed(SEED); np.random.seed(SEED)
    cfg = _build_mf_dro_config(
        "h11_probe", "Hartmann_6D", "hist", SEED,
        bo_iterations=n_iters, num_epochs=10, minimum_hf_fraction=0.25,
        real_hf_warmup=2, cost_budget=1e9, initial_hf=36, initial_lf=60,
        dkl_threshold=9999, bes_delta=0.0, rollout_length=K)
    cfg.seed = SEED
    mf = DirectMFRegretOptimization(cfg, hf["make_objective"](), lf["make_objective"](), bounds)

    # Record the REAL state handed to propose_mf at each real iteration.
    hist = []
    _orig = mf.dt.propose_mf
    def _rec(state, rtg_target, btg_target, **kw):
        hist.append({'state': state.detach().clone(),
                     'rtg': float(rtg_target), 'btg': float(btg_target)})
        return _orig(state, rtg_target, btg_target, **kw)
    mf.dt.propose_mf = _rec
    mf.run()
    mf.dt.propose_mf = _orig
    return mf, hist, bounds


def realized_improvements(mf):
    """Per-iteration realized HF improvement, for the DT-style RTG decrement."""
    best = -float('inf'); out = []
    for rec in mf.iteration_log:
        if rec['ell_t'] == 1:
            y = float(rec['y_t'])
            out.append(max(0.0, y - best) if np.isfinite(best) else 0.0)
            best = max(best, y)
        else:
            out.append(0.0)
    return out


def readout(mf, states, rtgs, btgs):
    """Build a T=len(states) 4-token sequence; return h at the LAST state token."""
    T = len(states)
    H = mf.dt.hidden_size
    s = torch.stack(states).unsqueeze(0).to(FD)                       # [1,T,S]
    r = torch.tensor(rtgs, dtype=FD).view(1, T, 1)
    b = torch.tensor(btgs, dtype=FD).view(1, T, 1)
    ax = torch.zeros(1, T, mf.d, dtype=FD)
    ae = torch.zeros(1, T, 1, dtype=FD)
    rtg_e = mf.dt.reward_ln(mf.dt.reward_embedding(r))
    btg_e = mf.dt.btg_ln(mf.dt.btg_embed(b))
    s_e = mf.dt.state_ln(mf.dt.state_embedding(s))
    a_e = mf.dt.action_ln(mf.dt.action_embed_mf(torch.cat([ax, ae], dim=-1)))
    ts = torch.arange(T, dtype=torch.long).unsqueeze(0)
    pos = mf.dt.position_embedding(ts).repeat_interleave(4, dim=1)
    seq = torch.stack([rtg_e, btg_e, s_e, a_e], dim=2).reshape(1, 4 * T, H) + pos
    cm = torch.triu(torch.ones(4 * T, 4 * T, dtype=torch.bool), diagonal=1)
    h = mf.dt.transformer(seq, mask=cm)[0, 2::4, :][-1]     # LAST state token
    return h


def score(mf, h, cf):
    w = mf.dt.coef_head(h); b2 = mf.dt.bias_head(h)
    return ((cf.to(FD) * w.unsqueeze(0)).sum(-1) + b2).detach()


def main():
    mf, hist, bounds = build_with_history()
    imps = realized_improvements(mf)
    print(f"\nreal history recorded: {len(hist)} iterations, using last k={K}\n")

    tail = hist[-K:]
    tail_imp = imps[-K:]
    states = [h['state'].double() for h in tail]
    btgs = [h['btg'] for h in tail]
    base_rtg = tail[-1]['rtg']

    ysa = _y_star_for_model(mf.ko_ensemble[0], mf.y_star_pool, seed=SEED)
    mf.dt.eval()
    res = {}
    spans = []
    with torch.no_grad():
        for arm in ("A", "B", "C"):
            moved = 0
            for p in range(N_POOLS):
                Xc = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(
                    200, mf.d, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(900 + p))
                cf = build_candidate_features(
                    mf.ko_ensemble[0], Xc, bounds, mf.c_H, mf.c_L,
                    torch.zeros(mf.d, dtype=torch.float64), y_star_arr=ysa)
                argmaxes = []
                for m in MULTS:
                    tgt = base_rtg * m
                    if arm == "A":
                        h = readout(mf, states[-1:], [tgt], btgs[-1:])
                    elif arm == "B":
                        h = readout(mf, states, [tgt] * len(states), btgs)
                    else:
                        # DT-style: R_{t+1} = R_t - r_t, so position j carries
                        # the return still OUTSTANDING at that point.
                        rem, seq_r = tgt, []
                        for j in range(len(states)):
                            seq_r.append(rem)
                            rem = rem - tail_imp[j]
                        h = readout(mf, states, seq_r, btgs)
                        if p == 0 and m == 1.0:
                            lo, hi = min(seq_r), max(seq_r)
                            spans.append(hi / lo if lo > 1e-12 else float('inf'))
                            print(f"  arm C within-pass RTG sequence: "
                                  f"{[round(v,4) for v in seq_r]}")
                    argmaxes.append(int(score(mf, h, cf).argmax()))
                if len(set(argmaxes)) > 1:
                    moved += 1
            res[arm] = moved / N_POOLS
            print(f"[arm {arm}] argmax moved {moved}/{N_POOLS} = {res[arm]:.1%}")

    span = spans[0] if spans else 1.0
    print("\n" + "=" * 68)
    print(f"MANIPULATION CHECK (arm C within-pass RTG span >= 3x): "
          f"{span:.2f}x -> {'PASS' if span >= 3.0 else 'FAIL -- ARM C VOID'}")
    print(f"PREDICTION 1 (arm C > 30%): {res['C']:.1%} -> "
          f"{'PASS' if res['C'] > 0.30 else 'FAIL'}")
    print(f"PREDICTION 2 (arm C > arm B): C={res['C']:.1%} B={res['B']:.1%} -> "
          f"{'PASS' if res['C'] > res['B'] else 'FAIL -- effect is context, not RTG'}")
    if max(res.values()) < 0.05:
        print("\nNULL-GUARD TRIPPED: all arms ~0%. The DT is insensitive to RTG")
        print("AND to its own real history. This CLOSES the inert-vs-starved")
        print("confound in the negative direction -- stronger than H8-H10.")
    print("=" * 68)
    json.dump({'arms': res, 'rtg_span': span, 'n_hist': len(hist)},
              open(os.path.join(os.path.dirname(__file__), "..",
                                "results", "h11.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
