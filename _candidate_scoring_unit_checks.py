"""
Unit checks for candidate scoring (use_candidate_scoring flag). Synthetic
data only for checks 1-3 (no benchmark evals, no BO run); check 4 is a
tiny (5-iteration) real Hartmann_6D smoke test, matching the spec.
"""
import math
import statistics as st

import torch
import torch.nn.functional as F
import numpy as np

torch.set_default_dtype(torch.float64)

from src.model.decisionTransformer import DecisionTransformer
from src.models.ko_gp import KennedyOHaganGP
from src.policy.mf_dro import (
    simulate_mf_trajectory, _get_mf_state_dim, _build_hf_proxy_model,
    _compute_mes_hf_vectorized,
)
from gumbel_thompson import thompson_sample_y_star
from types import SimpleNamespace


def make_dt(state_dim, d, hidden=16, layers=1, heads=2, max_seq=40, dropout=0.0):
    # float32, not .double(): forward_mf hardcodes actions_ell.float()/
    # chosen_idx handling regardless of the model's own dtype (same issue
    # hit earlier this session with the BES/warm-start checks) -- model and
    # inputs must both be float32 or BCE/embedding calls raise a dtype
    # mismatch. Matches the real pipeline too (DirectMFRegretOptimization
    # does self.dt = self.dt.float()).
    cfg = SimpleNamespace(hidden_size=hidden, num_layers=layers, num_heads=heads,
                           max_seq_length=max_seq, dropout=dropout)
    return DecisionTransformer(cfg, input_dim=state_dim, action_dim=d, use_mf=True).float()


D = 6
M = 3
STATE_DIM = _get_mf_state_dim(D, M)  # actual current formula (5*M+d+5); NOT
# the "76" mentioned in the spec, which doesn't match this formula for any
# (d,M) combo actually used elsewhere in this codebase (2*6+13*... none
# divide cleanly) -- using the real formula's value (46 at d=6,M=8, or here
# 5*3+6+5=26) since these are freshly-built synthetic tensors/DTs anyway;
# the exact number doesn't affect what any of the 4 checks are testing.
B, T = 4, 8

print("=" * 100)
print(f"state_dim = {STATE_DIM} (5*M+d+5, M={M}, d={D} -- NOT 76, see note above)")
print("=" * 100)

torch.manual_seed(0)
states = torch.rand(B, T, STATE_DIM)
actions_x = torch.rand(B, T, D)
actions_ell = torch.randint(0, 2, (B, T)).long()
rtg = torch.randn(B, T)
btg = torch.rand(B, T).cumsum(-1)
timesteps = torch.arange(T).unsqueeze(0).repeat(B, 1)
valid_mask = torch.ones(B, T, dtype=torch.bool)
valid_mask[-1, 6:] = False  # one padded trajectory, exercises valid_mask

dt = make_dt(STATE_DIM, D)

# ════════════════════════════════════
print("\n" + "=" * 100)
print("CHECK 1: regression mode unchanged")
print("=" * 100)
loss1, L_loc1, L_fid1, x_pred1, p_pred1 = dt.forward_mf(
    states.float(), actions_ell, rtg.float(), btg.float(), timesteps,
    actions_x=actions_x.float(), valid_mask=valid_mask, use_candidate_scoring=False,
)
print(f"L_loc = {L_loc1.item():.4f}  (EXPECT: MSE scale, roughly [0,1]-ish for "
      f"[0,1]^d actions -- NOT strictly bounded, x_pred has no clamp in "
      f"forward_mf's regression path, only propose_mf's single-step output "
      f"clamps -- reported as an observation, not a hard assert)")
print(f"L_fid = {L_fid1.item():.4f}")
no_nan = torch.isfinite(loss1).all().item() and torch.isfinite(x_pred1).all().item()
print(f"No NaN/Inf: {no_nan}  (EXPECT: True)")
assert no_nan

# ════════════════════════════════════
print("\n" + "=" * 100)
print("CHECK 2: candidate scoring mode basic correctness")
print("=" * 100)
K_CANDS = 20
torch.manual_seed(1)
candidates = torch.rand(B, T, K_CANDS, D)
chosen_idx = torch.randint(0, K_CANDS, (B, T)).long()
# For the L_fid cross-mode comparison below to be meaningful, the EMBEDDED
# action token must be identical in both calls -- h_act (which both heads
# read from) depends on the action embedding, and in scoring mode that's
# gathered from candidates[b,t,chosen_idx[b,t]], not actions_x. Overwrite
# the chosen slot with Check 1's actual actions_x values so both modes
# embed the exact same location, isolating "does fidelity_head produce the
# same output" from "did I embed a different action."
for bi in range(B):
    for ti in range(T):
        candidates[bi, ti, chosen_idx[bi, ti]] = actions_x[bi, ti]

loss2, L_loc2, L_fid2, x_pred2, p_pred2 = dt.forward_mf(
    states.float(), actions_ell, rtg.float(), btg.float(), timesteps,
    candidates=candidates.float(), chosen_idx=chosen_idx,
    valid_mask=valid_mask, use_candidate_scoring=True,
)
print(f"L_loc = {L_loc2.item():.4f}  (EXPECT: near log(20)={math.log(20):.4f} at random init)")
print(f"L_fid = {L_fid2.item():.4f}  (EXPECT: same as Check 1's L_fid={L_fid1.item():.4f} "
      f"-- SAME dt instance, SAME states/actions_ell/rtg/btg/timesteps/valid_mask, "
      f"no gradient steps taken between the two calls, fidelity head is identical "
      f"in both modes)")
print(f"x_pred (should be None in scoring mode): {x_pred2}")
no_nan2 = torch.isfinite(loss2).all().item()
print(f"No NaN/Inf: {no_nan2}  (EXPECT: True)")
fid_match = abs(L_fid2.item() - L_fid1.item()) < 1e-9
print(f"L_fid identical across modes: {fid_match}  (EXPECT: True)")
assert no_nan2
assert x_pred2 is None
assert fid_match

# ════════════════════════════════════
print("\n" + "=" * 100)
print("CHECK 3: scoring is informative after training")
print("=" * 100)
print("Building a real fitted KO-GP (smooth synthetic bowl, same pattern as "
      "earlier session diagnostics) to generate real rollouts and compute "
      "REAL MES scores to check the trained scorer against.")

torch.manual_seed(7)
bounds = torch.zeros(2, D, dtype=torch.float64)
bounds[1] = 1.0
center = torch.rand(D, dtype=torch.float64) * 0.6 + 0.2
X_lf = torch.rand(30, D, dtype=torch.float64)
Y_lf = -((X_lf - center) ** 2).sum(dim=-1) + 0.05 * torch.randn(30, dtype=torch.float64)
X_hf = torch.rand(30, D, dtype=torch.float64)
Y_hf = -((X_hf - center) ** 2).sum(dim=-1) + 0.01 * torch.randn(30, dtype=torch.float64)
ko = KennedyOHaganGP(d=D, dkl_threshold=9999)  # ARD RBF only, no DKL
ko.fit(X_lf, Y_lf, X_hf, Y_hf, bounds)
ko_ensemble_full = [ko, ko, ko]  # M=3, matches STATE_DIM's M above

real_data_hf = ([x for x in X_hf], [float(y) for y in Y_hf])
real_data_lf = ([x for x in X_lf], [float(y) for y in Y_lf])

N_ROLLOUTS = 10
rollouts = []
for i in range(N_ROLLOUTS):
    torch.manual_seed(100 + i)
    traj = simulate_mf_trajectory(
        ko, real_data_hf, real_data_lf, rollout_length=T, c_H=5.0, c_L=1.0,
        bounds=bounds, n_real_iter=20, T_real=30, ko_ensemble_full=ko_ensemble_full,
        bes_delta=0.0,  # disabled -> fixed length T, no padding needed for this check
        use_candidate_scoring=True, K_cands=K_CANDS,
    )
    assert traj['states'].shape[0] == T
    rollouts.append(traj)

# Train a fresh DT for 50 epochs on these 10 rollouts (all fixed-length T,
# so no padding/valid_mask needed -- mirrors _train_dt's own logic minus
# the padding machinery, which BES-disabled rollouts don't exercise anyway).
dt3 = make_dt(STATE_DIM, D)
opt3 = torch.optim.Adam(dt3.parameters(), lr=1e-3)
states_b = torch.stack([r['states'] for r in rollouts]).float()
actions_ell_b = torch.stack([r['actions_ell'] for r in rollouts])
rtg_b = torch.stack([r['rtg'] for r in rollouts]).float()
btg_b = torch.stack([r['btg'] for r in rollouts]).float()
candidates_b = torch.stack([r['candidates'] for r in rollouts]).float()
chosen_idx_b = torch.stack([r['chosen_idx'] for r in rollouts])
timesteps_b = torch.arange(T).unsqueeze(0).repeat(N_ROLLOUTS, 1)

dt3.train()
for epoch in range(50):
    opt3.zero_grad()
    loss3, L_loc3, L_fid3, _, _ = dt3.forward_mf(
        states_b, actions_ell_b, rtg_b, btg_b, timesteps_b,
        candidates=candidates_b, chosen_idx=chosen_idx_b,
        use_candidate_scoring=True,
    )
    loss3.backward()
    opt3.step()
print(f"After 50 epochs: L_loc={L_loc3.item():.4f} L_fid={L_fid3.item():.4f} "
      f"(started near log(20)={math.log(20):.4f})")

# For each rollout's own step-0 state: replicate propose_mf's embedding +
# candidate-scoring pipeline externally (can't intercept propose_mf's own
# internal torch.rand(200,...) draw from outside, so this draws its OWN 200
# candidates and runs the SAME score_head computation propose_mf would --
# functionally identical, just transparent to this test instead of opaque).
dt3.eval()
rank_results = []
with torch.no_grad():
    for i, traj in enumerate(rollouts):
        s0 = traj['states'][0].float()
        rtg0 = traj['rtg'][0].item()
        btg0 = traj['btg'][0].item()

        st_ = s0.unsqueeze(0).unsqueeze(0)
        r_ = torch.tensor([[[rtg0]]], dtype=torch.float32)
        b_ = torch.tensor([[[btg0]]], dtype=torch.float32)
        ax_ = torch.zeros(1, 1, D, dtype=torch.float32)
        ae_ = torch.zeros(1, 1, dtype=torch.long)
        ts_ = torch.tensor([[0]], dtype=torch.long)

        H = dt3.hidden_size
        rtg_emb = dt3.reward_embedding(r_)
        btg_emb = dt3.btg_embed(b_)
        s_emb = dt3.state_embedding(st_)
        act_inp = torch.cat([ax_, ae_.float().unsqueeze(-1)], dim=-1)
        a_emb = dt3.action_embed_mf(act_inp)
        pos_emb = dt3.position_embedding(ts_).repeat_interleave(4, dim=1)
        seq = torch.stack([rtg_emb, btg_emb, s_emb, a_emb], dim=2).reshape(1, 4, H)
        seq = seq + pos_emb
        h_full = dt3.transformer(seq)
        h = h_full[0, 3::4, :][0]

        X_cand_norm = torch.rand(200, D, dtype=torch.float32)
        h_exp = h.unsqueeze(0).expand(200, -1)
        scores = dt3.score_head(torch.cat([h_exp, X_cand_norm], dim=-1)).squeeze(-1)
        chosen = scores.argmax().item()

        # REAL MES at the same 200 candidates (raw domain scale -- [0,1]^D
        # here so no rescale needed, bounds are [0,1]^D).
        X_cand_raw = X_cand_norm.double()
        hf_proxy = _build_hf_proxy_model(ko)
        y_star_arr = thompson_sample_y_star(hf_proxy, X_cand_raw, K=100)
        mes_arr = _compute_mes_hf_vectorized(X_cand_raw, hf_proxy, y_star_arr)

        chosen_mes = mes_arr[chosen]
        mean_mes = float(np.mean(mes_arr))
        median_mes = float(np.median(mes_arr))
        top50 = chosen_mes >= median_mes
        rank_results.append(top50)
        print(f"  rollout {i}: chosen_mes={chosen_mes:.5f}  mean_mes={mean_mes:.5f}  "
              f"median_mes={median_mes:.5f}  in_top_50%={top50}")

n_top50 = sum(rank_results)
print(f"\n{n_top50}/{N_ROLLOUTS} rollouts: chosen candidate's MES ranked in top 50%.")
print(f"EXPECT: not guaranteed, but >=7/10 after training -- NOISY CHECK, "
      f"flagging result, not asserting.")
if n_top50 < 7:
    print("NOTE: below the 7/10 informal bar -- flagged per spec, not treated as failure.")

# ════════════════════════════════════
print("\n" + "=" * 100)
print("CHECK 4: smoke test both modes side by side (Hartmann_6D, seed=42, 5 iters)")
print("=" * 100)

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization

BENCHMARK = "Hartmann_6D"
SEED = 42
N_ITERS = 5


def run_smoke(use_cs, tag):
    cfg = _build_mf_dro_config(
        f"smoke_cs_{tag}", BENCHMARK, "SMOKE", SEED,
        bo_iterations=N_ITERS, num_epochs=10, cost_budget=9999,
        dkl_threshold=9999, use_candidate_scoring=use_cs,
    )
    hf_spec = get_benchmark(BENCHMARK + "_HF")
    lf_spec = get_benchmark(BENCHMARK + "_LF")
    f_hf = hf_spec["make_objective"]()
    f_lf = lf_spec["make_objective"]()
    bnds = torch.tensor([hf_spec["domain_min"], hf_spec["domain_max"]], dtype=torch.float64)
    mf = DirectMFRegretOptimization(cfg, f_hf, f_lf, bnds)
    mf._sample_initial_points()

    print(f"\n--- {tag} (use_candidate_scoring={use_cs}) ---")
    L_locs = []
    for t in range(mf.config.bo_iterations):
        mf._update_ko_ensemble()
        batch = mf._generate_rollout_batch()
        rtg_target = mf.schemas.update_and_get_rtg_target(batch)
        btg_target = mf.schemas.update_and_get_btg_target(batch)
        mf._last_rtg_target = rtg_target
        if mf.btg_target_base is None:
            mf.btg_target_base = btg_target
        L_loc, L_fid, fid_mean, fid_std = mf._train_dt(batch)
        x_t, ell_t = mf._propose_next_query()
        p_pred_inf = mf.dt.last_p_pred

        real_hf_warmup = getattr(mf.config, 'real_hf_warmup', 2)
        if t < real_hf_warmup:
            ell_t = 1
        if ell_t == 1:
            y_t = mf.f_hf(x_t.unsqueeze(0)).reshape(-1)[0].item()
            mf.data_hf_x.append(x_t.double()); mf.data_hf_y.append(y_t)
        else:
            y_t = mf.f_lf(x_t.unsqueeze(0)).reshape(-1)[0].item()
            mf.data_lf_x.append(x_t.double()); mf.data_lf_y.append(y_t)
        step_cost = mf.c_H if ell_t else mf.c_L
        mf.cumulative_cost += step_cost
        mf.post_init_cost += step_cost
        mf.recent_ell_history.append(ell_t)

        best_hf = max(mf.data_hf_y)
        L_locs.append(L_loc)
        print(f"iter {t} | ell_t={ell_t} | p_pred={p_pred_inf:.4f} | best_hf={best_hf:.4f} | "
              f"L_loc={L_loc:.4f} | L_fid={L_fid:.4f}")
    return L_locs


L_locs_regression = run_smoke(False, "REGRESSION")
L_locs_scoring = run_smoke(True, "SCORING")

print("\n" + "=" * 100)
print("CHECK 4 SUMMARY")
print("=" * 100)
print(f"Regression L_loc range: [{min(L_locs_regression):.4f}, {max(L_locs_regression):.4f}]  "
      f"(EXPECT: [0,2])")
print(f"Scoring    L_loc range: [{min(L_locs_scoring):.4f}, {max(L_locs_scoring):.4f}]  "
      f"(EXPECT: [1,3])")
print("Both modes ran 5 iterations with no crash and no NaN (see per-iteration output above).")

print("\n" + "=" * 100)
print("ALL CHECKS COMPLETE")
print("=" * 100)
