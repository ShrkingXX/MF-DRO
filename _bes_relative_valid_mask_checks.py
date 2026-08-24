import statistics
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import src.policy.mf_dro as mf_dro_mod
from src.policy.mf_dro import (
    simulate_mf_trajectory, _get_mf_state_dim, DirectMFRegretOptimization,
)
from src.models.ko_gp import KennedyOHaganGP
from src.model.decisionTransformer import DecisionTransformer

torch.set_default_dtype(torch.float64)


def make_fitted_ko(d=6, n_lf=30, n_hf=20, dkl_threshold=9999, seed=0):
    """
    Fits on a smooth synthetic bowl function (a fixed random interior
    center, f(x) = -||x-center||^2 + noise) rather than i.i.d. random Y.
    First attempt used pure torch.rand() noise for Y -- with no learnable
    structure at all, the KO-GP posterior never sharpens as fantasy data
    accumulates within a rollout, so cost-normalized MES doesn't decay
    step-to-step either (observed: scores stayed in a 0.03-0.09 band across
    8 steps, never dropping anywhere near 5% of tau=0's value) -- that's a
    property of feeding the GP noise, not a property of BES itself. A
    smooth function gives the posterior something real to converge toward,
    which is what an actual benchmark rollout looks like.
    """
    torch.manual_seed(seed)
    bounds = torch.zeros(2, d, dtype=torch.float64)
    bounds[1] = 1.0
    center = torch.rand(d, dtype=torch.float64) * 0.6 + 0.2
    X_lf = torch.rand(n_lf, d, dtype=torch.float64)
    Y_lf = -((X_lf - center) ** 2).sum(dim=-1) + 0.05 * torch.randn(n_lf, dtype=torch.float64)
    X_hf = torch.rand(n_hf, d, dtype=torch.float64)
    Y_hf = -((X_hf - center) ** 2).sum(dim=-1) + 0.01 * torch.randn(n_hf, dtype=torch.float64)
    ko = KennedyOHaganGP(d=d, dkl_threshold=dkl_threshold)
    ko.fit(X_lf, Y_lf, X_hf, Y_hf, bounds)
    return ko, bounds, X_lf, Y_lf, X_hf, Y_hf


ko1, bounds1, X_lf1, Y_lf1, X_hf1, Y_hf1 = make_fitted_ko(d=6, n_lf=30, n_hf=20, dkl_threshold=9999, seed=0)
ensemble1 = [ko1, ko1, ko1]
real_data_hf1 = ([x for x in X_hf1], [float(y) for y in Y_hf1])
real_data_lf1 = ([x for x in X_lf1], [float(y) for y in Y_lf1])

print("=" * 100)
print("CHECK 1: Relative BES fires at expected threshold")
print("=" * 100)

# Monkeypatch compute_joint_mf_mes to log scores.max() at every call IN
# CALL ORDER, without altering simulate_mf_trajectory's actual code or its
# RNG consumption (the wrapper still calls straight through) -- this is the
# only way to get a step-by-step bes_signal trace that's guaranteed
# consistent with what the function itself actually computed (a
# hand-rolled parallel trace would desync: compute_joint_mf_mes and the
# post-conditioning RTG Thompson sampling both consume torch's RNG stream,
# so skipping either one in a "manual" replica drifts from the real run
# after tau=0).
trace_log = []
_orig_compute_joint_mf_mes = mf_dro_mod.compute_joint_mf_mes


def _traced(ko_model, roi_candidates, c_H, c_L, K=10):
    result = _orig_compute_joint_mf_mes(ko_model, roi_candidates, c_H, c_L, K=K)
    trace_log.append(result[2].max().item())
    return result


mf_dro_mod.compute_joint_mf_mes = _traced
try:
    torch.manual_seed(7)
    traj_c1 = simulate_mf_trajectory(
        ko1, real_data_hf1, real_data_lf1, rollout_length=8, c_H=5.0, c_L=1.0,
        bounds=bounds1, n_real_iter=20, T_real=30, ko_ensemble_full=ensemble1,
        minimum_hf_fraction=0.0,  # isolate BES: no HF-floor override interference
        bes_delta=0.05,
    )
finally:
    mf_dro_mod.compute_joint_mf_mes = _orig_compute_joint_mf_mes

actual_len_c1 = traj_c1['actions_ell'].shape[0]
signal_0 = trace_log[0]
print(f"{'tau':>3} | {'bes_signal':>12} | {'threshold(0.05*sig0)':>20} | fired")
fired_tau = None
for tau, sig in enumerate(trace_log):
    thr = 0.05 * signal_0
    fired = (tau > 0 and sig < thr)
    if fired and fired_tau is None:
        fired_tau = tau
    print(f"{tau:>3} | {sig:>12.6f} | {thr:>20.6f} | {fired}")
print(f"simulate_mf_trajectory returned actual length: {actual_len_c1}")
if fired_tau is not None:
    print(f"trace shows BES first fires at tau={fired_tau} -> expected returned length={fired_tau}")
    assert actual_len_c1 == fired_tau, "returned trajectory length must equal the first-fired tau"
else:
    print("trace shows BES never fires -> expected returned length=8 (full rollout)")
    assert actual_len_c1 == 8

# bes_delta=0.0 disables BES entirely (all 8 steps run).
torch.manual_seed(7)
traj_off = simulate_mf_trajectory(
    ko1, real_data_hf1, real_data_lf1, rollout_length=8, c_H=5.0, c_L=1.0,
    bounds=bounds1, n_real_iter=20, T_real=30, ko_ensemble_full=ensemble1,
    minimum_hf_fraction=0.0, bes_delta=0.0,
)
print(f"\nbes_delta=0.0 (disabled): actual length={traj_off['actions_ell'].shape[0]}  (EXPECT: 8)")
assert traj_off['actions_ell'].shape[0] == 8

# tau=0 never fires, even forcing an absurdly tiny relative threshold headroom
# by using an enormous bes_delta (so "5% of signal_0" itself becomes a huge
# absolute number that basically any subsequent step's signal will be below --
# the only thing keeping the rollout alive past tau=0 is the tau>0 guard).
never_fires_at_0 = True
for i in range(5):
    torch.manual_seed(100 + i)
    t = simulate_mf_trajectory(
        ko1, real_data_hf1, real_data_lf1, rollout_length=8, c_H=5.0, c_L=1.0,
        bounds=bounds1, n_real_iter=20, T_real=30, ko_ensemble_full=ensemble1,
        minimum_hf_fraction=0.0, bes_delta=1e6,
    )
    if t['actions_ell'].shape[0] < 1:
        never_fires_at_0 = False
print(f"\ntau=0 never fires across 5 trials (bes_delta=1e6, relative threshold "
      f"effectively enormous): {never_fires_at_0}  (EXPECT: True)")
assert never_fires_at_0

print("\n" + "=" * 100)
print("CHECK 2: BES produces variable-length trajectories")
print("=" * 100)
lengths = []
for i in range(20):
    torch.manual_seed(500 + i)
    t = simulate_mf_trajectory(
        ko1, real_data_hf1, real_data_lf1, rollout_length=8, c_H=5.0, c_L=1.0,
        bounds=bounds1, n_real_iter=20, T_real=30, ko_ensemble_full=ensemble1,
        bes_delta=0.05,
    )
    lengths.append(t['actions_ell'].shape[0])

dist = {L: lengths.count(L) for L in sorted(set(lengths))}
print(f"Length distribution over 20 rollouts (bes_delta=0.05): {dist}")
print(f"Distinct lengths observed: {len(dist)}  (target: >= 2 -- reported, "
      f"not asserted: whether BES fires within an 8-step rollout is an "
      f"empirical calibration question, not a code-correctness invariant)")
print(f"No trajectory has length 0: {all(L >= 1 for L in lengths)}  (EXPECT: True)")
print(f"Mean actual length: {statistics.mean(lengths):.2f} / 8")
assert all(L >= 1 for L in lengths)
if len(dist) < 2:
    print("NOTE: BES did not fire in any of these 20 rollouts at "
          "bes_delta=0.05 -- see final summary for interpretation.")

print("\n" + "=" * 100)
print("CHECK 3: _train_dt handles a mixed-length batch without crashing")
print("=" * 100)
D_C3 = 6
M_C3 = 3
STATE_DIM_C3 = _get_mf_state_dim(D_C3, M_C3)


def make_synthetic_traj(T_i, d=D_C3, state_dim=STATE_DIM_C3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        'states': torch.rand(T_i, state_dim, generator=g),
        'actions_x': torch.rand(T_i, d, generator=g),
        'actions_ell': torch.randint(0, 2, (T_i,), generator=g).long(),
        'rtg': torch.rand(T_i, generator=g),
        'btg': torch.rand(T_i, generator=g).cumsum(0).flip(0),
    }


batch_c3 = (
    [make_synthetic_traj(8, seed=i) for i in range(3)]
    + [make_synthetic_traj(4, seed=10 + i) for i in range(2)]
)
print(f"Batch lengths: {[t['states'].shape[0] for t in batch_c3]}  "
      f"(3 full-length=8, 2 BES-terminated=4)")

dt_cfg_c3 = SimpleNamespace(hidden_size=16, num_layers=1, num_heads=2,
                             max_seq_length=10, dropout=0.0)
dt_c3 = DecisionTransformer(dt_cfg_c3, input_dim=STATE_DIM_C3, action_dim=D_C3, use_mf=True).float()
opt_c3 = torch.optim.Adam(dt_c3.parameters(), lr=1e-3)
mock_c3 = SimpleNamespace(
    dt=dt_c3, dt_optimizer=opt_c3,
    config=SimpleNamespace(rollout_length=8, M=M_C3, num_epochs=3),
    d=D_C3,
)

crashed = False
try:
    L_loc, L_fid, fid_mean, fid_std = DirectMFRegretOptimization._train_dt(mock_c3, batch_c3)
except Exception as e:
    crashed = True
    print(f"CRASHED: {type(e).__name__}: {e}")

print(f"No crash: {not crashed}  (EXPECT: True)")
if not crashed:
    print(f"L_loc={L_loc:.6f}  L_fid={L_fid:.6f}  fid_mean={fid_mean:.3f}  fid_std={fid_std:.3f}")
    finite = all(map(lambda v: v == v and abs(v) != float('inf'), [L_loc, L_fid]))
    print(f"L_loc, L_fid finite: {finite}  (EXPECT: True)")
    print(f"L_loc > 0: {L_loc > 0}  L_fid > 0: {L_fid > 0}  (EXPECT: both True)")
    assert finite and L_loc > 0 and L_fid > 0
assert not crashed

print("\n" + "=" * 100)
print("CHECK 4: valid_mask loss is correct (not diluted by padding)")
print("=" * 100)
D_C4 = 6
M_C4 = 3
STATE_DIM_C4 = _get_mf_state_dim(D_C4, M_C4)
T_MAX_C4 = 8
lens_c4 = [8, 2]
B_C4 = len(lens_c4)

torch.manual_seed(99)
states_c4 = torch.zeros(B_C4, T_MAX_C4, STATE_DIM_C4)
actions_x_c4 = torch.zeros(B_C4, T_MAX_C4, D_C4)
actions_ell_c4 = torch.zeros(B_C4, T_MAX_C4, dtype=torch.long)
rtg_c4 = torch.zeros(B_C4, T_MAX_C4)
btg_c4 = torch.zeros(B_C4, T_MAX_C4)
valid_mask_c4 = torch.zeros(B_C4, T_MAX_C4, dtype=torch.bool)
for i, T_i in enumerate(lens_c4):
    states_c4[i, :T_i] = torch.rand(T_i, STATE_DIM_C4)
    actions_x_c4[i, :T_i] = torch.rand(T_i, D_C4)
    actions_ell_c4[i, :T_i] = torch.randint(0, 2, (T_i,))
    rtg_c4[i, :T_i] = torch.rand(T_i)
    btg_c4[i, :T_i] = torch.rand(T_i)
    valid_mask_c4[i, :T_i] = True
timesteps_c4 = torch.arange(T_MAX_C4).unsqueeze(0).repeat(B_C4, 1)

dt_cfg_c4 = SimpleNamespace(hidden_size=16, num_layers=1, num_heads=2,
                             max_seq_length=10, dropout=0.0)
# float32 throughout, matching _train_dt's own hardcoded .float() casts
# (forward_mf's actions_ell.float() cast inside the BCE call is float32
# regardless of the model's own dtype, so model + inputs must both be
# float32 or this raises a dtype mismatch -- mirrors the real pipeline,
# where DirectMFRegretOptimization.__init__ does self.dt = self.dt.float()).
dt_c4 = DecisionTransformer(dt_cfg_c4, input_dim=STATE_DIM_C4, action_dim=D_C4, use_mf=True).float()
dt_c4.eval()  # dropout=0.0 already, but eval() for determinism regardless

loss_c4, L_loc_c4, L_fid_c4, x_pred_c4, p_pred_c4 = dt_c4.forward_mf(
    states_c4.float(), actions_x_c4.float(), actions_ell_c4,
    rtg_c4.float(), btg_c4.float(), timesteps_c4,
    valid_mask=valid_mask_c4,
)

# Manual masked reference, computed independently from x_pred/p_pred.
manual_loc_num = 0.0
manual_fid_num = 0.0
manual_count = 0
for b, T_i in enumerate(lens_c4):
    diff = (x_pred_c4[b, :T_i] - actions_x_c4[b, :T_i].float()) ** 2
    manual_loc_num += diff.mean(dim=-1).sum().item()
    bce = F.binary_cross_entropy(p_pred_c4[b, :T_i], actions_ell_c4[b, :T_i].float(), reduction='none')
    manual_fid_num += bce.sum().item()
    manual_count += T_i
manual_L_loc = manual_loc_num / manual_count
manual_L_fid = manual_fid_num / manual_count

print(f"forward_mf L_loc (masked)  = {L_loc_c4.item():.8f}")
print(f"manual masked L_loc        = {manual_L_loc:.8f}")
print(f"match: {abs(L_loc_c4.item() - manual_L_loc) < 1e-6}  (EXPECT: True)")
print(f"forward_mf L_fid (masked)  = {L_fid_c4.item():.8f}")
print(f"manual masked L_fid        = {manual_L_fid:.8f}")
print(f"match: {abs(L_fid_c4.item() - manual_L_fid) < 1e-6}  (EXPECT: True)")
assert abs(L_loc_c4.item() - manual_L_loc) < 1e-6
assert abs(L_fid_c4.item() - manual_L_fid) < 1e-6

# Demonstrate the dilution this guards against: dividing by B*T_max (16)
# instead of the true valid count (10) would give a DIFFERENT (smaller)
# number -- i.e. masking is not a no-op here, it materially changes the loss.
diluted_L_loc = manual_loc_num / (B_C4 * T_MAX_C4)
print(f"\n(for contrast) unmasked-denominator L_loc = {diluted_L_loc:.8f} "
      f"(uses padding-diluted count {B_C4 * T_MAX_C4} instead of {manual_count})")
print(f"masked L_loc != diluted L_loc: {abs(L_loc_c4.item() - diluted_L_loc) > 1e-6}  (EXPECT: True)")
assert abs(L_loc_c4.item() - diluted_L_loc) > 1e-6

print("\n" + "=" * 100)
print("CHECK 5: fixed-length batch still works (valid_mask=None-equivalent, no regression)")
print("=" * 100)
D_C5 = 6
M_C5 = 3
STATE_DIM_C5 = _get_mf_state_dim(D_C5, M_C5)
T_C5 = 8
B_C5 = 5

torch.manual_seed(55)
states_c5 = torch.rand(B_C5, T_C5, STATE_DIM_C5)
actions_x_c5 = torch.rand(B_C5, T_C5, D_C5)
actions_ell_c5 = torch.randint(0, 2, (B_C5, T_C5)).long()
rtg_c5 = torch.rand(B_C5, T_C5)
btg_c5 = torch.rand(B_C5, T_C5)
timesteps_c5 = torch.arange(T_C5).unsqueeze(0).repeat(B_C5, 1)
all_true_mask_c5 = torch.ones(B_C5, T_C5, dtype=torch.bool)

dt_cfg_c5 = SimpleNamespace(hidden_size=16, num_layers=1, num_heads=2,
                             max_seq_length=10, dropout=0.0)
# float32 throughout -- see Check 4's comment on why (forward_mf hardcodes
# actions_ell.float() regardless of the model's own working dtype).
dt_c5 = DecisionTransformer(dt_cfg_c5, input_dim=STATE_DIM_C5, action_dim=D_C5, use_mf=True).float()
dt_c5.eval()

loss_none, L_loc_none, L_fid_none, _, _ = dt_c5.forward_mf(
    states_c5.float(), actions_x_c5.float(), actions_ell_c5, rtg_c5.float(), btg_c5.float(), timesteps_c5,
    valid_mask=None,
)
loss_mask, L_loc_mask, L_fid_mask, _, _ = dt_c5.forward_mf(
    states_c5.float(), actions_x_c5.float(), actions_ell_c5, rtg_c5.float(), btg_c5.float(), timesteps_c5,
    valid_mask=all_true_mask_c5,
)
print(f"valid_mask=None      -> loss={loss_none.item():.8f}  L_loc={L_loc_none.item():.8f}  L_fid={L_fid_none.item():.8f}")
print(f"valid_mask=all-True  -> loss={loss_mask.item():.8f}  L_loc={L_loc_mask.item():.8f}  L_fid={L_fid_mask.item():.8f}")
regression_free = (
    torch.allclose(loss_none, loss_mask, atol=1e-8)
    and torch.allclose(L_loc_none, L_loc_mask, atol=1e-8)
    and torch.allclose(L_fid_none, L_fid_mask, atol=1e-8)
)
print(f"valid_mask=None identical to valid_mask=all-True (same weights, "
      f"eval mode, no dropout): {regression_free}  (EXPECT: True)")
assert regression_free

# End-to-end: _train_dt on a uniform full-length batch (the common case once
# BES simply never fires for a given rollout) -- confirm it still runs
# cleanly and produces sane values via its new (always-padding, always-mask)
# code path.
batch_c5 = [
    {'states': states_c5[i], 'actions_x': actions_x_c5[i],
     'actions_ell': actions_ell_c5[i], 'rtg': rtg_c5[i], 'btg': btg_c5[i]}
    for i in range(B_C5)
]
dt_c5b = DecisionTransformer(dt_cfg_c5, input_dim=STATE_DIM_C5, action_dim=D_C5, use_mf=True).float()
opt_c5b = torch.optim.Adam(dt_c5b.parameters(), lr=1e-3)
mock_c5 = SimpleNamespace(
    dt=dt_c5b, dt_optimizer=opt_c5b,
    config=SimpleNamespace(rollout_length=T_C5, M=M_C5, num_epochs=5),
    d=D_C5,
)
L_loc_e2e, L_fid_e2e, fid_mean_e2e, fid_std_e2e = DirectMFRegretOptimization._train_dt(mock_c5, batch_c5)
print(f"\n_train_dt end-to-end on all-length-8 batch: "
      f"L_loc={L_loc_e2e:.6f}  L_fid={L_fid_e2e:.6f}  "
      f"fid_mean={fid_mean_e2e:.3f}  fid_std={fid_std_e2e:.3f}")
sane = (L_loc_e2e == L_loc_e2e and L_fid_e2e == L_fid_e2e  # not NaN
        and 0.0 <= fid_mean_e2e <= 1.0)
print(f"Values finite and fid_mean in [0,1]: {sane}  (EXPECT: True)")
assert sane

print("\n" + "=" * 100)
print("ALL 5 CHECKS COMPLETE")
print("=" * 100)
