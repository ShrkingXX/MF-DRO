import contextlib
import io
import statistics
import time

import torch

from src.models.ko_gp import KennedyOHaganGP, DeepKernel
from src.policy.mf_dro import simulate_mf_trajectory

torch.set_default_dtype(torch.float64)


def make_fitted_ko(d=6, n_lf=30, n_hf=20, dkl_threshold=9999, seed=0):
    torch.manual_seed(seed)
    bounds = torch.zeros(2, d, dtype=torch.float64)
    bounds[1] = 1.0
    X_lf = torch.rand(n_lf, d, dtype=torch.float64)
    Y_lf = torch.rand(n_lf, dtype=torch.float64)
    X_hf = torch.rand(n_hf, d, dtype=torch.float64)
    Y_hf = torch.rand(n_hf, dtype=torch.float64)
    ko = KennedyOHaganGP(d=d, dkl_threshold=dkl_threshold)
    ko.fit(X_lf, Y_lf, X_hf, Y_hf, bounds)
    return ko, bounds, X_lf, Y_lf, X_hf, Y_hf


print("=" * 100)
print("CHECK 1: BES termination behavior")
print("=" * 100)
ko1, bounds1, X_lf1, Y_lf1, X_hf1, Y_hf1 = make_fitted_ko(d=6, n_lf=30, n_hf=20, dkl_threshold=9999, seed=0)
ensemble1 = [ko1, ko1, ko1]
real_data_hf1 = ([x for x in X_hf1], [float(y) for y in Y_hf1])
real_data_lf1 = ([x for x in X_lf1], [float(y) for y in Y_lf1])

torch.manual_seed(1)
traj_hard = simulate_mf_trajectory(
    ko1, real_data_hf1, real_data_lf1, rollout_length=8, c_H=5.0, c_L=1.0,
    bounds=bounds1, n_real_iter=20, T_real=30, ko_ensemble_full=ensemble1,
    bes_delta=1e6,  # guaranteed to exceed any real MES score -> fires at tau=1
)
len_hard = traj_hard['actions_ell'].shape[0]
print(f"bes_delta=1e6 (should always fire): actual length={len_hard} "
      f"(EXPECT: 1 <= length < 8)")
assert 1 <= len_hard < 8, "BES with a huge delta should terminate early but never at tau=0"

torch.manual_seed(1)
traj_off = simulate_mf_trajectory(
    ko1, real_data_hf1, real_data_lf1, rollout_length=8, c_H=5.0, c_L=1.0,
    bounds=bounds1, n_real_iter=20, T_real=30, ko_ensemble_full=ensemble1,
    bes_delta=0,  # disabled
)
len_off = traj_off['actions_ell'].shape[0]
print(f"bes_delta=0 (disabled): actual length={len_off} (EXPECT: 8)")
assert len_off == 8, "BES disabled (bes_delta<=0) must always run the full rollout_length"

N_TRIALS_CHECK1 = 10
lengths_default = []
for i in range(N_TRIALS_CHECK1):
    torch.manual_seed(200 + i)
    traj = simulate_mf_trajectory(
        ko1, real_data_hf1, real_data_lf1, rollout_length=8, c_H=5.0, c_L=1.0,
        bounds=bounds1, n_real_iter=20, T_real=30, ko_ensemble_full=ensemble1,
        bes_delta=1e-4,  # the actual default
    )
    lengths_default.append(traj['actions_ell'].shape[0])
mean_len_check1 = statistics.mean(lengths_default)
print(f"bes_delta=1e-4 (default), {N_TRIALS_CHECK1} rollouts: lengths={lengths_default}")
print(f"  mean actual length = {mean_len_check1:.2f} / 8")

print("\n" + "=" * 100)
print("CHECK 2: BES signal properties")
print("=" * 100)
# 2a. Never fires at tau=0, even with an absurdly high delta.
never_fires_at_0 = all(
    simulate_mf_trajectory(
        ko1, real_data_hf1, real_data_lf1, rollout_length=8, c_H=5.0, c_L=1.0,
        bounds=bounds1, n_real_iter=20, T_real=30, ko_ensemble_full=ensemble1,
        bes_delta=1e6,
    )['actions_ell'].shape[0] >= 1
    for _ in range(5)
)
print(f"Never terminates at tau=0 across 5 trials (bes_delta=1e6): {never_fires_at_0}  (EXPECT: True)")

# 2b. Negative delta also disables the check (same contract as 0).
torch.manual_seed(1)
traj_neg = simulate_mf_trajectory(
    ko1, real_data_hf1, real_data_lf1, rollout_length=8, c_H=5.0, c_L=1.0,
    bounds=bounds1, n_real_iter=20, T_real=30, ko_ensemble_full=ensemble1,
    bes_delta=-1.0,
)
print(f"bes_delta=-1.0 (disabled): actual length={traj_neg['actions_ell'].shape[0]}  (EXPECT: 8)")
assert traj_neg['actions_ell'].shape[0] == 8

# 2c. Monotonicity: mean actual length should be non-increasing as bes_delta
# increases (a stricter/higher threshold can only make BES fire MORE often
# or at the same rate, never less).
def mean_length(delta, n=5, seed_base=300):
    lens = []
    for i in range(n):
        torch.manual_seed(seed_base + i)
        t = simulate_mf_trajectory(
            ko1, real_data_hf1, real_data_lf1, rollout_length=8, c_H=5.0, c_L=1.0,
            bounds=bounds1, n_real_iter=20, T_real=30, ko_ensemble_full=ensemble1,
            bes_delta=delta,
        )
        lens.append(t['actions_ell'].shape[0])
    return statistics.mean(lens)

m_tiny = mean_length(1e-8)
m_default = mean_length(1e-4)
m_huge = mean_length(1e6)
print(f"mean length @ bes_delta=1e-8:  {m_tiny:.2f}")
print(f"mean length @ bes_delta=1e-4:  {m_default:.2f}")
print(f"mean length @ bes_delta=1e6:   {m_huge:.2f}")
monotonic = (m_tiny >= m_default >= m_huge)
print(f"Monotonically non-increasing in bes_delta: {monotonic}  (EXPECT: True)")

print("\n" + "=" * 100)
print("CHECK 3: GP warm-start timing speedup")
print("=" * 100)
ko3 = KennedyOHaganGP(d=6, dkl_threshold=9999)  # force RBF only
bounds3 = torch.zeros(2, 6, dtype=torch.float64)
bounds3[1] = 1.0
torch.manual_seed(10)
X_lf3 = torch.rand(30, 6, dtype=torch.float64)
Y_lf3 = torch.rand(30, dtype=torch.float64)
X_hf3 = torch.rand(20, 6, dtype=torch.float64)
Y_hf3 = torch.rand(20, dtype=torch.float64)

t0 = time.time()
ko3.fit(X_lf3, Y_lf3, X_hf3, Y_hf3, bounds3)  # cold start (no prior state)
t_cold = time.time() - t0
print(f"Call 1 (cold start, train_iter={ko3.train_iter}): {t_cold:.3f}s")
print(f"prev_state_dict_lf stored after call 1: {ko3.prev_state_dict_lf is not None}  (EXPECT: True)")

# Simulate the next real BO iteration: one more HF point added.
X_hf3b = torch.cat([X_hf3, torch.rand(1, 6, dtype=torch.float64)])
Y_hf3b = torch.cat([Y_hf3, torch.rand(1, dtype=torch.float64)])
t0 = time.time()
ko3.fit(X_lf3, Y_lf3, X_hf3b, Y_hf3b, bounds3)  # round 0 warm-starts
t_warm = time.time() - t0
print(f"Call 2 (warm-started, train_iter_warm={ko3.train_iter_warm} on round 0): {t_warm:.3f}s")
print(f"Warm-started call faster than cold-start call: {t_warm < t_cold}  (EXPECT: True, may be noisy)")
print(f"train_iter_warm == max(10, train_iter // 2): "
      f"{ko3.train_iter_warm == max(10, ko3.train_iter // 2)}  (EXPECT: True)")

print("\n" + "=" * 100)
print("CHECK 4: DKL weight preservation under warm-start (isolated _build_gp test)")
print("=" * 100)
# Directly exercises _build_gp's prev_state_dict path, bypassing fit()'s
# 3-round loop (rounds 1-2 always cold-rebuild gp_lf/gp_delta regardless of
# round 0's warm start, so testing via fit()'s *final* self.gp_lf would not
# isolate the warm-start mechanism at all -- see simulate_mf_trajectory/
# _build_gp docstrings for why only round 0 warm-starts).
ko4 = KennedyOHaganGP(d=6, dkl_threshold=1)  # DKL config only; fit() not used here
ko4.bounds = bounds3
X4 = torch.rand(10, 6, dtype=torch.float64)
Y4 = torch.rand(10, dtype=torch.float64)

gp_cold = ko4._build_gp(X4, Y4, use_dkl=True)  # full dkl_train_iter=100 Adam steps
w_cold = gp_cold.covar_module.feature_extractor.net[0].weight.detach().clone()
prev_sd = gp_cold.state_dict()

ko4.train_iter_warm = 0  # isolate: warm-start load with ZERO further Adam steps
gp_warm = ko4._build_gp(X4, Y4, use_dkl=True, prev_state_dict=prev_sd)
w_warm = gp_warm.covar_module.feature_extractor.net[0].weight.detach().clone()

weights_preserved = torch.allclose(w_cold, w_warm)
print(f"DKL feature_extractor weights preserved exactly under a 0-iteration "
      f"warm-start load: {weights_preserved}  (EXPECT: True)")
assert weights_preserved

# Sanity: with train_iter_warm restored to a nonzero value, warm-started
# weights should still be FINITE and generally close to (not a random
# reinit of) the source weights after a few more Adam steps.
ko4.train_iter_warm = 10
gp_warm2 = ko4._build_gp(X4, Y4, use_dkl=True, prev_state_dict=prev_sd)
w_warm2 = gp_warm2.covar_module.feature_extractor.net[0].weight.detach().clone()
print(f"After 10 warm-start Adam steps, weights still finite: {w_warm2.isfinite().all().item()}")
print(f"After 10 warm-start Adam steps, weights close to source "
      f"(atol=0.5): {torch.allclose(w_cold, w_warm2, atol=0.5)}")

print("\n" + "=" * 100)
print("CHECK 5: DKL-switch discards warm-start state correctly")
print("=" * 100)
ko5 = KennedyOHaganGP(d=6, dkl_threshold=15)
bounds5 = torch.zeros(2, 6, dtype=torch.float64)
bounds5[1] = 1.0
torch.manual_seed(20)
X_lf5 = torch.rand(30, 6, dtype=torch.float64)
Y_lf5 = torch.rand(30, dtype=torch.float64)

X_hf5a = torch.rand(10, 6, dtype=torch.float64)  # below threshold -> RBF
Y_hf5a = torch.rand(10, dtype=torch.float64)
ko5.fit(X_lf5, Y_lf5, X_hf5a, Y_hf5a, bounds5)
print(f"After call 1 (n_hf=10 < threshold=15): use_dkl={ko5.use_dkl}  (EXPECT: False)")
rbf_keys_have_fe = any('feature_extractor' in k for k in ko5.prev_state_dict_lf.keys())
print(f"prev_state_dict_lf stored, RBF-shaped (no feature_extractor keys): "
      f"{ko5.prev_state_dict_lf is not None and not rbf_keys_have_fe}  (EXPECT: True)")

X_hf5b = torch.rand(20, 6, dtype=torch.float64)  # crosses threshold -> DKL activates
Y_hf5b = torch.rand(20, dtype=torch.float64)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ko5.fit(X_lf5, Y_lf5, X_hf5b, Y_hf5b, bounds5)
call2_stdout = buf.getvalue()
print(call2_stdout.rstrip())
print(f"use_dkl after call 2: {ko5.use_dkl}  (EXPECT: True)")
print(f"gp_lf is DeepKernel: {isinstance(ko5.gp_lf.covar_module, DeepKernel)}  (EXPECT: True)")
no_fallback_logged = "warm-start load_state_dict failed" not in call2_stdout
print(f"No warm-start fallback/failure message logged during the switch "
      f"(preemptive discard worked): {no_fallback_logged}  (EXPECT: True)")
dkl_keys_have_fe = any('feature_extractor' in k for k in ko5.prev_state_dict_lf.keys())
print(f"prev_state_dict_lf re-stored post-switch now HAS feature_extractor "
      f"keys (fresh DKL state, not stale RBF): {dkl_keys_have_fe}  (EXPECT: True)")
assert ko5.use_dkl and isinstance(ko5.gp_lf.covar_module, DeepKernel)
assert no_fallback_logged
assert dkl_keys_have_fe

print("\n" + "=" * 100)
print("CHECK 6: rollout_length=8+BES vs rollout_length=4-fixed timing comparison")
print("=" * 100)
ko6, bounds6, X_lf6, Y_lf6, X_hf6, Y_hf6 = make_fitted_ko(d=6, n_lf=30, n_hf=20, dkl_threshold=9999, seed=42)
ensemble6 = [ko6, ko6, ko6]
real_data_hf6 = ([x for x in X_hf6], [float(y) for y in Y_hf6])
real_data_lf6 = ([x for x in X_lf6], [float(y) for y in Y_lf6])

N_TRIALS = 10
lengths_bes = []
t0 = time.time()
for i in range(N_TRIALS):
    torch.manual_seed(400 + i)
    traj = simulate_mf_trajectory(
        ko6, real_data_hf6, real_data_lf6, rollout_length=8, c_H=5.0, c_L=1.0,
        bounds=bounds6, n_real_iter=20, T_real=30, ko_ensemble_full=ensemble6,
        bes_delta=1e-4,
    )
    lengths_bes.append(traj['actions_ell'].shape[0])
t_bes = time.time() - t0
mean_len_check6 = statistics.mean(lengths_bes)

t0 = time.time()
for i in range(N_TRIALS):
    torch.manual_seed(400 + i)
    traj = simulate_mf_trajectory(
        ko6, real_data_hf6, real_data_lf6, rollout_length=4, c_H=5.0, c_L=1.0,
        bounds=bounds6, n_real_iter=20, T_real=30, ko_ensemble_full=ensemble6,
        bes_delta=0,
    )
    assert traj['actions_ell'].shape[0] == 4
t_fixed4 = time.time() - t0

print(f"rollout_length=8 + BES(delta=1e-4): {N_TRIALS} rollouts in {t_bes:.2f}s, "
      f"lengths={lengths_bes}, mean actual length={mean_len_check6:.2f}/8")
print(f"rollout_length=4, BES disabled:     {N_TRIALS} rollouts in {t_fixed4:.2f}s, "
      f"fixed length=4 always")

print("\n" + "=" * 100)
print("ALL 6 CHECKS COMPLETE")
print("=" * 100)
overall_mean_length = statistics.mean(lengths_default + lengths_bes)
print(f"Combined mean actual rollout length across Check 1 + Check 6 "
      f"({len(lengths_default) + len(lengths_bes)} rollouts total, "
      f"bes_delta=1e-4, rollout_length=8): {overall_mean_length:.2f}")
if overall_mean_length > 7.5:
    print("  -> BES rarely fires (mean length > 7.5): bes_delta=1e-4 is likely "
          "TOO TIGHT for this MES score scale -- BES is close to a no-op, "
          "consider raising bes_delta or rescaling it relative to typical "
          "cost-normalized MES magnitudes.")
elif overall_mean_length < 2.0:
    print("  -> BES fires almost always (mean length < 2.0): bes_delta=1e-4 "
          "is likely TOO LOOSE -- rollouts are being cut short almost "
          "immediately, close to rollout_length=1 in practice, which may "
          "starve the DT of longer-horizon training examples.")
else:
    print("  -> BES fires appropriately.")
