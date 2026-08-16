import math
import torch
import numpy as np

from benchmarks import get_benchmark
from src.models.ko_gp import KennedyOHaganGP
from src.policy.mf_dro import (_lf_mes_info_gain, compute_joint_mf_mes,
                                 simulate_mf_trajectory, _build_hf_proxy_model)
from gumbel_thompson import thompson_sample_y_star

# --- Setup: same Currin 2D KO GP pattern from Prompt 3 (20 LF + 10 HF obs) ---
f_hf = get_benchmark('Currin_2D_HF')['make_objective']()
f_lf = get_benchmark('Currin_2D_LF')['make_objective']()
bounds = torch.tensor([[0.0] * 2, [1.0] * 2])
torch.manual_seed(42)
X_lf = torch.rand(20, 2)
X_hf = torch.rand(10, 2)
Y_lf = f_lf(X_lf)
Y_hf = f_hf(X_hf)
ko = KennedyOHaganGP(d=2)
ko.fit(X_lf, Y_lf, X_hf, Y_hf, bounds)

roi_candidates = torch.rand(50, 2, dtype=torch.float64)
hf_proxy = _build_hf_proxy_model(ko)
y_star = thompson_sample_y_star(hf_proxy, roi_candidates, K=10)

print("=== Check 1: LF MES is non-negative at all candidates ===")
mes_lf_vals = [_lf_mes_info_gain(roi_candidates[i], ko, y_star, n_quad=32) for i in range(50)]
all_nonneg = all(v >= 0 for v in mes_lf_vals)
print(f"min={min(mes_lf_vals):.6f} max={max(mes_lf_vals):.6f}")
print("PASS" if all_nonneg else "FAIL")
print()

print("=== Check 2: LF MES ~0 when LF is perfectly known (sigma_L -> 0.001) ===")
class _MockKO:
    def __init__(self, mu_L, sigma_L, mu_H, sigma_H, sigma_delta, rho):
        self._mu_L, self._sigma_L = mu_L, sigma_L
        self._mu_H, self._sigma_H = mu_H, sigma_H
        self._sigma_delta = sigma_delta
        self.rho = torch.tensor(rho)

        class _GPDelta:
            def __init__(self, var):
                self._var = var
            def posterior(self_inner, x):
                class P:
                    variance = torch.tensor([[self_inner._var]], dtype=torch.float64)
                return P()
        self.gp_delta = _GPDelta(sigma_delta ** 2)

    def lf_posterior(self, x):
        return torch.tensor([self._mu_L]), torch.tensor([self._sigma_L ** 2])

    def hf_posterior(self, x):
        return torch.tensor([self._mu_H]), torch.tensor([self._sigma_H ** 2])

# Use realistic mu/sigma_H/sigma_delta/rho from the real fitted ko at some candidate,
# but override sigma_L to 0.001 to simulate "LF perfectly known" at that point.
with torch.no_grad():
    mu_L0, var_L0 = ko.lf_posterior(roi_candidates[0:1])
    mu_H0, var_H0 = ko.hf_posterior(roi_candidates[0:1])
    var_delta0 = ko.gp_delta.posterior(roi_candidates[0:1]).variance
mock = _MockKO(
    mu_L=mu_L0.item(), sigma_L=0.001,
    mu_H=mu_H0.item(), sigma_H=var_H0.sqrt().item(),
    sigma_delta=var_delta0.sqrt().item(), rho=ko.rho.item(),
)
val = _lf_mes_info_gain(roi_candidates[0], mock, y_star, n_quad=32)
print(f"LF MES with sigma_L=0.001: {val:.6f}  (expect < 0.01)")
print("PASS" if val < 0.01 else "FAIL")
print()

print("=== Check 3: HF MES > LF MES when rho is low (rho~0.1) ===")
ko_lowrho = KennedyOHaganGP(d=2)
ko_lowrho.fit(X_lf, Y_lf, X_hf, Y_hf, bounds)
with torch.no_grad():
    ko_lowrho.log_rho.data = torch.tensor(math.log(0.1 / 0.9))
print(f"rho set to: {ko_lowrho.rho.item():.4f}")
x, ell, scores = compute_joint_mf_mes(ko_lowrho, roi_candidates, c_H=3.0, c_L=1.0)
n_hf_wins = (scores[:, 1] > scores[:, 0]).float().mean().item()
print(f"fraction where HF/c_H > LF/c_L score: {n_hf_wins:.3f}  (expect > 0.5)")
print("PASS" if n_hf_wins > 0.5 else "FAIL")
print()

print("=== Check 4: LF preferred when rho is high (rho~0.9) and c_H large (=8) ===")
ko_highrho = KennedyOHaganGP(d=2)
ko_highrho.fit(X_lf, Y_lf, X_hf, Y_hf, bounds)
with torch.no_grad():
    ko_highrho.log_rho.data = torch.tensor(math.log(0.9 / 0.1))
print(f"rho set to: {ko_highrho.rho.item():.4f}")
x, ell, scores = compute_joint_mf_mes(ko_highrho, roi_candidates, c_H=8.0, c_L=1.0)
n_lf_wins = (scores[:, 0] > scores[:, 1]).float().mean().item()
print(f"fraction where LF/c_L > HF/c_H score: {n_lf_wins:.3f}  (expect > 0.5)")
print("PASS" if n_lf_wins > 0.5 else "FAIL")
print()

print("=== Check 5: Fidelity mix over 20 rollouts (Currin 2D, c_H=3, c_L=1, minimum_hf_fraction=0.0) ===")
hf_x_list = [X_hf[i] for i in range(X_hf.shape[0])]
hf_y_list = [Y_hf[i].item() for i in range(Y_hf.shape[0])]
lf_x_list = [X_lf[i] for i in range(X_lf.shape[0])]
lf_y_list = [Y_lf[i].item() for i in range(Y_lf.shape[0])]

lf_fracs = []
for _ in range(20):
    t = simulate_mf_trajectory(
        ko, (hf_x_list, hf_y_list), (lf_x_list, lf_y_list),
        rollout_length=4, c_H=3.0, c_L=1.0,
        bounds=bounds, n_real_iter=1, T_real=30,
        ko_ensemble_full=[ko] * 10, minimum_hf_fraction=0.0,
    )
    lf_fracs.append(t['lf_fraction'])
mean_lf = sum(lf_fracs) / len(lf_fracs)
print(f"lf_fraction per rollout: {lf_fracs}")
print(f"mean lf_fraction: {mean_lf:.4f}  (expect in [0.3, 0.8])")
if 0.3 <= mean_lf <= 0.8:
    print("PASS")
else:
    print("FAIL -- printing raw mes_hf/mes_lf at first 5 candidates for diagnosis")
    x, ell, scores = compute_joint_mf_mes(ko, roi_candidates, c_H=3.0, c_L=1.0)
    for i in range(5):
        print(f"  candidate {i}: mes_lf={scores[i,0].item()*1.0:.4f} mes_hf={scores[i,1].item()*3.0:.4f}")
