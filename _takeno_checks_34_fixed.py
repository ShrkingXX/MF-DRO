import math
import torch

from benchmarks import get_benchmark
from src.models.ko_gp import KennedyOHaganGP
from src.policy.mf_dro import compute_joint_mf_mes

f_hf = get_benchmark('Currin_2D_HF')['make_objective']()
f_lf = get_benchmark('Currin_2D_LF')['make_objective']()
bounds = torch.tensor([[0.0] * 2, [1.0] * 2])
torch.manual_seed(42)
X_lf = torch.rand(20, 2)
X_hf = torch.rand(10, 2)
Y_lf = f_lf(X_lf)
Y_hf = f_hf(X_hf)
roi_candidates = torch.rand(50, 2, dtype=torch.float64)


def fit_with_pinned_rho(rho_target):
    """
    Build a KO model whose gp_delta is fit on residuals computed against a
    FIXED rho (not the value fit() would converge to on its own), so
    hf_posterior's mu_H = rho*mu_L + mu_delta is semantically consistent
    with that target rho -- fixes the flaw in the original Check 3/4 setup
    (which overwrote log_rho AFTER fit() without touching gp_delta, leaving
    gp_delta's learned surface tied to whatever rho fit() actually
    converged to).
    """
    ko = KennedyOHaganGP(d=2)
    ko.gp_lf = ko._build_gp(X_lf.to(torch.float64), Y_lf.to(torch.float64))
    with torch.no_grad():
        mu_lf_at_hf = ko.gp_lf.posterior(X_hf.to(torch.float64)).mean.reshape(-1)
    Y_delta = Y_hf.to(torch.float64) - rho_target * mu_lf_at_hf
    ko.gp_delta = ko._build_gp(X_hf.to(torch.float64), Y_delta)
    with torch.no_grad():
        ko.log_rho.data = torch.tensor(math.log(rho_target / (1.0 - rho_target)))
    ko.bounds = bounds.to(torch.float64)
    ko.train_x_lf = X_lf.to(torch.float64)
    ko.train_y_lf = Y_lf.to(torch.float64)
    ko.train_x_hf = X_hf.to(torch.float64)
    ko.train_y_hf = Y_hf.to(torch.float64)
    ko.train_y_delta = Y_delta
    return ko


print("=== Check 3 (corrected setup): HF MES > LF MES when rho is low (rho=0.1) ===")
ko_lowrho = fit_with_pinned_rho(0.1)
print(f"rho: {ko_lowrho.rho.item():.4f}")
x, ell, scores = compute_joint_mf_mes(ko_lowrho, roi_candidates, c_H=3.0, c_L=1.0)
n_hf_wins = (scores[:, 1] > scores[:, 0]).float().mean().item()
print(f"fraction where HF/c_H > LF/c_L score: {n_hf_wins:.3f}  (expect > 0.5)")
print("PASS" if n_hf_wins > 0.5 else "FAIL")
print()

print("=== Check 4 (corrected setup): LF preferred when rho is high (rho=0.9) and c_H=8 ===")
ko_highrho = fit_with_pinned_rho(0.9)
print(f"rho: {ko_highrho.rho.item():.4f}")
x, ell, scores = compute_joint_mf_mes(ko_highrho, roi_candidates, c_H=8.0, c_L=1.0)
n_lf_wins = (scores[:, 0] > scores[:, 1]).float().mean().item()
print(f"fraction where LF/c_L > HF/c_H score: {n_lf_wins:.3f}  (expect > 0.5)")
print("PASS" if n_lf_wins > 0.5 else "FAIL")
