"""
Does the KO model's HF-posterior calibration on Ackley_10D degrade
specifically because of LF-heavy conditioning (many LF, few HF), or is it
similarly limited regardless of the LF:HF mix (i.e. just a consequence of
the intrinsically lower r=0.75 LF-HF correlation)?

Fits a KO ensemble (same construction as the real pipeline: lognormal
prior, dkl off) on three synthetic datasets of the SAME total size (160
points, matching Stage 2 v3's 2x init: hf=60+lf=100) but different mixes:
  A) real-world mix: hf=16, lf=144  (~90% LF, matching observed Stage 2 v3
     lf_fraction on Ackley)
  B) balanced: hf=80, lf=80
  C) HF-heavy: hf=144, lf=16 (mirrors the synthetic-expert ablation's
     effectively all-HF regime)
Then measures oracle calibration (corr(mu_H, f_true), coverage_95) over
300 random probe points, and reports fitted rho -- all THREE at the same
total budget, so any calibration difference is attributable to the MIX,
not to having more total data.
"""
import torch
import numpy as np
from scipy.stats import norm as scipy_norm

torch.set_default_dtype(torch.float64)

from benchmarks import get_benchmark
from src.models.ko_gp import KennedyOHaganGP

BENCHMARK = "Ackley_10D"
hf_spec = get_benchmark(BENCHMARK + "_HF")
lf_spec = get_benchmark(BENCHMARK + "_LF")
f_hf = hf_spec["make_objective"]()
f_lf = lf_spec["make_objective"]()
bounds = torch.tensor([hf_spec["domain_min"], hf_spec["domain_max"]], dtype=torch.float64)
d = hf_spec["dim"]

MIXES = {"A_real_mix(hf16/lf144)": (16, 144), "B_balanced(hf80/lf80)": (80, 80),
         "C_hf_heavy(hf144/lf16)": (144, 16)}

torch.manual_seed(0)
probe_X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(300, d)
probe_y_true = f_hf(probe_X).reshape(-1)

for label, (n_hf, n_lf) in MIXES.items():
    torch.manual_seed(1)
    X_hf = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n_hf, d)
    Y_hf = f_hf(X_hf).reshape(-1)
    torch.manual_seed(2)
    X_lf = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n_lf, d)
    Y_lf = f_lf(X_lf).reshape(-1)

    ko = KennedyOHaganGP(d=d, dkl_threshold=9999)
    ko.fit(X_lf, Y_lf, X_hf, Y_hf, bounds)

    with torch.no_grad():
        mu_H, var_H = ko.hf_posterior(probe_X)
    sigma_H = var_H.clamp_min(1e-12).sqrt()

    corr = torch.corrcoef(torch.stack([mu_H, probe_y_true]))[0, 1].item()
    z = (probe_y_true - mu_H) / sigma_H
    coverage_95 = (z.abs() <= 1.96).float().mean().item()
    rho = ko.rho.item()
    ls_lf = ko.gp_lf.covar_module.base_kernel.lengthscale.mean().item()
    ls_delta = ko.gp_delta.covar_module.base_kernel.lengthscale.mean().item()

    believed_best_idx = mu_H.argmax()
    believed_best_true_val = probe_y_true[believed_best_idx].item()
    true_best_val = probe_y_true.max().item()

    print(f"{label}: corr(mu_H,true)={corr:.4f}  coverage_95={coverage_95:.3f}  rho={rho:.4f}  "
          f"ls_lf={ls_lf:.3f}  ls_delta={ls_delta:.3f}  "
          f"believed_best_true_val={believed_best_true_val:.4f} (true_best={true_best_val:.4f})")
