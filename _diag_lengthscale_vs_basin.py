"""
Quick check: does the GP's fitted ARD lengthscale (per dimension) match
Hartmann_6D's actual basin width around the true optimum x*? If the fitted
lengthscale is much LONGER than the true basin, the GP would over-smooth
and systematically underestimate near the peak -- exactly the symptom seen
in _diag_gp_calibration.py (GP's own best guess across 500 probes stayed at
0.3-0.5 while the true peak is 3.322).

Basin width is measured empirically (not from Hartmann's analytic
coefficients, to keep this simple and not risk transcription error): for
each dimension i, sweep x_i away from x*_i (holding all other dims at
x*), and find how far you can move before f_hf drops to half of
f_hf(x*) -- a rough full-width-at-half-max style estimate, per dimension,
using the SAME oracle access _diag_gp_calibration.py used.
"""
import os
import sys

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization

torch.set_default_dtype(torch.float64)

X_STAR = torch.tensor([0.2017, 0.1500, 0.4769, 0.2753, 0.3116, 0.6573], dtype=torch.float64)
TRUE_HF_AT_XSTAR = 3.322
BENCHMARK = "Hartmann_6D"
SEED = 42
D = 6

hf_spec = get_benchmark(BENCHMARK + "_HF")
lf_spec = get_benchmark(BENCHMARK + "_LF")
f_hf = hf_spec["make_objective"]()
f_lf = lf_spec["make_objective"]()
bounds = torch.tensor([hf_spec["domain_min"], hf_spec["domain_max"]], dtype=torch.float64)

f_at_xstar = f_hf(X_STAR.unsqueeze(0)).item()
print(f"f_hf(x*) = {f_at_xstar:.4f}  (registry true_opt={TRUE_HF_AT_XSTAR})")

print("\n" + "=" * 100)
print("EMPIRICAL BASIN HALF-WIDTH PER DIMENSION (oracle sweep, f_hf only)")
print("=" * 100)
half_max = f_at_xstar / 2.0
N_SWEEP = 400
basin_halfwidths = []
for i in range(D):
    grid = torch.linspace(0.0, 1.0, N_SWEEP, dtype=torch.float64)
    X_sweep = X_STAR.unsqueeze(0).repeat(N_SWEEP, 1)
    X_sweep[:, i] = grid
    with torch.no_grad():
        f_sweep = f_hf(X_sweep).reshape(-1)
    xstar_i = X_STAR[i].item()
    # walk outward from x*_i on both sides until f drops below half_max
    idx_star = int(torch.argmin(torch.abs(grid - xstar_i)))
    left = idx_star
    while left > 0 and f_sweep[left].item() >= half_max:
        left -= 1
    right = idx_star
    while right < N_SWEEP - 1 and f_sweep[right].item() >= half_max:
        right += 1
    halfwidth = ((grid[right] - grid[left]) / 2.0).item()
    basin_halfwidths.append(halfwidth)
    print(f"  dim {i}: x*_{i}={xstar_i:.4f}  half-max-width={grid[right]-grid[left]:.4f}  "
          f"(halfwidth={halfwidth:.4f})  f range on sweep: [{f_sweep.min().item():.3f}, {f_sweep.max().item():.3f}]")

mean_basin_halfwidth = float(np.mean(basin_halfwidths))
print(f"\nMean basin half-width across dims: {mean_basin_halfwidth:.4f}")

print("\n" + "=" * 100)
print("GP LENGTHSCALE CONSTRAINT BOUNDS (for context)")
print("=" * 100)
import math
low, high = 0.05 * math.sqrt(D), 2.0 * math.sqrt(D)
print(f"KennedyOHaganGP._lengthscale_bounds() for d={D}: [{low:.4f}, {high:.4f}]")
print(f"Geometric-mean default init: {math.sqrt(low * high):.4f}")

print("\n" + "=" * 100)
print("ACTUAL FITTED LENGTHSCALE (fresh fit on D_0, ensemble[0], DKL disabled)")
print("=" * 100)
cfg = _build_mf_dro_config(
    "diag_lengthscale", BENCHMARK, "DIAG", SEED,
    bo_iterations=1, num_epochs=1, cost_budget=9999,
    dkl_threshold=9999,
)
mf = DirectMFRegretOptimization(cfg, f_hf, f_lf, bounds)
mf._sample_initial_points()
mf._update_ko_ensemble()

ko0 = mf.ko_ensemble[0]
ls_lf = ko0.gp_lf.covar_module.base_kernel.lengthscale.detach().reshape(-1)
ls_delta = ko0.gp_delta.covar_module.base_kernel.lengthscale.detach().reshape(-1)
print(f"gp_lf lengthscale per dim:    {[f'{v:.4f}' for v in ls_lf.tolist()]}")
print(f"gp_delta lengthscale per dim: {[f'{v:.4f}' for v in ls_delta.tolist()]}")
print(f"gp_lf mean lengthscale:    {ls_lf.mean().item():.4f}")
print(f"gp_delta mean lengthscale: {ls_delta.mean().item():.4f}")

print("\n" + "=" * 100)
print("COMPARISON: fitted lengthscale vs. empirical basin half-width, per dimension")
print("=" * 100)
print(f"{'dim':>4} | {'basin_halfwidth':>16} | {'gp_lf_ls':>10} | {'gp_delta_ls':>12} | {'delta_ls/basin':>15}")
for i in range(D):
    ratio = ls_delta[i].item() / max(basin_halfwidths[i], 1e-6)
    print(f"{i:>4} | {basin_halfwidths[i]:>16.4f} | {ls_lf[i].item():>10.4f} | "
          f"{ls_delta[i].item():>12.4f} | {ratio:>15.2f}x")

mean_ratio = ls_delta.mean().item() / mean_basin_halfwidth
print(f"\nMean gp_delta lengthscale / mean basin half-width = {mean_ratio:.2f}x")
if mean_ratio > 2:
    print("-> Fitted lengthscale is MUCH LONGER than the basin: the GP is over-smoothing, "
          "averaging the narrow peak away in the mean function. Consistent with the flat, "
          "low posterior mean observed in the calibration diagnostic.")
elif mean_ratio < 0.5:
    print("-> Fitted lengthscale is SHORTER than the basin: unlikely to be the over-smoothing "
          "explanation; look elsewhere.")
else:
    print("-> Fitted lengthscale is roughly the same order as the basin width -- "
          "over-smoothing from lengthscale alone doesn't obviously explain the flat posterior mean.")
