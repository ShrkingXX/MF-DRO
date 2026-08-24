import json
import torch
import numpy as np
from scipy.stats.qmc import LatinHypercube

from benchmarks import get_benchmark
from src.models.ko_gp import KennedyOHaganGP

torch.set_default_dtype(torch.float64)

BENCHMARK = "Hartmann_6D"
SEED = 43
INITIAL_POINTS = 5
X_TRUE_OPT = torch.tensor(
    [0.2017, 0.1500, 0.4769, 0.2753, 0.3116, 0.6573], dtype=torch.float64
)
RESULT_PATH = "results/mfdro_diag_xt/checkpoints/Hartmann_6D__MF-DRO__seed43.mf.json"

hf_spec = get_benchmark(BENCHMARK + "_HF")
lf_spec = get_benchmark(BENCHMARK + "_LF")
f_hf = hf_spec["make_objective"]()
f_lf = lf_spec["make_objective"]()
d = hf_spec["dim"]
bounds = torch.tensor(
    [hf_spec["domain_min"], hf_spec["domain_max"]], dtype=torch.float64
)

torch.manual_seed(SEED)


def _lhs_points(seed_offset):
    sampler = LatinHypercube(d=d, seed=SEED + seed_offset)
    unit_X = torch.tensor(sampler.random(n=INITIAL_POINTS), dtype=torch.float64)
    return bounds[0] + (bounds[1] - bounds[0]) * unit_X


X_hf_init = _lhs_points(0)
Y_hf_init = f_hf(X_hf_init).reshape(-1)
X_lf_init = _lhs_points(1)
Y_lf_init = f_lf(X_lf_init).reshape(-1)

with open(RESULT_PATH) as f:
    result = json.load(f)
fidelity_trace = result["fidelity_trace"]
x_t_trace = [torch.tensor(x, dtype=torch.float64) for x in result["x_t_trace"]]
y_t_trace = result["y_t_trace"]

# ════════════════════════════════════
# CHECK 1: Initial HF point quality
# ════════════════════════════════════
print("=" * 100)
print("CHECK 1: Initial HF point quality")
print("=" * 100)
for i in range(INITIAL_POINTS):
    print(f"  x_hf[{i}] = {X_hf_init[i].tolist()}  ->  "
          f"f_hf = {Y_hf_init[i].item():.4f}")
max_init_hf = Y_hf_init.max().item()
print(f"max(initial_hf_values) = {max_init_hf:.4f}")
print(f"EXPECT coverage failure if all < 2.0: "
      f"{'ALL < 2.0 (coverage failure signature)' if (Y_hf_init < 2.0).all() else 'at least one >= 2.0'}")

# ════════════════════════════════════
# CHECK 2: Rho and LF GP at true optimum (iteration 0, initial fit only)
# ════════════════════════════════════
print("\n" + "=" * 100)
print("CHECK 2: Rho and LF GP at true optimum (after initial fitting, "
      "before any real BO queries)")
print("=" * 100)
ko0 = KennedyOHaganGP(d=d)
ko0.fit(X_lf_init, Y_lf_init, X_hf_init, Y_hf_init, bounds)

rho_val = ko0.rho.item()
with torch.no_grad():
    mu_L, var_L = ko0.lf_posterior(X_TRUE_OPT.unsqueeze(0))
mu_L = mu_L.item()
sigma_L = var_L.sqrt().item()
true_lf_at_opt = f_lf(X_TRUE_OPT.unsqueeze(0)).reshape(-1)[0].item()

print(f"rho = {rho_val:.4f}  EXPECT in [0.5, 0.9]: "
      f"{'PASS' if 0.5 <= rho_val <= 0.9 else 'OUTSIDE RANGE'}")
print(f"mu_L at x* = {mu_L:.4f} +/- {sigma_L:.4f}  "
      f"EXPECT mu_L > 1.0: {'PASS' if mu_L > 1.0 else 'FAIL'}")
print(f"true LF value at x* = f_lf(x*) = {true_lf_at_opt:.4f}")

# ════════════════════════════════════
# CHECK 3: ROI coverage of true optimum
# ════════════════════════════════════
print("\n" + "=" * 100)
print("CHECK 3: ROI coverage of true optimum")
print("Using GP/ROI state BEFORE each iteration's own query (data through "
      "iteration t-1) -- the ROI that actually governed that iteration's "
      "acquisition decision, same convention as Diagnostic 1.")
print("=" * 100)

CHECK3_ITERS = [0, 3, 6, 9]
n_samples = 500
kappa = 2.0

for t in CHECK3_ITERS:
    X_hf_list = [X_hf_init[i] for i in range(INITIAL_POINTS)]
    Y_hf_list = [Y_hf_init[i].item() for i in range(INITIAL_POINTS)]
    X_lf_list = [X_lf_init[i] for i in range(INITIAL_POINTS)]
    Y_lf_list = [Y_lf_init[i].item() for i in range(INITIAL_POINTS)]
    for j in range(t):
        if fidelity_trace[j] == 1:
            X_hf_list.append(x_t_trace[j])
            Y_hf_list.append(y_t_trace[j])
        else:
            X_lf_list.append(x_t_trace[j])
            Y_lf_list.append(y_t_trace[j])

    ko = KennedyOHaganGP(d=d)
    ko.fit(
        torch.stack(X_lf_list), torch.tensor(Y_lf_list, dtype=torch.float64),
        torch.stack(X_hf_list), torch.tensor(Y_hf_list, dtype=torch.float64),
        bounds,
    )

    # Reproduce _compute_mf_roi_candidates's exact sampling + UCB/LCB filter
    # inline, so max_lcb (not returned by the real function) is accessible.
    X_cand = (bounds[0] + (bounds[1] - bounds[0])
              * torch.rand(n_samples, d, dtype=ko.dtype))
    with torch.no_grad():
        mu_cand, var_cand = ko.hf_posterior(X_cand)
    sigma_cand = var_cand.clamp_min(0).sqrt()
    ucb_cand = mu_cand + kappa * sigma_cand
    lcb_cand = mu_cand - kappa * sigma_cand
    max_lcb = lcb_cand.max().item()
    mask = ucb_cand >= max_lcb
    roi = X_cand[mask]
    if roi.shape[0] >= 10:
        if roi.shape[0] > 50:
            idx = torch.randperm(roi.shape[0])[:50]
            roi = roi[idx]
    else:
        roi = X_cand[:50]

    dists = (roi - X_TRUE_OPT.unsqueeze(0)).norm(dim=1)
    min_dist = dists.min().item()

    with torch.no_grad():
        mu_opt, var_opt = ko.hf_posterior(X_TRUE_OPT.unsqueeze(0))
    mu_opt = mu_opt.item()
    sigma_opt = var_opt.clamp_min(0).sqrt().item()
    ucb_opt = mu_opt + kappa * sigma_opt

    print(f"\niter {t}: roi.shape={tuple(roi.shape)}")
    print(f"  min L2 distance from x* to roi_candidates: {min_dist:.4f}  "
          f"(within 0.1: {'YES' if min_dist <= 0.1 else 'NO'})")
    print(f"  GP posterior at x*: mu={mu_opt:.4f} +/- {sigma_opt:.4f}, "
          f"UCB={ucb_opt:.4f}")
    print(f"  max(LCB) threshold: {max_lcb:.4f}  "
          f"x* passes ROI filter (UCB>=max_LCB): "
          f"{'YES' if ucb_opt >= max_lcb else 'NO'}")

print("\n" + "=" * 100)
print("Checks complete. See conversation for hypothesis interpretation.")
print("=" * 100)
