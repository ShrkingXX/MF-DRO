import json
import math
import torch
import numpy as np
from scipy.stats.qmc import LatinHypercube

from benchmarks import get_benchmark
from src.models.ko_gp import KennedyOHaganGP
from src.policy.mf_dro import (
    _build_hf_proxy_model, _lf_mes_info_gain, _compute_mf_roi_candidates,
)
from gumbel_thompson import thompson_sample_y_star
from mes_reward import compute_mes_reward

torch.set_default_dtype(torch.float64)

BENCHMARK = "Hartmann_6D"
SEED = 43
INITIAL_POINTS = 5
N_DIAG_ITERS = 12  # Diagnostic 1: replay first 12 iterations
DIAG2_LF_CHECKPOINTS = [3, 6, 10]
X_TRUE_OPT = torch.tensor(
    [0.2017, 0.1500, 0.4769, 0.2753, 0.3116, 0.6573], dtype=torch.float64
)

RESULT_PATH = "results/mfdro_diag_xt/checkpoints/Hartmann_6D__MF-DRO__seed43.mf.json"

with open(RESULT_PATH) as f:
    result = json.load(f)

fidelity_trace = result["fidelity_trace"]
x_t_trace = [torch.tensor(x, dtype=torch.float64) for x in result["x_t_trace"]]
y_t_trace = result["y_t_trace"]
n_real_iters = len(fidelity_trace)
print(f"Loaded diagnostic run: {n_real_iters} real iterations, "
      f"fidelity_trace={fidelity_trace}")

hf_spec = get_benchmark(BENCHMARK + "_HF")
lf_spec = get_benchmark(BENCHMARK + "_LF")
f_hf = hf_spec["make_objective"]()
f_lf = lf_spec["make_objective"]()
d = hf_spec["dim"]
c_H = hf_spec["cost"]
c_L = lf_spec["cost"]
bounds = torch.tensor(
    [hf_spec["domain_min"], hf_spec["domain_max"]], dtype=torch.float64
)

# Regenerate the deterministic initial LHS points (mirrors
# DirectMFRegretOptimization._sample_initial_points exactly: seed_offset=0
# for HF, seed_offset=1 for LF).
torch.manual_seed(SEED)


def _lhs_points(seed_offset):
    sampler = LatinHypercube(d=d, seed=SEED + seed_offset)
    unit_X = torch.tensor(sampler.random(n=INITIAL_POINTS), dtype=torch.float64)
    return bounds[0] + (bounds[1] - bounds[0]) * unit_X


X_hf_init = _lhs_points(0)
Y_hf_init = f_hf(X_hf_init).reshape(-1)
X_lf_init = _lhs_points(1)
Y_lf_init = f_lf(X_lf_init).reshape(-1)

print(f"Initial points regenerated: {INITIAL_POINTS} HF + {INITIAL_POINTS} LF "
      f"(deterministic LHS, seed={SEED})")

# ════════════════════════════════════
# DIAGNOSTIC 1: Raw MES values at queried locations
# ════════════════════════════════════
print("\n" + "=" * 100)
print("DIAGNOSTIC 1: Raw MES values at queried locations "
      f"(first {N_DIAG_ITERS} iterations)")
print("CAVEAT: roi_candidates and Thompson y* draws are NOT seeded in the "
      "original code, so these are fresh unbiased re-estimates of the same "
      "quantity the real run saw, not bit-exact reproductions of that "
      "iteration's internal random draw.")
print("=" * 100)
print(f"{'iter':>4} | {'ell_t':>5} | {'mes_hf_queried':>14} | "
      f"{'mes_lf_queried':>14} | {'mes_hf_true_opt':>15} | "
      f"{'ratio':>8} | cost_norm_winner")

flat_by_iter = 0
ratio_gt3_but_lf = 0

for t in range(min(N_DIAG_ITERS, n_real_iters)):
    # Data that existed BEFORE this iteration's own query: initial points +
    # real iterations 0..t-1.
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

    hf_proxy = _build_hf_proxy_model(ko)
    roi_candidates = _compute_mf_roi_candidates(ko, bounds)

    x_t = x_t_trace[t]
    ell_t = fidelity_trace[t]

    mes_hf_queried = compute_mes_reward(
        x_t.unsqueeze(0), hf_proxy, roi_candidates
    ).item()
    y_star_arr = thompson_sample_y_star(hf_proxy, roi_candidates, K=10)
    mes_lf_queried = _lf_mes_info_gain(x_t, ko, y_star_arr, n_quad=32)

    mes_hf_true_opt = compute_mes_reward(
        X_TRUE_OPT.unsqueeze(0), hf_proxy, roi_candidates
    ).item()

    ratio = mes_hf_queried / mes_lf_queried if mes_lf_queried > 1e-300 else float('inf')
    hf_score = mes_hf_queried / c_H
    lf_score = mes_lf_queried / c_L
    winner = "HF" if hf_score > lf_score else "LF"

    if mes_hf_queried < 1e-4 and mes_lf_queried < 1e-4:
        flat_by_iter += 1
    if ratio > 3 and winner == "LF":
        ratio_gt3_but_lf += 1

    print(f"{t:4d} | {ell_t:5d} | {mes_hf_queried:14.6f} | "
          f"{mes_lf_queried:14.6f} | {mes_hf_true_opt:15.6f} | "
          f"{ratio:8.3f} | {winner}")

print("\n--- Diagnostic 1 interpretation ---")
if flat_by_iter >= 5:
    print(f"MES collapsed. ({flat_by_iter}/{min(N_DIAG_ITERS, n_real_iters)} "
          f"iterations had both mes_hf and mes_lf < 1e-4.)")
if ratio_gt3_but_lf > 0:
    print(f"Cost ratio dominates switching. "
          f"({ratio_gt3_but_lf} iterations had mes_hf/mes_lf > 3 "
          f"but LF still won after cost normalization.)")

# ════════════════════════════════════
# DIAGNOSTIC 2: KO GP posterior at queried HF locations
# ════════════════════════════════════
print("\n" + "=" * 100)
print("DIAGNOSTIC 2: KO GP posterior at queried HF locations / true opt")
print("Using GP state AFTER conditioning on that iteration's own query "
      "(cumulative knowledge as of iteration t).")
print("=" * 100)

hf_query_iters = [t for t in range(min(N_DIAG_ITERS, n_real_iters))
                   if fidelity_trace[t] == 1]
checkpoints = sorted(set(hf_query_iters + [c for c in DIAG2_LF_CHECKPOINTS
                                            if c < n_real_iters]))

f_hf_true_opt_val = f_hf(X_TRUE_OPT.unsqueeze(0)).reshape(-1)[0].item()

for t in checkpoints:
    X_hf_list = [X_hf_init[i] for i in range(INITIAL_POINTS)]
    Y_hf_list = [Y_hf_init[i].item() for i in range(INITIAL_POINTS)]
    X_lf_list = [X_lf_init[i] for i in range(INITIAL_POINTS)]
    Y_lf_list = [Y_lf_init[i].item() for i in range(INITIAL_POINTS)]
    for j in range(t + 1):  # inclusive: through iteration t
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

    with torch.no_grad():
        mu_opt, var_opt = ko.hf_posterior(X_TRUE_OPT.unsqueeze(0))
    mu_opt = mu_opt.item()
    sigma_opt = max(var_opt.sqrt().item(), 0.0)

    if fidelity_trace[t] == 1:
        x_hf = x_t_trace[t]
        with torch.no_grad():
            mu_q, var_q = ko.hf_posterior(x_hf.unsqueeze(0))
        mu_q = mu_q.item()
        sigma_q = max(var_q.sqrt().item(), 0.0)
        f_true_q = f_hf(x_hf.unsqueeze(0)).reshape(-1)[0].item()
        print(f"\nHF query at iter {t}:")
        print(f"  Queried location: {x_hf.tolist()}")
        print(f"  GP posterior at queried: mu={mu_q:.4f} +/- {sigma_q:.4f}")
        print(f"  GP posterior at true opt: mu={mu_opt:.4f} +/- {sigma_opt:.4f}")
        print(f"  True function value at queried: {f_true_q:.4f}")
        print(f"  True function value at true opt: {f_hf_true_opt_val:.4f}")
    else:
        print(f"\nIter {t} (LF query): GP posterior at true opt:")
        print(f"  mu={mu_opt:.4f} +/- {sigma_opt:.4f}")

print("\n--- Diagnostic 2 interpretation ---")
last_mu_opt = mu_opt
last_sigma_opt = sigma_opt
if last_mu_opt > 2.0 and last_sigma_opt < 1.0:
    print("GP correctly identifies the optimum but DT is not going there. "
          "Location head failure.")
elif last_mu_opt < 0.5:
    print("GP does not think the true optimum is good. "
          "KO GP calibration failure.")
if last_sigma_opt > 1.5:
    print("GP has not explored anywhere near the optimum. "
          "Coverage failure -- ROI not reaching optimum region.")
