import json
import torch
import numpy as np
from scipy.stats.qmc import LatinHypercube

from benchmarks import get_benchmark
from src.models.ko_gp import KennedyOHaganGP
from src.policy.mf_dro import _compute_mf_roi_candidates, simulate_mf_trajectory

torch.set_default_dtype(torch.float64)

BENCHMARK = "Hartmann_6D"
SEED = 43
INITIAL_POINTS = 5
X_TRUE_OPT = torch.tensor(
    [0.2017, 0.1500, 0.4769, 0.2753, 0.3116, 0.6573], dtype=torch.float64
)
# SOURCE NOTE: results/mfdro_stage1_v3/checkpoints/Hartmann_6D__MF-DRO__seed43
# .mf.json (the run actually named in this request) predates the x_t_trace/
# y_t_trace logging added two turns ago -- it has no recorded query
# locations, so its exact per-iteration data state can't be reconstructed.
# Substituting results/mfdro_diag_xt/checkpoints/Hartmann_6D__MF-DRO__seed43
# .mf.json instead: same benchmark, same seed, same v3 config
# (real_hf_warmup=2, minimum_hf_fraction=0.25), also a post-warmup collapse
# to all-LF -- functionally the same experiment, just the specific run that
# has x_t/y_t recorded. Its fidelity_trace is [1,1,1,1,0,0,...] (LF starts
# at iteration 4, not iteration 2 like the named run), so the "post-warmup,
# all-LF phase" checkpoints are adjusted to iterations 4, 6, 9 (not 3, 6, 9)
# to actually land in the collapsed phase for THIS run.
RESULT_PATH = "results/mfdro_diag_xt/checkpoints/Hartmann_6D__MF-DRO__seed43.mf.json"
ROI_CHECK_ITERS = [4, 6, 9]
ROLLOUT_CHECK_ITER = 6
M = 10
ROLLOUTS_PER_MODEL = 7
ROLLOUT_LENGTH = 4
MIN_HF_FRACTION = 0.25
T_REAL = 30
kappa = 2.0

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
print(f"Substitute run loaded: fidelity_trace={fidelity_trace}")


def _data_through(t):
    """Cumulative data BEFORE iteration t's own query (initial + 0..t-1)."""
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
    return X_hf_list, Y_hf_list, X_lf_list, Y_lf_list


def _fit_ko(X_hf_list, Y_hf_list, X_lf_list, Y_lf_list):
    ko = KennedyOHaganGP(d=d)
    ko.fit(
        torch.stack(X_lf_list), torch.tensor(Y_lf_list, dtype=torch.float64),
        torch.stack(X_hf_list), torch.tensor(Y_hf_list, dtype=torch.float64),
        bounds,
    )
    return ko


# ════════════════════════════════════
# PART 1: ROI coverage at iterations 4, 6, 9
# ════════════════════════════════════
print("\n" + "=" * 100)
print("PART 1: ROI coverage of x* at iterations 4, 6, 9 (post-warmup, all-LF phase)")
print("Using GP/ROI state BEFORE each iteration's own query (data through t-1).")
print("=" * 100)

for t in ROI_CHECK_ITERS:
    X_hf_list, Y_hf_list, X_lf_list, Y_lf_list = _data_through(t)
    ko = _fit_ko(X_hf_list, Y_hf_list, X_lf_list, Y_lf_list)

    roi = _compute_mf_roi_candidates(ko, bounds)
    dists = (roi - X_TRUE_OPT.unsqueeze(0)).norm(dim=1)
    min_dist = dists.min().item()

    with torch.no_grad():
        mu_opt, var_opt = ko.hf_posterior(X_TRUE_OPT.unsqueeze(0))
    mu_opt = mu_opt.item()
    sigma_opt = var_opt.clamp_min(0).sqrt().item()
    ucb_opt = mu_opt + kappa * sigma_opt

    # Recompute the SAME max_lcb threshold this ko/iteration used, by
    # replicating _compute_mf_roi_candidates's internal sampling+filter
    # (max_lcb isn't returned by the real function).
    X_cand = (bounds[0] + (bounds[1] - bounds[0])
              * torch.rand(500, d, dtype=ko.dtype))
    with torch.no_grad():
        mu_cand, var_cand = ko.hf_posterior(X_cand)
    lcb_cand = mu_cand - kappa * var_cand.clamp_min(0).sqrt()
    max_lcb = lcb_cand.max().item()
    passes = ucb_opt >= max_lcb

    print(f"\niter {t}: roi_candidates.shape={tuple(roi.shape)} "
          f"(size passing UCB filter, capped at 50)")
    print(f"  min L2 distance from x* to roi_candidates: {min_dist:.4f}")
    print(f"  x* UCB={ucb_opt:.4f}  max(LCB)={max_lcb:.4f}  "
          f"x* passes UCB>=max(LCB): {'YES' if passes else 'NO'}")

# ════════════════════════════════════
# PART 2: Rollout coverage at iteration 6
# ════════════════════════════════════
print("\n" + "=" * 100)
print(f"PART 2: Rollout coverage at iteration {ROLLOUT_CHECK_ITER} "
      f"({M} models x {ROLLOUTS_PER_MODEL} rollouts/model = "
      f"{M * ROLLOUTS_PER_MODEL} rollouts, {ROLLOUT_LENGTH} steps each)")
print("=" * 100)

X_hf_list, Y_hf_list, X_lf_list, Y_lf_list = _data_through(ROLLOUT_CHECK_ITER)
ko_ensemble = [_fit_ko(X_hf_list, Y_hf_list, X_lf_list, Y_lf_list) for _ in range(M)]

all_step_dists = []
for ko in ko_ensemble:
    for _ in range(ROLLOUTS_PER_MODEL):
        traj = simulate_mf_trajectory(
            ko,
            (X_hf_list, Y_hf_list), (X_lf_list, Y_lf_list),
            rollout_length=ROLLOUT_LENGTH, c_H=c_H, c_L=c_L,
            bounds=bounds,
            n_real_iter=len(Y_hf_list), T_real=T_REAL,
            ko_ensemble_full=ko_ensemble,
            minimum_hf_fraction=MIN_HF_FRACTION,
        )
        actions_x = traj['actions_x']  # [rollout_length, d]
        dists = (actions_x - X_TRUE_OPT.unsqueeze(0)).norm(dim=1)
        all_step_dists.extend(dists.tolist())

all_step_dists = np.array(all_step_dists)
n_steps = len(all_step_dists)
n_within_02 = int((all_step_dists <= 0.2).sum())
frac_within_02 = n_within_02 / n_steps

print(f"\nTotal rollout steps generated: {n_steps} "
      f"({M}x{ROLLOUTS_PER_MODEL}x{ROLLOUT_LENGTH})")
print(f"min step distance to x* across all rollouts: {all_step_dists.min():.4f}")
print(f"steps within 0.2 L2 of x*: {n_within_02}/{n_steps} "
      f"({frac_within_02*100:.2f}%)")

print("\n" + "=" * 100)
print("Check complete.")
print("=" * 100)
