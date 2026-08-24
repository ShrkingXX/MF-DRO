"""
Follow-up analysis on the SAME run as _diag_gp_calibration.py (identical
config/seed -> deterministic, bit-identical y_t trace, verified below):
for each real HF query, was it near the probe point the GP itself ranked
highest (gp_bestprobe, true value 2.598, found at iter9/14 of the prior
run)? Answers: is the failure "DT/acquisition ignoring the GP's own
signal" (location pathology) or "GP signal is used but the basin genuinely
needs more iterations / a better prior" (still-immature GP)?

No new BO iterations beyond the original 15 -- this instruments the exact
same deterministic run to capture x_t and the per-probe GP posterior mean
at each iteration (neither of which the original script saved), rather
than computing anything new.
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

BENCHMARK = "Hartmann_6D"
SEED = 42
N_ITERS = 15
N_PROBE = 500

cfg = _build_mf_dro_config(
    "diag_hf_vs_bestprobe", BENCHMARK, "DIAG_NO_DKL", SEED,
    bo_iterations=N_ITERS, num_epochs=10, cost_budget=9999,
    dkl_threshold=9999,
)

hf_spec = get_benchmark(BENCHMARK + "_HF")
lf_spec = get_benchmark(BENCHMARK + "_LF")
f_hf = hf_spec["make_objective"]()
f_lf = lf_spec["make_objective"]()
bounds = torch.tensor([hf_spec["domain_min"], hf_spec["domain_max"]], dtype=torch.float64)

mf = DirectMFRegretOptimization(cfg, f_hf, f_lf, bounds)
d = mf.d

torch.manual_seed(999)  # SAME probe set as _diag_gp_calibration.py
PROBE_X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(N_PROBE, d, dtype=torch.float64)
with torch.no_grad():
    PROBE_Y_TRUE = f_hf(PROBE_X).reshape(-1)

mf._sample_initial_points()
print(f"max(data_hf_y) at init = {max(mf.data_hf_y):.4f}  "
      f"(EXPECT 2.5652, confirms this is the same deterministic run)")

x_t_by_iter = {}
ell_by_iter = {}
y_by_iter = {}
mu_pred_at_query_by_iter = {}
mu_H_probes_by_iter = {}  # [500] mu_H over the FIXED probe set, per iter

for t in range(mf.config.bo_iterations):
    mf._update_ko_ensemble()
    ko0 = mf.ko_ensemble[0]
    with torch.no_grad():
        mu_probes, _ = ko0.hf_posterior(PROBE_X)
    mu_H_probes_by_iter[t] = mu_probes.numpy().copy()

    batch = mf._generate_rollout_batch()
    rtg_target = mf.schemas.update_and_get_rtg_target(batch)
    btg_target = mf.schemas.update_and_get_btg_target(batch)
    mf._last_rtg_target = rtg_target
    if mf.btg_target_base is None:
        mf.btg_target_base = btg_target

    mf._train_dt(batch)
    x_t, ell_t = mf._propose_next_query()

    real_hf_warmup = getattr(mf.config, 'real_hf_warmup', 2)
    if t < real_hf_warmup:
        ell_t = 1

    with torch.no_grad():
        mu_pred, _ = ko0.hf_posterior(x_t.unsqueeze(0))
    mu_pred_at_query_by_iter[t] = mu_pred.item()

    if ell_t == 1:
        y_t = mf.f_hf(x_t.unsqueeze(0)).reshape(-1)[0].item()
        mf.data_hf_x.append(x_t.double())
        mf.data_hf_y.append(y_t)
    else:
        y_t = mf.f_lf(x_t.unsqueeze(0)).reshape(-1)[0].item()
        mf.data_lf_x.append(x_t.double())
        mf.data_lf_y.append(y_t)
    step_cost = mf.c_H if ell_t else mf.c_L
    mf.cumulative_cost += step_cost
    mf.post_init_cost += step_cost
    mf.recent_ell_history.append(ell_t)

    x_t_by_iter[t] = x_t.clone()
    ell_by_iter[t] = ell_t
    y_by_iter[t] = y_t

    print(f"iter {t:2d} | ell_t={ell_t} | y_t={y_t:7.4f}  (verifying against prior run's trace)")

# gp_bestprobe from the FINAL (iter14) state, matching the prior run's report.
final_mu = mu_H_probes_by_iter[N_ITERS - 1]
gp_best_idx = int(np.argmax(final_mu))
gp_bestprobe_x = torch.tensor(PROBE_X[gp_best_idx].numpy(), dtype=torch.float64)
gp_bestprobe_true_y = PROBE_Y_TRUE[gp_best_idx].item()
print(f"\ngp_bestprobe (argmax mu_H at iter14): true_y={gp_bestprobe_true_y:.4f}  "
      f"(EXPECT ~2.598, confirms same probe/run as before)")
print(f"gp_bestprobe coords: {[f'{v:.4f}' for v in gp_bestprobe_x.tolist()]}")

print("\n" + "=" * 100)
print("1. REAL HF QUERIES: iter | x_t | true y_t | dist_to_gp_bestprobe")
print("=" * 100)
hf_iters = [t for t in range(N_ITERS) if ell_by_iter[t] == 1]
dists = []
for t in hf_iters:
    x_t = x_t_by_iter[t]
    dist = (x_t - gp_bestprobe_x).norm().item()
    dists.append(dist)
    print(f"iter {t:2d} | x_t={[f'{v:.4f}' for v in x_t.tolist()]} | "
          f"y_t={y_by_iter[t]:7.4f} | dist_to_gp_bestprobe={dist:.4f}")

mean_dist = float(np.mean(dists))
print(f"\nMean distance (9 HF queries -> gp_bestprobe): {mean_dist:.4f}")

print("\n" + "=" * 100)
print("2. BEST REAL HF QUERY vs gp_bestprobe")
print("=" * 100)
best_hf_iter = max(hf_iters, key=lambda t: y_by_iter[t])
best_hf_dist = (x_t_by_iter[best_hf_iter] - gp_bestprobe_x).norm().item()
print(f"Best HF query: iter {best_hf_iter}, y_t={y_by_iter[best_hf_iter]:.4f}")
print(f"Distance(best HF query, gp_bestprobe) = {best_hf_dist:.4f}")

print("\n" + "=" * 100)
print(f"3. AT ITERATION {best_hf_iter} (best HF query, y_t={y_by_iter[best_hf_iter]:.4f}): "
      f"was the GP already pointing at gp_bestprobe?")
print("=" * 100)
mu_at_query_iter7 = mu_pred_at_query_by_iter[best_hf_iter]
mu_at_bestprobe_iter7 = float(mu_H_probes_by_iter[best_hf_iter][gp_best_idx])
print(f"GP posterior mean at x_t (the actual query, iter {best_hf_iter}'s own GP state) = "
      f"{mu_at_query_iter7:.4f}")
print(f"GP posterior mean at gp_bestprobe (SAME iter {best_hf_iter} GP state)          = "
      f"{mu_at_bestprobe_iter7:.4f}")
if mu_at_bestprobe_iter7 > mu_at_query_iter7:
    print(f"-> At iteration {best_hf_iter}, the GP ALREADY believed gp_bestprobe "
          f"(mu={mu_at_bestprobe_iter7:.4f}) was better than where it actually queried "
          f"(mu={mu_at_query_iter7:.4f}), by {mu_at_bestprobe_iter7 - mu_at_query_iter7:.4f}. "
          f"The GP was pointing the right way; the query didn't go there.")
else:
    print(f"-> At iteration {best_hf_iter}, the GP did NOT yet believe gp_bestprobe was better "
          f"than the actual query location -- the GP itself hadn't identified it as good yet "
          f"at that point in the run.")

print("\n" + "=" * 100)
print("INTERPRETATION")
print("=" * 100)
best_y = y_by_iter[best_hf_iter]
print(f"Mean dist(HF queries, gp_bestprobe) = {mean_dist:.4f}")
print(f"Best HF y observed = {best_y:.4f}")
if mean_dist > 0.3:
    print("-> dist > 0.3: DT/acquisition NOT using the GP's own signal -- location pathology.")
    print("   Fix 3 (candidate scoring) is next priority.")
else:
    print("-> dist < 0.3: GP signal IS being used, but the basin is too narrow / GP still "
          "immature -- need more iterations or a lognormal prior for a better GP.")
    print("   Recommend: run 50 iterations to see if it converges.")
