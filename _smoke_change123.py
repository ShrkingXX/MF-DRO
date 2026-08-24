"""
Smoke test for Changes 1-3 (initial_hf/lf=30 fixed + LHS init, DKL active
from iter 0 via dkl_threshold=30, per-step relative RTG). Single run,
Hartmann_6D seed=42, 5 iterations -- mirrors the instrumented-loop pattern
in _diagnostic_v2_worker.py/scripts/test_changes.py (incl. real_hf_warmup
override), but no pre-pass, no resume, no result/diag JSON, no summary
table: this is a smoke test only, not a diagnostic run.
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

cfg = _build_mf_dro_config(
    "smoke_change123", BENCHMARK, "SMOKETEST", SEED,
    bo_iterations=5, num_epochs=10, cost_budget=9999,
)
print("Config check (Changes 1/2 defaults, no explicit override needed):")
print(f"  initial_hf={cfg.initial_hf}  initial_lf={cfg.initial_lf}  "
      f"dkl_threshold={cfg.dkl_threshold}  use_sequential_init={cfg.use_sequential_init}  "
      f"use_rtg_grounding={cfg.use_rtg_grounding}")

hf_spec = get_benchmark(BENCHMARK + "_HF")
lf_spec = get_benchmark(BENCHMARK + "_LF")
f_hf = hf_spec["make_objective"]()
f_lf = lf_spec["make_objective"]()
bounds = torch.tensor([hf_spec["domain_min"], hf_spec["domain_max"]], dtype=torch.float64)

mf = DirectMFRegretOptimization(cfg, f_hf, f_lf, bounds)
mf._sample_initial_points()

print("\n" + "=" * 100)
print("AFTER INITIALIZATION")
print("=" * 100)
print(f"n_hf={len(mf.data_hf_x)}  n_lf={len(mf.data_lf_x)}")
print(f"max(data_hf_y) = {max(mf.data_hf_y):.4f}")

mf._update_ko_ensemble()
ko0 = mf.ko_ensemble[0]
print(f"ko.rho (ensemble[0]) = {ko0.rho.item():.4f}")
print(f"ko.use_dkl (ensemble[0]) = {ko0.use_dkl}")
with torch.no_grad():
    mu0, var0 = ko0.hf_posterior(X_STAR.unsqueeze(0))
sigma0 = var0.clamp_min(0).sqrt().item()
print(f"GP at x*: mu_H={mu0.item():.4f}  sigma_H={sigma0:.4f}  (true={TRUE_HF_AT_XSTAR})")

print("\n" + "=" * 100)
print("PER-ITERATION")
print("=" * 100)

for t in range(mf.config.bo_iterations):
    mf._update_ko_ensemble()
    batch = mf._generate_rollout_batch()
    rtg_target = mf.schemas.update_and_get_rtg_target(batch)
    btg_target = mf.schemas.update_and_get_btg_target(batch)
    mf._last_rtg_target = rtg_target
    if mf.btg_target_base is None:
        mf.btg_target_base = btg_target

    L_loc, L_fid, fid_mean, fid_std = mf._train_dt(batch)
    x_t, ell_t = mf._propose_next_query()
    p_pred_inference = mf.dt.last_p_pred

    real_hf_warmup = getattr(mf.config, 'real_hf_warmup', 2)
    if t < real_hf_warmup:
        ell_t = 1

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

    best_hf = max(mf.data_hf_y)
    regret = -best_hf - mf.config.true_opt
    neg_rtg_frac_batch = float(np.mean([tr['neg_rtg_frac'] for tr in batch]))
    use_dkl_now = mf.ko_ensemble[0].use_dkl

    print(f"iter {t} | use_dkl={use_dkl_now} | n_hf={len(mf.data_hf_x)} | "
          f"n_lf={len(mf.data_lf_x)} | ell_t={ell_t} | "
          f"p_pred_inference={p_pred_inference:.4f} | regret={regret:.4f} | "
          f"best_hf={best_hf:.4f} | rtg_target={rtg_target:.4f} | "
          f"neg_rtg_frac_batch={neg_rtg_frac_batch:.4f}")

print("\nSmoke test complete.")
