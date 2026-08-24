"""
Generalized version of _diag_gp_calibration.py, parameterized by benchmark
and iteration count, for the Currin_2D / Borehole_8D sanity checks before
committing to the full 4-variant diagnostic. Same lognormal-prior fix,
noise_lb=1e-2, dkl_threshold=9999 (disabled), initial_hf=initial_lf=30,
seed=42 as the Hartmann_6D run. Drops the Hartmann-specific x*/true-value
print (not one of the 4 requested metrics here, and Currin_2D/Borehole_8D
don't have a memorized x* the way Hartmann_6D's registry comment does).

Usage: python _diag_gp_calibration_multi.py <BENCHMARK> <N_ITERS>
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

BENCHMARK = sys.argv[1]
N_ITERS = int(sys.argv[2])
SEED = 42
N_PROBE = 500

cfg = _build_mf_dro_config(
    f"diag_gp_calibration_{BENCHMARK}", BENCHMARK, "DIAG_NO_DKL", SEED,
    bo_iterations=N_ITERS, num_epochs=10, cost_budget=9999,
    dkl_threshold=9999,
)
print(f"BENCHMARK={BENCHMARK}  N_ITERS={N_ITERS}  "
      f"dkl_threshold={cfg.dkl_threshold} (disabled)  "
      f"initial_hf={cfg.initial_hf}  initial_lf={cfg.initial_lf}")

hf_spec = get_benchmark(BENCHMARK + "_HF")
lf_spec = get_benchmark(BENCHMARK + "_LF")
f_hf = hf_spec["make_objective"]()
f_lf = lf_spec["make_objective"]()
bounds = torch.tensor([hf_spec["domain_min"], hf_spec["domain_max"]], dtype=torch.float64)
print(f"dim={hf_spec['dim']}  true_opt={hf_spec['known_optimal_value']:.4f}  "
      f"c_H={hf_spec['cost']}  c_L={lf_spec['cost']}")

mf = DirectMFRegretOptimization(cfg, f_hf, f_lf, bounds)
d = mf.d
M = cfg.M

torch.manual_seed(999)  # fixed probe set, independent of the run's own RNG stream
PROBE_X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(N_PROBE, d, dtype=torch.float64)
with torch.no_grad():
    PROBE_Y_TRUE = f_hf(PROBE_X).reshape(-1)  # oracle, diagnostic only


def gp_calibration_report(tag):
    ko0 = mf.ko_ensemble[0]
    with torch.no_grad():
        mu, var = ko0.hf_posterior(PROBE_X)
    sigma = var.clamp_min(0).sqrt()
    mu_np, sigma_np, true_np = mu.numpy(), sigma.numpy(), PROBE_Y_TRUE.numpy()

    corr = float(np.corrcoef(mu_np, true_np)[0, 1])
    resid = np.abs(true_np - mu_np)
    coverage_95 = float(np.mean(resid <= 1.96 * sigma_np))
    mae = float(np.mean(resid))

    best_hf_now = max(mf.data_hf_y)
    gp_best_idx = int(np.argmax(mu_np))
    gp_believed_best = float(mu_np[gp_best_idx])
    gp_true_at_believed_best = float(true_np[gp_best_idx])

    ls_lf_mean = ko0.gp_lf.covar_module.base_kernel.lengthscale.detach().mean().item()
    ls_delta_mean = ko0.gp_delta.covar_module.base_kernel.lengthscale.detach().mean().item()

    print(f"\n[GP CALIBRATION @ {tag}] (N={N_PROBE} random probe points, oracle-checked)")
    print(f"  fitted lengthscale (mean over ARD dims): gp_lf={ls_lf_mean:.4f}  gp_delta={ls_delta_mean:.4f}")
    print(f"  corr(mu_H, f_true)        = {corr:.4f}")
    print(f"  MAE(mu_H, f_true)         = {mae:.4f}")
    print(f"  95% coverage              = {coverage_95:.3f}")
    print(f"  current best_hf           = {best_hf_now:.4f}")
    print(f"  GP's own believed-best mu = {gp_believed_best:.4f}")
    print(f"  TRUE value at that point  = {gp_true_at_believed_best:.4f}")
    return dict(ls_lf=ls_lf_mean, ls_delta=ls_delta_mean, corr=corr, mae=mae,
                coverage_95=coverage_95, best_hf=best_hf_now,
                gp_believed_best=gp_believed_best,
                gp_true_at_believed_best=gp_true_at_believed_best)


mf._sample_initial_points()

# Borehole normalization gate REMOVED per explicit correction: 18.7-162.1
# m^3/yr and known_optimal_value=-309.5756 are both correct for this
# benchmark's actual domain -- the earlier [0.002, 0.310] reference range
# was the wrong expectation, not a bug in the code.

mf._update_ko_ensemble()
print(f"\nmax(data_hf_y) at init = {max(mf.data_hf_y):.4f}")
init_report = gp_calibration_report("INIT")

print("\n" + "=" * 100)
print(f"PER-ITERATION ({N_ITERS} iterations, ARD RBF only, no DKL, lognormal prior)")
print("=" * 100)

for t in range(mf.config.bo_iterations):
    mf._update_ko_ensemble()
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
    print(f"iter {t:2d} | ell_t={ell_t} | y_t={y_t:9.4f} | regret={regret:.4f} | best_hf={best_hf:.4f}")

final_report = gp_calibration_report(f"FINAL (iter {N_ITERS - 1})")

print("\n" + "=" * 100)
print(f"SUMMARY: {BENCHMARK}")
print("=" * 100)
print(f"lengthscale (gp_lf/gp_delta): init={init_report['ls_lf']:.4f}/{init_report['ls_delta']:.4f}  "
      f"final={final_report['ls_lf']:.4f}/{final_report['ls_delta']:.4f}")
print(f"corr(mu_H, f_true): init={init_report['corr']:.4f}  final={final_report['corr']:.4f}")
print(f"best_hf: init={init_report['best_hf']:.4f}  final={final_report['best_hf']:.4f}  "
      f"improved={final_report['best_hf'] > init_report['best_hf']}")
print(f"GP believed-best mu_H: init={init_report['gp_believed_best']:.4f} "
      f"(true={init_report['gp_true_at_believed_best']:.4f})  "
      f"final={final_report['gp_believed_best']:.4f} "
      f"(true={final_report['gp_true_at_believed_best']:.4f})")
