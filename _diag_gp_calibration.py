"""
Small diagnostic: DKL disabled (dkl_threshold=9999 -> plain ARD RBF kernel
the whole run), same ensemble-diversity/state-vector fixes as before,
Hartmann_6D seed=42, 15 iterations. Purpose: distinguish "GP miscalibration"
from "policy/location collapse" as the driver of the incumbent freeze, using
oracle access to Hartmann_6D's true function (a luxury real BO doesn't have,
but legitimate here purely for MEASUREMENT -- these oracle evals are never
fed to the GP or counted against cost_budget).

Metrics tracked, each aimed at a specific hypothesis:

  GP CALIBRATION (vs. oracle, on a fixed 500-point random probe set):
    - corr(mu_H(probe), f_true(probe))  -- does the GP's ranking of
      good/bad regions match reality? Low/negative = miscalibrated.
    - coverage_95 = fraction of probe points where
      |f_true - mu_H| <= 1.96*sigma_H -- should be ~0.95 if calibrated;
      far below = overconfident (sigma too tight).
    - max(mu_H(probe)) vs current best_hf -- does the GP itself believe
      anything beats the incumbent? If not, that's the GP's own opinion,
      not just the policy's.
    - true value at argmax(mu_H(probe)) vs best_hf -- if the GP's own
      top pick is actually good (oracle-checked) but the policy never
      queries near it, that's a policy/DT gap, not a GP problem.

  QUERY-LEVEL SURPRISE (is the GP calibrated AT THE POINTS IT ACTUALLY
  GETS QUERIED, as opposed to random probe points):
    - z_t = (y_t - mu_pred(x_t)) / sigma_pred(x_t), predicted BEFORE
      observing y_t. |z| consistently > 2-3 = GP surprised by its own
      queries = miscalibration where it matters.

  POLICY/LOCATION COLLAPSE (independent of GP quality):
    - dist_from_prev = ||x_t - x_{t-1}||_2
    - dist_from_incumbent = ||x_t - best_position_HF||_2
    Tight clustering near the same point every iteration = location
    collapse regardless of what the GP believes.
"""
import os
import sys
import statistics as st

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization, _get_mf_state_dim

torch.set_default_dtype(torch.float64)

X_STAR = torch.tensor([0.2017, 0.1500, 0.4769, 0.2753, 0.3116, 0.6573], dtype=torch.float64)
TRUE_HF_AT_XSTAR = 3.322
BENCHMARK = "Hartmann_6D"
SEED = 42
N_ITERS = 50
N_PROBE = 500

cfg = _build_mf_dro_config(
    "diag_gp_calibration", BENCHMARK, "DIAG_NO_DKL", SEED,
    bo_iterations=N_ITERS, num_epochs=10, cost_budget=9999,
    dkl_threshold=9999,  # DISABLE DKL -- plain ARD RBF kernel the whole run
)
print(f"dkl_threshold={cfg.dkl_threshold} (effectively disabled -- ARD RBF only)")

hf_spec = get_benchmark(BENCHMARK + "_HF")
lf_spec = get_benchmark(BENCHMARK + "_LF")
f_hf = hf_spec["make_objective"]()
f_lf = lf_spec["make_objective"]()
bounds = torch.tensor([hf_spec["domain_min"], hf_spec["domain_max"]], dtype=torch.float64)

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
    print(f"  corr(mu_H, f_true)        = {corr:.4f}   (EXPECT: high positive if calibrated)")
    print(f"  MAE(mu_H, f_true)         = {mae:.4f}")
    print(f"  95% coverage              = {coverage_95:.3f}  (EXPECT: ~0.95 if calibrated; "
          f"low = overconfident sigma)")
    print(f"  current best_hf           = {best_hf_now:.4f}")
    print(f"  GP's own believed-best mu = {gp_believed_best:.4f}  "
          f"(does the GP itself think something beats the incumbent?)")
    print(f"  TRUE value at that point  = {gp_true_at_believed_best:.4f}  "
          f"(is the GP's top pick actually good, oracle-checked?)")
    return dict(corr=corr, mae=mae, coverage_95=coverage_95,
                gp_believed_best=gp_believed_best, gp_true_at_believed_best=gp_true_at_believed_best)


mf._sample_initial_points()
mf._update_ko_ensemble()
print(f"\nmax(data_hf_y) at init = {max(mf.data_hf_y):.4f}")
gp_calibration_report("INIT")

print("\n" + "=" * 100)
print(f"PER-ITERATION ({N_ITERS} iterations, ARD RBF only, no DKL)")
print("=" * 100)

prev_x_t = None
z_scores = []
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

    # Query-level surprise: predict BEFORE observing, using the CURRENT
    # (just-refit, pre-this-query) ensemble[0] posterior.
    with torch.no_grad():
        mu_pred, var_pred = mf.ko_ensemble[0].hf_posterior(x_t.unsqueeze(0))
    sigma_pred = max(var_pred.clamp_min(0).sqrt().item(), 1e-8)

    if ell_t == 1:
        y_t = mf.f_hf(x_t.unsqueeze(0)).reshape(-1)[0].item()
        mf.data_hf_x.append(x_t.double())
        mf.data_hf_y.append(y_t)
        z = (y_t - mu_pred.item()) / sigma_pred
        z_scores.append(z)
    else:
        y_t = mf.f_lf(x_t.unsqueeze(0)).reshape(-1)[0].item()
        mf.data_lf_x.append(x_t.double())
        mf.data_lf_y.append(y_t)
        z = float('nan')
    step_cost = mf.c_H if ell_t else mf.c_L
    mf.cumulative_cost += step_cost
    mf.post_init_cost += step_cost
    mf.recent_ell_history.append(ell_t)

    best_hf = max(mf.data_hf_y)
    regret = -best_hf - mf.config.true_opt
    best_idx = int(np.argmax(mf.data_hf_y))
    best_pos = mf.data_hf_x[best_idx]
    dist_from_incumbent = (x_t - best_pos).norm().item()
    dist_from_prev = (x_t - prev_x_t).norm().item() if prev_x_t is not None else float('nan')
    prev_x_t = x_t.clone()

    print(f"iter {t:2d} | ell_t={ell_t} | y_t={y_t:7.4f} | mu_pred={mu_pred.item():7.4f} | "
          f"sigma_pred={sigma_pred:.4f} | z={z:6.2f} | regret={regret:.4f} | best_hf={best_hf:.4f} | "
          f"dist_prev={dist_from_prev:.4f} | dist_incumbent={dist_from_incumbent:.4f}")

    if t in (9, 24, 49):
        gp_calibration_report(f"iter {t}")

print("\n" + "=" * 100)
print("VERDICT")
print("=" * 100)
finite_z = [z for z in z_scores if z == z]  # drop nan (LF steps)
if finite_z:
    print(f"Real-query z-scores (n={len(finite_z)}): "
          f"mean={st.mean(finite_z):.2f}  stdev={st.pstdev(finite_z) if len(finite_z) > 1 else 0:.2f}  "
          f"max|z|={max(abs(z) for z in finite_z):.2f}")
    frac_surprised = sum(1 for z in finite_z if abs(z) > 2) / len(finite_z)
    print(f"Fraction of real queries with |z|>2 (GP 'surprised' by its own query): {frac_surprised:.2f}")
    if frac_surprised > 0.3:
        print("-> GP is frequently surprised by its OWN queries: consistent with miscalibration "
              "AT THE POINTS THAT MATTER, not just in aggregate.")
    else:
        print("-> GP's predictions at queried points are reasonably calibrated -- surprise is "
              "NOT the obvious driver of the freeze.")
else:
    print("No HF real queries with a valid z-score recorded (unexpected).")

print("\nSee per-iteration [GP CALIBRATION] blocks above for the oracle-vs-mu_H trend "
      "(init / iter 4 / iter 9 / iter 14) and dist_prev/dist_incumbent for location-collapse signs.")
