"""
Minor diagnostic for the ensemble-diversity fix (rho_init/initial_lengthscale
per member, no warm-start when diversified) and the state-vector fix (raw KO
hyperparameters instead of candidate-dependent sigma summaries, Fix 1).
Single run, Hartmann_6D seed=42, 15 iterations, no cost cap -- checks (a) no
crash, (b) ensemble members are actually diverse post-fit (not just at
init), (c) whether the incumbent-freeze pathology still occurs.
"""
import os
import sys
import statistics as st

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization, _get_mf_state_dim, _ko_hp_features

torch.set_default_dtype(torch.float64)

X_STAR = torch.tensor([0.2017, 0.1500, 0.4769, 0.2753, 0.3116, 0.6573], dtype=torch.float64)
TRUE_HF_AT_XSTAR = 3.322
BENCHMARK = "Hartmann_6D"
SEED = 42
N_ITERS = 15

cfg = _build_mf_dro_config(
    "diag_ensemble_state_fix", BENCHMARK, "DIAG", SEED,
    bo_iterations=N_ITERS, num_epochs=10, cost_budget=9999,
)
hf_spec = get_benchmark(BENCHMARK + "_HF")
lf_spec = get_benchmark(BENCHMARK + "_LF")
f_hf = hf_spec["make_objective"]()
f_lf = lf_spec["make_objective"]()
bounds = torch.tensor([hf_spec["domain_min"], hf_spec["domain_max"]], dtype=torch.float64)

mf = DirectMFRegretOptimization(cfg, f_hf, f_lf, bounds)
d = mf.d
M = cfg.M
print(f"d={d}  M={M}  expected state_dim = 5*M+d+5 = {5*M+d+5}  "
      f"(_get_mf_state_dim reports {_get_mf_state_dim(d, M)})")

mf._sample_initial_points()
mf._update_ko_ensemble()

print("\n" + "=" * 100)
print("ENSEMBLE DIVERSITY CHECK (post-fit, not just at construction)")
print("=" * 100)
rhos = [ko.rho.item() for ko in mf.ko_ensemble]
ls_lf_vals = [_ko_hp_features(ko)[0] for ko in mf.ko_ensemble]
print(f"rho per member:        {[f'{r:.3f}' for r in rhos]}")
print(f"  min={min(rhos):.3f} max={max(rhos):.3f} std={st.pstdev(rhos):.4f}  (EXPECT: std > 0, not ~identical)")
print(f"gp_lf lengthscale per member: {[f'{v:.3f}' for v in ls_lf_vals]}")
print(f"  min={min(ls_lf_vals):.3f} max={max(ls_lf_vals):.3f} std={st.pstdev(ls_lf_vals):.4f}  (EXPECT: std > 0)")

ko0 = mf.ko_ensemble[0]
print(f"\nko_ensemble[0].use_dkl = {ko0.use_dkl}")
with torch.no_grad():
    mu0, var0 = ko0.hf_posterior(X_STAR.unsqueeze(0))
sigma0 = var0.clamp_min(0).sqrt().item()
print(f"GP[0] at x*: mu_H={mu0.item():.4f}  sigma_H={sigma0:.4f}  (true={TRUE_HF_AT_XSTAR})")
print(f"max(data_hf_y) at init = {max(mf.data_hf_y):.4f}")

print("\n" + "=" * 100)
print(f"PER-ITERATION ({N_ITERS} iterations)")
print("=" * 100)

regrets = []
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
    regrets.append(regret)
    neg_rtg_frac_batch = float(np.mean([tr['neg_rtg_frac'] for tr in batch]))
    use_dkl_now = mf.ko_ensemble[0].use_dkl

    rho_std_now = st.pstdev([ko.rho.item() for ko in mf.ko_ensemble])

    print(f"iter {t:2d} | use_dkl={use_dkl_now} | ell_t={ell_t} | "
          f"p_pred_inf={p_pred_inference:.4f} | regret={regret:.4f} | "
          f"best_hf={best_hf:.4f} | rtg_target={rtg_target:.4f} | "
          f"neg_rtg_frac={neg_rtg_frac_batch:.4f} | ensemble_rho_std={rho_std_now:.4f}")

print("\n" + "=" * 100)
print("VERDICT")
print("=" * 100)
distinct_regrets = len(set(f"{r:.6f}" for r in regrets))
print(f"Distinct regret values across {N_ITERS} iterations: {distinct_regrets}")
if distinct_regrets == 1:
    print("INCUMBENT FROZEN: regret never changed across all iterations.")
elif distinct_regrets <= 2:
    print("MOSTLY FROZEN: regret changed at most once (likely just warmup).")
else:
    print("NOT FROZEN: regret changed multiple times.")
print(f"regret trace: {[f'{r:.4f}' for r in regrets]}")
