import statistics
import torch

import src.policy.mf_dro as mf_dro_mod
from src.policy.mf_dro import simulate_mf_trajectory
from src.models.ko_gp import KennedyOHaganGP

torch.set_default_dtype(torch.float64)


def make_fitted_ko(d=6, n_lf=30, n_hf=20, dkl_threshold=9999, seed=0):
    torch.manual_seed(seed)
    bounds = torch.zeros(2, d, dtype=torch.float64)
    bounds[1] = 1.0
    center = torch.rand(d, dtype=torch.float64) * 0.6 + 0.2
    X_lf = torch.rand(n_lf, d, dtype=torch.float64)
    Y_lf = -((X_lf - center) ** 2).sum(dim=-1) + 0.05 * torch.randn(n_lf, dtype=torch.float64)
    X_hf = torch.rand(n_hf, d, dtype=torch.float64)
    Y_hf = -((X_hf - center) ** 2).sum(dim=-1) + 0.01 * torch.randn(n_hf, dtype=torch.float64)
    ko = KennedyOHaganGP(d=d, dkl_threshold=dkl_threshold)
    ko.fit(X_lf, Y_lf, X_hf, Y_hf, bounds)
    return ko, bounds, X_lf, Y_lf, X_hf, Y_hf


ko, bounds, X_lf, Y_lf, X_hf, Y_hf = make_fitted_ko(seed=0)
ensemble = [ko, ko, ko]
real_data_hf = ([x for x in X_hf], [float(y) for y in Y_hf])
real_data_lf = ([x for x in X_lf], [float(y) for y in Y_lf])

orig = mf_dro_mod.compute_joint_mf_mes
min_ratios = []
last_ratios = []
all_ratios = []

for i in range(15):
    trace = []

    def traced(ko_model, roi_candidates, c_H, c_L, K=10):
        r = orig(ko_model, roi_candidates, c_H, c_L, K=K)
        trace.append(r[2].max().item())
        return r

    mf_dro_mod.compute_joint_mf_mes = traced
    torch.manual_seed(1000 + i)
    simulate_mf_trajectory(
        ko, real_data_hf, real_data_lf, rollout_length=8, c_H=5.0, c_L=1.0,
        bounds=bounds, n_real_iter=20, T_real=30, ko_ensemble_full=ensemble,
        minimum_hf_fraction=0.0, bes_delta=0.0,  # disabled -> always runs all 8, full trace captured
    )
    mf_dro_mod.compute_joint_mf_mes = orig

    sig0 = trace[0]
    ratios = [s / sig0 for s in trace[1:]]
    min_ratios.append(min(ratios))
    last_ratios.append(ratios[-1])
    all_ratios.extend(ratios)
    print(f"rollout {i}: sig0={sig0:.4f}  ratios(tau1..7)="
          f"{[f'{r:.3f}' for r in ratios]}  min={min(ratios):.3f}")

print()
print(f"Across 15 rollouts (8 steps each, 105 tau>0 steps total):")
print(f"  mean of per-rollout MIN ratio reached: {statistics.mean(min_ratios):.3f}")
print(f"  lowest ratio ever observed (any tau, any rollout): {min(all_ratios):.3f}")
print(f"  mean of per-rollout LAST-step ratio: {statistics.mean(last_ratios):.3f}")
print(f"  overall mean ratio across all tau>0 steps: {statistics.mean(all_ratios):.3f}")
print()
print("Implication: bes_delta would need to be set close to (or above) the "
      "lowest ratio actually reached within an 8-step rollout for BES to "
      "ever fire on this setup; 0.05 is far below that floor.")
