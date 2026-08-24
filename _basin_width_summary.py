"""
Summary table for the basin-width sweep (mfdro_basin_width_sweep):
alpha | mean_neg_rtg_frac (zero_reward_frac substitute -- see
_basin_width_worker.py's module docstring) | mean_frac_near_xstar |
mean_n_improved | mean_final_regret -- each averaged over the 3 seeds run
per alpha_basin variant.
"""
import json
import glob
import os

CKPT_DIR = "results/mfdro_basin_width_sweep/checkpoints"
VARIANTS = ["Hartmann6D_w10", "Hartmann6D_w03", "Hartmann6D_w01"]
SEEDS = [42, 43, 44]

rows = []
for variant in VARIANTS:
    per_seed = []
    for seed in SEEDS:
        path = os.path.join(CKPT_DIR, f"{variant}__seed{seed}.json")
        if not os.path.exists(path):
            print(f"MISSING: {path}")
            continue
        per_seed.append(json.load(open(path)))
    if not per_seed:
        continue
    alpha = per_seed[0]["alpha_basin"]
    n = len(per_seed)
    mean_neg_rtg = sum(d["mean_neg_rtg_frac"] for d in per_seed) / n
    mean_near_xstar = sum(d["mean_frac_rollout_near_xstar"] for d in per_seed) / n
    mean_n_improved = sum(d["incumbent_improved_count"] for d in per_seed) / n
    mean_final_regret = sum(d["final_regret"] for d in per_seed) / n
    rows.append((variant, alpha, n, mean_neg_rtg, mean_near_xstar, mean_n_improved, mean_final_regret))

print(f"\n{'variant':<16}{'alpha':<8}{'n_seeds':<9}{'mean_neg_rtg_frac':<20}"
      f"{'mean_frac_near_xstar':<22}{'mean_n_improved':<18}{'mean_final_regret':<18}")
for variant, alpha, n, neg_rtg, near_xstar, n_improved, final_regret in rows:
    print(f"{variant:<16}{alpha:<8}{n:<9}{neg_rtg:<20.4f}{near_xstar:<22.4f}"
          f"{n_improved:<18.2f}{final_regret:<18.4f}")

print("\nNote: zero_reward_frac (as literally named in the request) only exists under "
      "rollout_reward='improvement'; this sweep used the default 'current best' "
      "rollout_reward='mes_entropy', so mean_neg_rtg_frac (fraction of rollout steps "
      "with RTG<0) is reported in its place -- see _basin_width_worker.py docstring.")
