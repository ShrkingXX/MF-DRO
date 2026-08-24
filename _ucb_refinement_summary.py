"""
Summary table for mfdro_ucb_refinement_test: variant | seed | n_improved |
final_regret | frac_closer_first20 | frac_closer_last20, plus a comparison
against the EI refinement test's results (mfdro_gp_refinement_test) and the
INTERPRET decision tree from the original spec.
"""
import json
import os
import numpy as np

UCB_DIR = "results/mfdro_ucb_refinement_test/checkpoints"
EI_DIR = "results/mfdro_gp_refinement_test/checkpoints"
SEEDS = [42, 43, 44]

print(f"{'variant':<14}{'seed':<8}{'n_improved':<13}{'final_regret':<15}"
      f"{'frac_closer_first20':<22}{'frac_closer_last20':<20}")
rows = {}
for variant in ["BASELINE", "REFINED_UCB"]:
    rows[variant] = []
    for seed in SEEDS:
        path = os.path.join(UCB_DIR, f"{variant}__seed{seed}.json")
        if not os.path.exists(path):
            print(f"{variant:<14}{seed:<8} MISSING")
            continue
        d = json.load(open(path))
        rows[variant].append(d)
        f20 = d.get("frac_closer_first20")
        l20 = d.get("frac_closer_last20")
        print(f"{variant:<14}{seed:<8}{d['incumbent_improved_count']:<13}"
              f"{d['final_regret']:<15.4f}"
              f"{(f20 if f20 is not None else float('nan')):<22.4f}"
              f"{(l20 if l20 is not None else float('nan')):<20.4f}")

print()
mean_regret_baseline = np.mean([d["final_regret"] for d in rows["BASELINE"]])
mean_regret_ucb = np.mean([d["final_regret"] for d in rows["REFINED_UCB"]])
mean_improved_baseline = np.mean([d["incumbent_improved_count"] for d in rows["BASELINE"]])
mean_improved_ucb = np.mean([d["incumbent_improved_count"] for d in rows["REFINED_UCB"]])
mean_last20 = np.mean([d["frac_closer_last20"] for d in rows["REFINED_UCB"] if d.get("frac_closer_last20") is not None])
mean_first20 = np.mean([d["frac_closer_first20"] for d in rows["REFINED_UCB"] if d.get("frac_closer_first20") is not None])

print(f"BASELINE    mean: final_regret={mean_regret_baseline:.4f}  n_improved={mean_improved_baseline:.2f}")
print(f"REFINED_UCB mean: final_regret={mean_regret_ucb:.4f}  n_improved={mean_improved_ucb:.2f}")
print(f"REFINED_UCB mean frac_closer: first20={mean_first20:.1%}  last20={mean_last20:.1%}")

# Comparison against the EI test's REFINED results, if present.
ei_regrets = []
for seed in SEEDS:
    path = os.path.join(EI_DIR, f"REFINED__seed{seed}.json")
    if os.path.exists(path):
        ei_regrets.append(json.load(open(path))["final_regret"])
if ei_regrets:
    print(f"\nFor reference -- REFINED (EI) mean final_regret: {np.mean(ei_regrets):.4f}")
    print(f"REFINED_UCB vs REFINED_EI: {'better' if mean_regret_ucb < np.mean(ei_regrets) else 'worse or equal'} "
          f"({mean_regret_ucb:.4f} vs {np.mean(ei_regrets):.4f})")

print("\n--- INTERPRETATION ---")
stays_active = mean_last20 > 0.2
improved_more = mean_improved_ucb > mean_improved_baseline
better_than_ei = ei_regrets and mean_regret_ucb < np.mean(ei_regrets)

if stays_active and improved_more and better_than_ei:
    print("UCB refinement is the correct fix -> ready to run Stage 2 with use_gp_refinement=True.")
elif not stays_active:
    print("frac_closer_last20 still collapsed to ~0: GP sigma is likely collapsing (over-confident model). "
          "Consider beta=5.0 or pure posterior-mean-argmax refinement (no UCB exploration term). "
          "See sigma_at_x_dt values printed in the per-seed logs.")
elif not improved_more:
    print("n_improved unchanged vs BASELINE but final_regret still improves: same early-window-only mechanism "
          "as EI. Consider gp_refinement_steps=100 to strengthen the early effect.")
else:
    print("Mixed result -- see per-seed detail above before deciding.")
