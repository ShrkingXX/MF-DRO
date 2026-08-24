"""
Summary table for mfdro_gp_refinement_test: variant | seed | n_improved |
final_regret | mean_dist_xDT | mean_dist_xrefined, plus the aggregate
CRITICAL CHECK and interpretation per the original spec.
"""
import json
import os
import numpy as np

CKPT_DIR = "results/mfdro_gp_refinement_test/checkpoints"
SEEDS = [42, 43, 44]

print(f"{'variant':<12}{'seed':<8}{'n_improved':<13}{'final_regret':<15}"
      f"{'mean_dist_xDT':<16}{'mean_dist_xrefined':<20}")
rows = {}
for variant in ["BASELINE", "REFINED"]:
    rows[variant] = []
    for seed in SEEDS:
        path = os.path.join(CKPT_DIR, f"{variant}__seed{seed}.json")
        if not os.path.exists(path):
            print(f"{variant:<12}{seed:<8} MISSING")
            continue
        d = json.load(open(path))
        rows[variant].append(d)
        dxdt = d["mean_dist_xDT"]
        dxref = d["mean_dist_xrefined"]
        print(f"{variant:<12}{seed:<8}{d['incumbent_improved_count']:<13}"
              f"{d['final_regret']:<15.4f}"
              f"{(dxdt if dxdt is not None else float('nan')):<16.4f}"
              f"{(dxref if dxref is not None else float('nan')):<20.4f}")

print()
mean_regret_baseline = np.mean([d["final_regret"] for d in rows["BASELINE"]])
mean_regret_refined = np.mean([d["final_regret"] for d in rows["REFINED"]])
mean_improved_baseline = np.mean([d["incumbent_improved_count"] for d in rows["BASELINE"]])
mean_improved_refined = np.mean([d["incumbent_improved_count"] for d in rows["REFINED"]])
print(f"BASELINE mean: final_regret={mean_regret_baseline:.4f}  n_improved={mean_improved_baseline:.2f}")
print(f"REFINED  mean: final_regret={mean_regret_refined:.4f}  n_improved={mean_improved_refined:.2f}")

dist_xdt_all = [d["mean_dist_xDT"] for d in rows["REFINED"] if d["mean_dist_xDT"] is not None]
dist_xref_all = [d["mean_dist_xrefined"] for d in rows["REFINED"] if d["mean_dist_xrefined"] is not None]
mean_dist_xdt = np.mean(dist_xdt_all) if dist_xdt_all else float("nan")
mean_dist_xref = np.mean(dist_xref_all) if dist_xref_all else float("nan")

print(f"\nREFINED aggregate: mean_dist_xDT={mean_dist_xdt:.4f}  mean_dist_xrefined={mean_dist_xref:.4f}")

print("\n--- INTERPRETATION ---")
refinement_helps_location = mean_dist_xref < mean_dist_xdt
n_improved_increases = mean_improved_refined > mean_improved_baseline
if refinement_helps_location and n_improved_increases:
    print("Refinement works: GP gradient ascent moves proposals closer to x*, "
          "and this translates to real incumbent improvements.")
elif refinement_helps_location and not n_improved_increases:
    print("Refinement helps location precision but n_improved did not increase -- "
          "still not reaching x*/beating the incumbent. Consider gp_refinement_steps=100 "
          "or running EI on multiple ensemble members and taking the best.")
else:
    print("mean_dist_xrefined >= mean_dist_xDT: EI landscape may be flat or gradient "
          "vanishing -- see per-seed ei_at_x_dt/ei_at_x_refined diagnostics in the "
          "worker logs (logs/mfdro_gp_refinement_test/REFINED__seed42.log) for whether "
          "EI is near zero everywhere (GP overconfident that no improvement exists).")
