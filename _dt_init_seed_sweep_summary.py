"""
Summary table for mfdro_dt_init_seed_sweep: dt_init_seed | n_improved |
final_regret | frozen, plus the fraction that freeze and the interpretation
per the original spec.
"""
import json
import os

CKPT_DIR = "results/mfdro_dt_init_seed_sweep/checkpoints"
OUTER_SEED = 44

print(f"{'dt_init_seed':<15}{'n_improved':<13}{'final_regret':<15}{'frozen':<8}")
rows = []
for dt_init_seed in range(10):
    path = os.path.join(CKPT_DIR, f"seed{OUTER_SEED}__dtinit{dt_init_seed}.json")
    if not os.path.exists(path):
        print(f"{dt_init_seed:<15} MISSING")
        continue
    d = json.load(open(path))
    rows.append(d)
    print(f"{dt_init_seed:<15}{d['incumbent_improved_count']:<13}{d['final_regret']:<15.4f}{str(d['frozen']):<8}")

n_frozen = sum(1 for d in rows if d["frozen"])
n_total = len(rows)
distinct_regrets = sorted(set(round(d["final_regret"], 4) for d in rows))

print(f"\n{n_frozen}/{n_total} dt_init_seeds froze (n_improved==0)")
print(f"Distinct final_regret values across all {n_total} runs: {distinct_regrets}")

print("\n--- INTERPRETATION ---")
if 0 < n_frozen < n_total:
    print(f"Initialization IS a controlling factor: {n_frozen}/{n_total} seeds landed in the frozen minimum, "
          f"{n_total - n_frozen}/{n_total} escaped it -- everything else (benchmark, outer seed/LHS init, "
          f"KO ensemble diversity, training data) was held fixed, so DT weight init alone determines the outcome "
          f"for this fraction of runs.")
elif n_frozen == n_total:
    print(f"All {n_total} dt_init_seeds froze -- initialization is NOT the sole controlling factor for this "
          f"(benchmark, outer_seed) pair. Something else shared across all inits (LHS initial sample itself, "
          f"the fixed KO ensemble diversity draws, or the training data/RTG construction) is responsible.")
else:
    print(f"All {n_total} dt_init_seeds escaped the frozen minimum -- the original freeze for outer_seed={OUTER_SEED} "
          f"was specific to whatever DT init config.seed+2000 happened to produce, not a structural property of "
          f"this (benchmark, outer_seed) pair.")
