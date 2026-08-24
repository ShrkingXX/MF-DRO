"""
Greedy-MES control for the basin-width sweep (mfdro_basin_width_sweep):
does a DT-free, no-training method built on the SAME KO-GP/MES machinery
show similarly dramatic basin-width sensitivity to MF-DRO, or is MF-DRO
disproportionately more sensitive? If Greedy-MES improves about as much as
MF-DRO did as alpha_basin shrinks, that's evidence the earlier finding
(real HF-query value-gap dropping 2.34/0.90/0.52 as alpha_basin went
1.0/0.3/0.1) is mostly a generic "narrower basin = harder search problem
for anyone" confound, not something specific to MF-DRO's own training/
freeze pathology. If MF-DRO's relative improvement is much larger, that
argues the basin-width effect is doing something MF-DRO-specific.

Same protocol as _basin_width_worker.py wherever it's shared: same 3
variants, same seeds, same initial_hf=36/initial_lf=60, same bo_iterations
cap (200 -- matches what the MF-DRO sweep actually ran to, since its
cost_budget=400 never bound). cost_budget=None (unlimited), matching
_stage2_v3_worker.py's own established Greedy-MES pattern -- iteration
count is the only stop condition, same as it effectively was for the
MF-DRO sweep.
"""
import sys
import os
import json

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VARIANT = sys.argv[1]        # Hartmann6D_w10 / Hartmann6D_w03 / Hartmann6D_w01
seed = int(sys.argv[2])

N_ITERS = 200
EXP_NAME = "mfdro_basin_width_sweep"

OUT_DIR = os.path.join("results", EXP_NAME, "checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"GreedyMES__{VARIANT}__seed{seed}.json")
tag = f"[GreedyMES {VARIANT} seed{seed}]"

if os.path.exists(out_path):
    print(f"{tag} SKIPPED (already exists)", flush=True)
    sys.exit(0)

from benchmarks import HARTMANN_WIDENED_OPTIMA, _HARTMANN_WIDENED_ALPHAS
from src.baselines.mf_baselines import MultiFidelityBenchmark, GreedyMFMESOptimizer

torch.set_default_dtype(torch.float64)
bench = MultiFidelityBenchmark(VARIANT)
true_opt_val, true_opt_x = HARTMANN_WIDENED_OPTIMA[VARIANT]
print(f"{tag} Starting (c_H={bench.c_H}, c_L={bench.c_L}, "
      f"true_optimal_value={true_opt_val:.4f}, true_optimal_x={np.round(true_opt_x, 4).tolist()})",
      flush=True)

opt = GreedyMFMESOptimizer(
    bench, n_initial_hf=36, n_initial_lf=60, seed=seed, cost_budget=None,
)
result = opt.run(bo_iterations=N_ITERS)

rc = result["regret_curve"]
n_improved = sum(1 for i in range(1, len(rc)) if rc[i] < rc[i - 1] - 1e-12)
distinct = len(set(f"{r:.6f}" for r in rc))

# Real HF-query value gap (true_opt - y_t for ell_t==1 queries), same metric
# used for MF-DRO's own real-query quality check -- the direct point of
# comparison for this control.
ell_t = result["fidelity_trace"]
y_t = result["y_t_trace"]
hf_gaps = [true_opt_val - y for y, e in zip(y_t, ell_t) if e == 1]

out = dict(
    result, variant=VARIANT, seed=seed, alpha_basin=_HARTMANN_WIDENED_ALPHAS[VARIANT],
    true_optimal_value=true_opt_val, true_optimal_x=true_opt_x,
    incumbent_improved_count=n_improved, distinct_regret_values=distinct,
    final_regret=rc[-1] if rc else None,
    n_hf_queries=len(hf_gaps),
    mean_hf_gap=float(np.mean(hf_gaps)) if hf_gaps else None,
    median_hf_gap=float(np.median(hf_gaps)) if hf_gaps else None,
)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"{tag} DONE final_regret={rc[-1] if rc else float('nan'):.4f} n_iters={len(rc)} "
      f"n_improved={n_improved} distinct={distinct}/{len(rc)} "
      f"lf_fraction={result['lf_fraction']:.3f} n_hf_queries={len(hf_gaps)} "
      f"mean_hf_gap={out['mean_hf_gap']}", flush=True)
