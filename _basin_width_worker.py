"""
Basin-width sweep worker: does MF-DRO's incumbent-freeze pathology respond
to landscape/basin width specifically (independent of the cost-ratio/
LF-correlation mechanism investigated separately -- see
mfdro_hartmann_cost_ratio)? Runs MF-DRO on HartmannWidened6D at three
alpha_basin values (1.0=standard Hartmann-6, 0.3, 0.1=widest), using REAL
cost_budget=400 termination (NOT the fixed-100-iteration protocol used
elsewhere this session -- explicit choice for this sweep).

known_optimal_x is set to each variant's EMPIRICALLY-determined true
optimum (benchmarks.HARTMANN_WIDENED_OPTIMA), not the fixed standard
Hartmann-6 x*=[0.2017,...] -- see benchmarks.py's registration comment and
_hartmann_widened_sanity_check.py: scaling the whole A matrix by one scalar
does NOT preserve the mixture's argmax, so x* genuinely moves as alpha_basin
shrinks (dist from standard x*: 0.0/0.18/0.55 for alpha=1.0/0.3/0.1). Using
the true per-alpha optimum keeps both the regret computation and the
frac_rollout_near_xstar diagnostic meaningful for every variant.

zero_reward_frac (as literally named) only exists under
rollout_reward="improvement" (see simulate_mf_trajectory) -- "current best"
config uses the default rollout_reward="mes_entropy" (matching every other
worker this session), under which that field is None. neg_rtg_frac_per_iter
is the mes_entropy-native analogue (fraction of rollout steps with RTG<0)
and is reported in its place, clearly labeled -- not fabricated as
zero_reward_frac.
"""
import sys
import os
import json

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VARIANT = sys.argv[1]        # Hartmann6D_w10 / Hartmann6D_w03 / Hartmann6D_w01
seed = int(sys.argv[2])

N_ITERS_CAP = 200             # generous cap -- cost_budget=400 is the real stopping rule
EXP_NAME = "mfdro_basin_width_sweep"
COST_BUDGET = 400.0

OUT_DIR = os.path.join("results", EXP_NAME, "checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"{VARIANT}__seed{seed}.json")
tag = f"[{VARIANT} seed{seed}]"

if os.path.exists(out_path):
    print(f"{tag} SKIPPED (already exists)", flush=True)
    sys.exit(0)

from benchmarks import get_benchmark, HARTMANN_WIDENED_OPTIMA, _HARTMANN_WIDENED_ALPHAS
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization

torch.set_default_dtype(torch.float64)
hf_spec = get_benchmark(VARIANT + "_HF")
lf_spec = get_benchmark(VARIANT + "_LF")
f_hf = hf_spec["make_objective"]()
f_lf = lf_spec["make_objective"]()
bounds = torch.tensor([hf_spec["domain_min"], hf_spec["domain_max"]], dtype=torch.float64)

true_opt_val, true_opt_x = HARTMANN_WIDENED_OPTIMA[VARIANT]
print(f"{tag} Starting (c_H={hf_spec['cost']}, c_L={lf_spec['cost']}, "
      f"true_optimal_value={true_opt_val:.4f}, true_optimal_x={np.round(true_opt_x, 4).tolist()})",
      flush=True)

torch.manual_seed(seed)
np.random.seed(seed)
config = _build_mf_dro_config(
    EXP_NAME, VARIANT, VARIANT, seed,
    bo_iterations=N_ITERS_CAP, num_epochs=10,
    minimum_hf_fraction=0.25, real_hf_warmup=2,
    cost_budget=COST_BUDGET, initial_hf=36, initial_lf=60,
    dkl_threshold=9999, bes_delta=0.0, rollout_length=8,
    known_optimal_x=true_opt_x,
)
config.seed = seed
mf = DirectMFRegretOptimization(config, f_hf, f_lf, bounds)
# Drives the [ROLLOUT-DIAG] block's frac_rollout_near_xstar print/log (see
# mf_dro.py's diag_frac_rollout_near_xstar_per_iter) -- raw-scale tensor,
# same convention as _rollout_diag_worker.py and friends. Domain is [0,1]^6
# so raw == normalized here.
mf._diag_xstar = torch.tensor(true_opt_x, dtype=torch.float64)

result = mf.run()

rc = result["hf_regret_curve"]
n_improved = sum(1 for i in range(1, len(rc)) if rc[i] < rc[i - 1] - 1e-12)
distinct = len(set(f"{r:.6f}" for r in rc))

# --- Per-iteration report (post-hoc, from the arrays run() already
# accumulated -- regret/incumbent_improved/neg_rtg_frac aren't otherwise
# printed live per-iteration; frac_rollout_near_xstar IS also already
# printed live via [ROLLOUT-DIAG] during the run itself). ---
neg_rtg = result.get("neg_rtg_frac_per_iter", [])
frac_near = result.get("diag_frac_rollout_near_xstar_per_iter", [])
print(f"{tag} --- per-iteration report ---", flush=True)
best_so_far = None
for i, r in enumerate(rc):
    improved = best_so_far is None or r < best_so_far - 1e-12
    if improved:
        best_so_far = r
    nr = neg_rtg[i] if i < len(neg_rtg) else float("nan")
    fn = frac_near[i] if i < len(frac_near) else float("nan")
    print(f"{tag} iter={i:3d} regret={r:.4f} incumbent_improved={int(improved)} "
          f"zero_reward_frac=N/A(mes_entropy; neg_rtg_frac={nr:.3f}) "
          f"frac_rollout_near_xstar={fn:.3f}", flush=True)

out = dict(
    result, variant=VARIANT, seed=seed, alpha_basin=_HARTMANN_WIDENED_ALPHAS[VARIANT],
    true_optimal_value=true_opt_val, true_optimal_x=true_opt_x,
    incumbent_improved_count=n_improved, distinct_regret_values=distinct,
    final_regret=rc[-1] if rc else None,
    mean_neg_rtg_frac=float(np.mean(neg_rtg)) if neg_rtg else None,
    mean_frac_rollout_near_xstar=float(np.mean(frac_near)) if frac_near else None,
)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"{tag} DONE final_regret={rc[-1] if rc else float('nan'):.4f} n_iters={len(rc)} "
      f"n_improved={n_improved} distinct={distinct}/{len(rc)} "
      f"lf_fraction={result['lf_fraction']:.3f} "
      f"mean_neg_rtg_frac={out['mean_neg_rtg_frac']} "
      f"mean_frac_near_xstar={out['mean_frac_rollout_near_xstar']}", flush=True)
