"""
UCB variant of the semi-amortized GP-refinement ablation on standard
Hartmann_6D: same warm-start gradient ascent as mfdro_gp_refinement_test,
but replacing EI with UCB (mu + beta*sigma, beta=2.0 fixed) in
DirectMFRegretOptimization._refine_proposal (src/policy/mf_dro.py),
selected via config.gp_refinement_acquisition="ucb". Motivation: EI's
gradient vanished almost everywhere once the GP's HF posterior got
confident (median EI collapsed to ~0.0 by the back half of the EI test's
380-iteration runs, and x_refined became bit-for-bit identical to x_DT in
every seed's last 20 iterations) -- UCB's sigma term should stay
gradient-informative even when EI's improvement term does not, since sigma
alone (not (mu-best)) drives the exploration term.

Two variants, both cost_budget=400, bo_iterations cap=450 (same rationale
as the EI test: near-100% LF query selection means iterations, not cost,
was the practical constraint at 200; 450 gives headroom to reach the
cost budget), "current best" config otherwise (lognormal prior [default],
diverse init [default], raw KO state [default], num_epochs=10,
rollout_length=8):
    BASELINE:     use_gp_refinement=False
    REFINED_UCB:  use_gp_refinement=True, gp_refinement_acquisition="ucb",
                  gp_refinement_steps=30, gp_refinement_lr=0.05,
                  gp_refinement_ucb_beta=2.0
"""
import sys
import os
import json

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VARIANT = sys.argv[1]        # BASELINE | REFINED_UCB
seed = int(sys.argv[2])

BENCHMARK = "Hartmann_6D"
N_ITERS_CAP = 450
EXP_NAME = "mfdro_ucb_refinement_test"
COST_BUDGET = 400.0
X_STAR = [0.2017, 0.1500, 0.4769, 0.2753, 0.3116, 0.6573]

OUT_DIR = os.path.join("results", EXP_NAME, "checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"{VARIANT}__seed{seed}.json")
tag = f"[{VARIANT} seed{seed}]"

if os.path.exists(out_path):
    print(f"{tag} SKIPPED (already exists)", flush=True)
    sys.exit(0)

from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization

torch.set_default_dtype(torch.float64)
hf_spec = get_benchmark(BENCHMARK + "_HF")
lf_spec = get_benchmark(BENCHMARK + "_LF")
f_hf = hf_spec["make_objective"]()
f_lf = lf_spec["make_objective"]()
bounds = torch.tensor([hf_spec["domain_min"], hf_spec["domain_max"]], dtype=torch.float64)

print(f"{tag} Starting (c_H={hf_spec['cost']}, c_L={lf_spec['cost']}, "
      f"use_gp_refinement={VARIANT == 'REFINED_UCB'}, acquisition=ucb)", flush=True)

torch.manual_seed(seed)
np.random.seed(seed)
config = _build_mf_dro_config(
    EXP_NAME, BENCHMARK, VARIANT, seed,
    bo_iterations=N_ITERS_CAP, num_epochs=10,
    minimum_hf_fraction=0.25, real_hf_warmup=2,
    cost_budget=COST_BUDGET, initial_hf=36, initial_lf=60,
    dkl_threshold=9999, bes_delta=0.0, rollout_length=8,
    known_optimal_x=X_STAR,
)
config.seed = seed
config.use_gp_refinement = (VARIANT == "REFINED_UCB")
config.gp_refinement_acquisition = "ucb"
config.gp_refinement_steps = 30
config.gp_refinement_lr = 0.05
config.gp_refinement_ucb_beta = 2.0
mf = DirectMFRegretOptimization(config, f_hf, f_lf, bounds)

result = mf.run()

rc = result["hf_regret_curve"]
n_improved = sum(1 for i in range(1, len(rc)) if rc[i] < rc[i - 1] - 1e-12)
distinct = len(set(f"{r:.6f}" for r in rc))
refine_log = result.get("gp_refinement_log", [])

closer_flags = []
if VARIANT == "REFINED_UCB":
    print(f"{tag} --- per-iteration report ---", flush=True)
    best_so_far = None
    for i, r in enumerate(rc):
        improved = best_so_far is None or r < best_so_far - 1e-12
        if improved:
            best_so_far = r
        entry = refine_log[i] if i < len(refine_log) else {}
        x_dt_dist = entry.get("x_dt_dist", float("nan"))
        x_ref_dist = entry.get("x_refined_dist", float("nan"))
        ucb_dt = entry.get("acq_at_x_dt", float("nan"))
        ucb_ref = entry.get("acq_at_x_refined", float("nan"))
        closer = (x_ref_dist < x_dt_dist) if entry.get("x_dt_dist") is not None else None
        closer_flags.append(closer)
        print(f"{tag} iter={i:3d} x_DT_dist_to_xstar={x_dt_dist:.4f} "
              f"x_refined_dist_to_xstar={x_ref_dist:.4f} "
              f"ucb_at_x_DT={ucb_dt:.4f} ucb_at_x_refined={ucb_ref:.4f} "
              f"frac_closer={closer} regret={r:.4f} "
              f"incumbent_improved={int(improved)}", flush=True)

    valid = [c for c in closer_flags if c is not None]
    n = len(valid)
    first20 = valid[:20]
    last20 = valid[-20:]
    frac_closer_first20 = (sum(first20) / len(first20)) if first20 else float("nan")
    frac_closer_last20 = (sum(last20) / len(last20)) if last20 else float("nan")
    frac_closer_overall = (sum(valid) / n) if n else float("nan")
    print(f"{tag} CRITICAL CHECK: frac_closer overall={frac_closer_overall:.1%} "
          f"first20={frac_closer_first20:.1%} last20={frac_closer_last20:.1%}", flush=True)
    if frac_closer_last20 == 0.0:
        sigmas_last20 = [refine_log[i].get("sigma_at_x_dt") for i in range(max(0, len(refine_log) - 20), len(refine_log))]
        sigmas_last20 = [s for s in sigmas_last20 if s is not None]
        print(f"{tag} WARNING: frac_closer collapsed to 0 in the last 20 iterations -- "
              f"UCB gradients may be vanishing. sigma_at_x_dt (last 20): "
              f"mean={np.mean(sigmas_last20) if sigmas_last20 else float('nan'):.6f} "
              f"min={np.min(sigmas_last20) if sigmas_last20 else float('nan'):.6f} "
              f"max={np.max(sigmas_last20) if sigmas_last20 else float('nan'):.6f}", flush=True)
    elif frac_closer_overall > 0.3:
        print(f"{tag} UCB refinement stays active throughout the run "
              f"(frac_closer={frac_closer_overall:.1%} > 30%).", flush=True)

out = dict(
    result, variant=VARIANT, seed=seed,
    incumbent_improved_count=n_improved, distinct_regret_values=distinct,
    final_regret=rc[-1] if rc else None,
    frac_closer_first20=(sum(closer_flags[:20]) / len([c for c in closer_flags[:20] if c is not None])
                          if [c for c in closer_flags[:20] if c is not None] else None) if closer_flags else None,
    frac_closer_last20=(sum(c for c in closer_flags[-20:] if c is not None) / len([c for c in closer_flags[-20:] if c is not None])
                         if [c for c in closer_flags[-20:] if c is not None] else None) if closer_flags else None,
)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"{tag} DONE final_regret={rc[-1] if rc else float('nan'):.4f} n_iters={len(rc)} "
      f"n_improved={n_improved} distinct={distinct}/{len(rc)} "
      f"lf_fraction={result['lf_fraction']:.3f} final_cost={result['cost_curve'][-1]:.1f} "
      f"frac_closer_first20={out['frac_closer_first20']} frac_closer_last20={out['frac_closer_last20']}",
      flush=True)
