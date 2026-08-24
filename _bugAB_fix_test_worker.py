"""
Semi-amortized GP-refinement ablation on standard Hartmann_6D: after the DT
proposes x_DT, run gradient ascent on EI (of the KO ensemble's first
member's HF posterior) starting from x_DT to get x_refined, and query
x_refined instead. Fidelity (ell_t) still comes from the DT untouched --
only the queried LOCATION changes. See DirectMFRegretOptimization.
_refine_proposal (src/policy/mf_dro.py) for the actual implementation and
a correctness note: the originally-specified implementation called
ko.hf_posterior(...), which wraps its body in torch.no_grad() and silently
breaks the gradient back to x (verified directly: ei.backward() raised
"does not require grad and does not have a grad_fn"). _refine_proposal
reimplements the same mu_H/var_H combination without that wrapper instead.

Two variants, both cost_budget=400, "current best" config otherwise
(lognormal prior [default], diverse init [default], raw KO state [default],
num_epochs=10, rollout_length=8):
    BASELINE: use_gp_refinement=False
    REFINED:  use_gp_refinement=True, gp_refinement_steps=30, gp_refinement_lr=0.05

bo_iterations cap set to 450 (not 200): the basin-width sweep found
lf_fraction~0.985-0.99 means iterations, not cost, is usually the binding
constraint -- 200 iterations there only reached ~cost=220 of a cost_budget=
400. 450 gives real headroom to actually hit the 400 cost budget this time.
"""
import sys
import os
import json

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VARIANT = sys.argv[1]        # BASELINE | REFINED
seed = int(sys.argv[2])

BENCHMARK = "Hartmann_6D"
N_ITERS_CAP = 450
EXP_NAME = "mfdro_bugAB_fix_test"
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
      f"use_gp_refinement={VARIANT == 'REFINED'})", flush=True)

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
config.use_gp_refinement = (VARIANT == "REFINED")
config.gp_refinement_steps = 30
config.gp_refinement_lr = 0.05
mf = DirectMFRegretOptimization(config, f_hf, f_lf, bounds)

result = mf.run()

rc = result["hf_regret_curve"]
n_improved = sum(1 for i in range(1, len(rc)) if rc[i] < rc[i - 1] - 1e-12)
distinct = len(set(f"{r:.6f}" for r in rc))
qd = result.get("query_dist_to_xstar_per_iter", [])
refine_log = result.get("gp_refinement_log", [])

if VARIANT == "REFINED":
    print(f"{tag} --- per-iteration report ---", flush=True)
    best_so_far = None
    n_closer = 0
    n_with_dist = 0
    for i, r in enumerate(rc):
        improved = best_so_far is None or r < best_so_far - 1e-12
        if improved:
            best_so_far = r
        entry = refine_log[i] if i < len(refine_log) else {}
        x_dt_dist = entry.get("x_dt_dist", float("nan"))
        x_ref_dist = entry.get("x_refined_dist", float("nan"))
        moved_closer = x_ref_dist < x_dt_dist if entry else None
        if entry.get("x_dt_dist") is not None:
            n_with_dist += 1
            if moved_closer:
                n_closer += 1
        print(f"{tag} iter={i:3d} x_DT_dist_to_xstar={x_dt_dist:.4f} "
              f"x_refined_dist_to_xstar={x_ref_dist:.4f} "
              f"improvement={moved_closer} regret={r:.4f} "
              f"incumbent_improved={int(improved)}", flush=True)

    # CRITICAL CHECK, per spec.
    frac_closer = n_closer / n_with_dist if n_with_dist else float("nan")
    print(f"{tag} CRITICAL CHECK: refined closer to x* than DT proposal on "
          f"{n_closer}/{n_with_dist} iterations ({frac_closer:.1%})", flush=True)
    if n_with_dist and frac_closer < 0.5:
        print(f"{tag} WARNING: x_refined_dist_to_xstar is NOT consistently "
              f"smaller than x_DT_dist_to_xstar -- EI landscape may be flat "
              f"or gradients vanishing. See per-iteration ei_at_x_dt/"
              f"ei_at_x_refined below.", flush=True)

    if seed == 42:
        print(f"{tag} --- EI diagnostic (seed=42 REFINED only) ---", flush=True)
        n_violations = 0
        for i, entry in enumerate(refine_log):
            ei_dt = entry.get("ei_at_x_dt", float("nan"))
            ei_ref = entry.get("ei_at_x_refined", float("nan"))
            ok = ei_ref >= ei_dt
            if not ok:
                n_violations += 1
            print(f"{tag} iter={i:3d} ei_at_x_DT={ei_dt:.6f} "
                  f"ei_at_x_refined={ei_ref:.6f} monotone_ok={ok}", flush=True)
        print(f"{tag} EI monotonicity violations: {n_violations}/{len(refine_log)}"
              f" (should be 0 -- Adam ascent should never leave EI(x_refined) < EI(x_DT))",
              flush=True)

out = dict(
    result, variant=VARIANT, seed=seed,
    incumbent_improved_count=n_improved, distinct_regret_values=distinct,
    final_regret=rc[-1] if rc else None,
    mean_dist_xDT=float(np.mean([e["x_dt_dist"] for e in refine_log if "x_dt_dist" in e])) if refine_log else None,
    mean_dist_xrefined=float(np.mean([e["x_refined_dist"] for e in refine_log if "x_refined_dist" in e])) if refine_log else None,
)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"{tag} DONE final_regret={rc[-1] if rc else float('nan'):.4f} n_iters={len(rc)} "
      f"n_improved={n_improved} distinct={distinct}/{len(rc)} "
      f"lf_fraction={result['lf_fraction']:.3f} final_cost={result['cost_curve'][-1]:.1f} "
      f"mean_dist_xDT={out['mean_dist_xDT']} mean_dist_xrefined={out['mean_dist_xrefined']}",
      flush=True)
