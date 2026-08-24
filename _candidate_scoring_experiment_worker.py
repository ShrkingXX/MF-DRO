"""
Head-to-head: MF-DRO REGRESSION vs MF-DRO SCORING (use_candidate_scoring)
vs Greedy-MES reference. Hartmann_6D, seeds 42/43/44, cost_budget=240.
Config matches Stage 2 v3 (dkl_threshold=9999, initial_hf=initial_lf=30,
rollout_length=8, bes_delta=0.0, minimum_hf_fraction=0.25, real_hf_warmup=2,
M=10, rollouts_per_model=7, num_epochs=100). noise_lb not passed anywhere
(not a valid config_override, and KennedyOHaganGP's own class default is
already 1e-2). K_cands not passed to _build_mf_dro_config either -- it's
not a config field there, it's simulate_mf_trajectory's own parameter,
threaded through _generate_rollout_batch with its default of 20, which
already matches what this experiment wants -- no code change needed.

Same MF-DRO/Greedy-MES construction pattern as _stage2_mini_worker.py
(direct construction, not the run_mf_single_seed/opt.run() wrappers) so
mean_ls_lf/mean_ls_delta/incumbent_improved_count can be read off the
fitted GP afterward.
"""
import sys
import os
import json

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

method = sys.argv[1]   # REGRESSION | SCORING | Greedy-MES
seed = int(sys.argv[2])

BENCHMARK = "Hartmann_6D"
COST_BUDGET = 240
INITIAL_HF = 30
INITIAL_LF = 30
EXP_NAME = "mfdro_candidate_scoring_test"

OUT_DIR = os.path.join("results", EXP_NAME, "checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"{method}__{BENCHMARK}__seed{seed}.json")
tag = f"[{method} {BENCHMARK} seed{seed}]"

if os.path.exists(out_path):
    print(f"{tag} SKIPPED (already exists)", flush=True)
    sys.exit(0)


def regret_at_cost(cost_curve, regret_curve, c_ref):
    idx = None
    for i, c in enumerate(cost_curve):
        if c <= c_ref:
            idx = i
        else:
            break
    return regret_curve[idx] if idx is not None else (regret_curve[0] if regret_curve else float('nan'))


def incumbent_improved_count(regret_curve):
    return sum(1 for i in range(1, len(regret_curve)) if regret_curve[i] < regret_curve[i - 1] - 1e-12)


print(f"{tag} Starting (cost_budget={COST_BUDGET}, initial_hf={INITIAL_HF}, initial_lf={INITIAL_LF})", flush=True)

if method in ("REGRESSION", "SCORING"):
    from benchmarks import get_benchmark
    from dro_runner import _build_mf_dro_config
    from src.policy.mf_dro import DirectMFRegretOptimization
    from src.models.ko_gp import DeepKernel

    torch.set_default_dtype(torch.float64)
    hf_spec = get_benchmark(BENCHMARK + "_HF")
    lf_spec = get_benchmark(BENCHMARK + "_LF")
    f_hf = hf_spec["make_objective"]()
    f_lf = lf_spec["make_objective"]()
    bounds = torch.tensor([hf_spec["domain_min"], hf_spec["domain_max"]], dtype=torch.float64)

    use_cs = (method == "SCORING")
    torch.manual_seed(seed)
    np.random.seed(seed)
    config = _build_mf_dro_config(
        EXP_NAME, BENCHMARK, method, seed,
        bo_iterations=500, num_epochs=100,
        minimum_hf_fraction=0.25, real_hf_warmup=2,
        cost_budget=COST_BUDGET, initial_hf=INITIAL_HF, initial_lf=INITIAL_LF,
        dkl_threshold=9999, use_candidate_scoring=use_cs,
    )
    config.seed = seed
    mf = DirectMFRegretOptimization(config, f_hf, f_lf, bounds)
    result = mf.run()

    def _mean_lengthscale(covar_module):
        scale_kernel = covar_module.base_kernel if isinstance(covar_module, DeepKernel) else covar_module
        return scale_kernel.base_kernel.lengthscale.detach().mean().item()

    ko0 = mf.ko_ensemble[0]
    mean_ls_lf = _mean_lengthscale(ko0.gp_lf.covar_module)
    mean_ls_delta = _mean_lengthscale(ko0.gp_delta.covar_module)

    rc, cc = result['hf_regret_curve'], result['cost_curve']
    best_hf = max(mf.data_hf_y)
    regret80 = regret_at_cost(cc, rc, 80)
    regret240 = regret_at_cost(cc, rc, 240)
    lf_frac = result['lf_fraction']
    n_improved = incumbent_improved_count(rc)

    out = dict(result, mean_ls_lf=mean_ls_lf, mean_ls_delta=mean_ls_delta,
               best_hf=best_hf, regret_at_80=regret80, regret_at_240=regret240,
               incumbent_improved_count=n_improved)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"{tag} DONE | best_hf={best_hf:.4f} | regret@240={regret240:.4f} | "
          f"regret@80={regret80:.4f} | lf_frac={lf_frac:.3f} | "
          f"mean_ls_lf={mean_ls_lf:.4f} | mean_ls_delta={mean_ls_delta:.4f} | "
          f"incumbent_improved_count={n_improved}", flush=True)

elif method == "Greedy-MES":
    from src.baselines.mf_baselines import MultiFidelityBenchmark, GreedyMFMESOptimizer

    bench = MultiFidelityBenchmark(BENCHMARK)
    opt = GreedyMFMESOptimizer(bench, n_initial_hf=INITIAL_HF, n_initial_lf=INITIAL_LF,
                                seed=seed, cost_budget=COST_BUDGET)
    result = opt.run(bo_iterations=500)

    ko0 = opt.ko_ensemble[0]
    mean_ls_lf = ko0.gp_lf.covar_module.base_kernel.lengthscale.detach().mean().item()
    mean_ls_delta = ko0.gp_delta.covar_module.base_kernel.lengthscale.detach().mean().item()

    rc, cc = result['regret_curve'], result['cost_curve']
    best_hf = max(opt.data_hf_y)
    regret80 = regret_at_cost(cc, rc, 80)
    regret240 = regret_at_cost(cc, rc, 240)
    lf_frac = result['lf_fraction']
    n_improved = incumbent_improved_count(rc)

    out = dict(result, mean_ls_lf=mean_ls_lf, mean_ls_delta=mean_ls_delta,
               best_hf=best_hf, regret_at_80=regret80, regret_at_240=regret240,
               incumbent_improved_count=n_improved)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"{tag} DONE | best_hf={best_hf:.4f} | regret@240={regret240:.4f} | "
          f"regret@80={regret80:.4f} | lf_frac={lf_frac:.3f} | "
          f"mean_ls_lf={mean_ls_lf:.4f} | mean_ls_delta={mean_ls_delta:.4f} | "
          f"incumbent_improved_count={n_improved}", flush=True)

else:
    raise ValueError(f"unknown method {method!r}")
