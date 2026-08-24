"""
Mini Stage 2 pre-check (mfdro_stage2_v3_mini): Hartmann_6D only, seeds 42/43,
methods MF-DRO / Greedy-MES / SF-DRO, cost_budget=80. Purpose: catch
pipeline problems before committing to the multi-hour full Stage 2 v3 run.

Deliberately does NOT call dro_runner.run_mf_single_seed for MF-DRO (and
does NOT call GreedyMFMESOptimizer through any wrapper) -- both are
constructed directly here so this script can inspect the fitted
ko_ensemble[0] afterward for mean_ls_lf/mean_ls_delta, which
_build_result()/GreedyMFBase.run()'s return dicts don't track. Otherwise
identical to what run_mf_single_seed / a direct GreedyMFMESOptimizer.run()
call would do.

noise_lb is NOT passed as a config override anywhere -- _build_mf_dro_config
has no such parameter (confirmed: would raise TypeError). KennedyOHaganGP's
own class default is already 1e-2 (changed earlier this session), so not
passing it is the correct way to get noise_lb=1e-2, not a gap.
"""
import sys
import os
import json

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

method = sys.argv[1]   # MF-DRO | Greedy-MES | SF-DRO
seed = int(sys.argv[2])

BENCHMARK = "Hartmann_6D"
COST_BUDGET = 80
INITIAL_HF = 30
INITIAL_LF = 30
EXP_NAME = "mfdro_stage2_v3_mini"

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
    # regret strictly decreasing = incumbent improved that step.
    return sum(1 for i in range(1, len(regret_curve)) if regret_curve[i] < regret_curve[i - 1] - 1e-12)


print(f"{tag} Starting (cost_budget={COST_BUDGET}, initial_hf={INITIAL_HF}, initial_lf={INITIAL_LF})", flush=True)

if method == "MF-DRO":
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

    torch.manual_seed(seed)
    np.random.seed(seed)
    config = _build_mf_dro_config(
        EXP_NAME, BENCHMARK, "MF-DRO", seed,
        bo_iterations=500, num_epochs=100,
        minimum_hf_fraction=0.25, real_hf_warmup=2,
        cost_budget=COST_BUDGET, initial_hf=INITIAL_HF, initial_lf=INITIAL_LF,
        dkl_threshold=9999,  # per Stage 2 v3 spec: DKL disabled. Also
        # required to avoid a crash: _build_mf_dro_config's own default is
        # 30, and initial_hf=30 crosses it immediately -- DKL would
        # silently activate, and DeepKernel.base_kernel is a ScaleKernel
        # (no .lengthscale of its own, one more level of .base_kernel than
        # plain RBF), which the mean_ls_lf/delta extraction below doesn't
        # handle. Fixed twice over: explicit dkl_threshold here, AND
        # DKL-aware extraction below (matching mf_dro.py's _ko_hp_features).
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
    lf_frac = result['lf_fraction']
    n_improved = incumbent_improved_count(rc)

    out = dict(result, mean_ls_lf=mean_ls_lf, mean_ls_delta=mean_ls_delta,
               best_hf=best_hf, regret_at_80=regret80,
               incumbent_improved_count=n_improved)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"{tag} DONE | best_hf={best_hf:.4f} | regret@80={regret80:.4f} | "
          f"lf_frac={lf_frac:.3f} | mean_ls_lf={mean_ls_lf:.4f} | "
          f"mean_ls_delta={mean_ls_delta:.4f} | incumbent_improved_count={n_improved}",
          flush=True)

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
    lf_frac = result['lf_fraction']
    n_improved = incumbent_improved_count(rc)

    out = dict(result, mean_ls_lf=mean_ls_lf, mean_ls_delta=mean_ls_delta,
               best_hf=best_hf, regret_at_80=regret80,
               incumbent_improved_count=n_improved)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"{tag} DONE | best_hf={best_hf:.4f} | regret@80={regret80:.4f} | "
          f"lf_frac={lf_frac:.3f} | mean_ls_lf={mean_ls_lf:.4f} | "
          f"mean_ls_delta={mean_ls_delta:.4f} | incumbent_improved_count={n_improved}",
          flush=True)

elif method == "SF-DRO":
    from checkpoint import setup_dirs
    setup_dirs(EXP_NAME)
    from dro_runner import run_single_seed

    bo_iters = int(COST_BUDGET / 8.0)  # Hartmann_6D c_H=8, HF-only
    result = run_single_seed(
        EXP_NAME, BENCHMARK, "SF-DRO-rotate-MES", seed,
        use_mes_reward=True, rtg_schema="floored", alpha_floor=0.5,
        rollout_acq_function="rotate", gp_num_models=5,
        rollouts_per_iter=75, rollout_length=4,
        bo_iterations=bo_iters, initial_points=INITIAL_HF,
    )
    rc = result["regret_curve"]
    cc = [8.0 * (i + 1) for i in range(len(rc))]
    regret80 = regret_at_cost(cc, rc, 80)
    n_improved = incumbent_improved_count(rc)

    out = {"regret_curve": rc, "cost_curve": cc,
           "regret_at_80": regret80, "incumbent_improved_count": n_improved,
           "lf_frac": None, "mean_ls_lf": None, "mean_ls_delta": None}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"{tag} DONE | regret@80={regret80:.4f} | final_regret={rc[-1]:.4f} | "
          f"lf_frac=N/A (single-fidelity) | mean_ls_lf=N/A | mean_ls_delta=N/A | "
          f"incumbent_improved_count={n_improved}", flush=True)

else:
    raise ValueError(f"unknown method {method!r} for mini stage 2")
