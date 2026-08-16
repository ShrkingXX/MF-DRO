"""
Phase 1 validation: audit the quality of the Gumbel approximation to the
max-value distribution used by mes_reward.py's gumbel_sample_y_star, across
all five Experiment 1 benchmarks and three GP-training stages (early/mid/late
real-query count). This is a prerequisite check before deciding whether to
switch to the joint RTG formulation.

Read-only diagnostic: does not modify any existing code. Reuses
mes_reward.py's _cdf_of_max / _bracket_for_quantiles directly rather than
duplicating them.

Usage:
    python validation/phase1_gumbel_quality.py
"""
import json
import math
import os
import sys

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.stats import ks_2samp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mes_reward import _cdf_of_max, _bracket_for_quantiles, _EPS_STD
from benchmarks import get_benchmark
from dro_runner import _build_dro_config
from src.policy.dro import DirectRegretOptimization

BENCHMARKS = ["Ackley_2D", "Ackley_5D", "Rosenbrock_2D", "Hartmann_6D", "Currin_2D"]
STAGES = {"early": 5, "mid": 25, "late": 45}
SEED = 42
N_EMPIRICAL = 500

_EULER_MASCHERONI = 0.5772156649015329

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "phase1_gumbel_quality.json")


def get_gumbel_params(mu_candidates: torch.Tensor, sigma_candidates: torch.Tensor):
    """
    Extract Gumbel(a, b) parameters using the same 25th/75th-percentile
    matching logic as mes_reward.gumbel_sample_y_star, but return (a, b)
    instead of drawing samples. Imports _cdf_of_max/_bracket_for_quantiles
    directly from mes_reward.py rather than duplicating them.

    Returns: (a: float, b: float)
    """
    mu_np = mu_candidates.detach().cpu().double().numpy()
    sigma_np = np.clip(sigma_candidates.detach().cpu().double().numpy(), _EPS_STD, None)

    lo, hi = _bracket_for_quantiles(mu_np, sigma_np)

    z_25 = brentq(lambda z: _cdf_of_max(z, mu_np, sigma_np) - 0.25, lo, hi)
    z_75 = brentq(lambda z: _cdf_of_max(z, mu_np, sigma_np) - 0.75, lo, hi)

    log_term_25 = math.log(-math.log(0.25))
    log_term_75 = math.log(-math.log(0.75))

    b = (z_75 - z_25) / (log_term_25 - log_term_75)
    a = z_25 + b * log_term_25
    return float(a), float(b)


def gumbel_fit_quality(model, roi_candidates: torch.Tensor, n_empirical: int = 500, seed: int = 42) -> dict:
    """
    Compare the Gumbel approximation to the empirical maximum distribution,
    both derived from the same GP posterior at roi_candidates.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    with torch.no_grad():
        posterior = model.posterior(roi_candidates)
        mu = posterior.mean.reshape(-1)
        sigma = posterior.variance.clamp_min(_EPS_STD ** 2).sqrt().reshape(-1)

        a, b = get_gumbel_params(mu, sigma)

        # Empirical maxima: joint rsample over roi_candidates -- captures the
        # GP posterior's actual cross-candidate correlation, unlike the Gumbel
        # approximation's product-of-independent-CDFs assumption. This is
        # exactly the gap being audited.
        samples = posterior.rsample(torch.Size([n_empirical])) # [n_empirical, N_roi, 1]
        samples = samples.reshape(n_empirical, -1)
        empirical_maxima = samples.max(dim=-1).values.cpu().numpy()

    u = torch.rand(n_empirical, dtype=torch.float64).clamp(1e-6, 1.0 - 1e-6)
    gumbel_samples = (a - b * torch.log(-torch.log(u))).numpy()

    ks_stat, ks_p = ks_2samp(empirical_maxima, gumbel_samples)

    gumbel_mean = a + b * _EULER_MASCHERONI
    gumbel_std = b * math.pi / math.sqrt(6)
    empirical_mean = float(np.mean(empirical_maxima))
    empirical_std = float(np.std(empirical_maxima))
    mean_bias = gumbel_mean - empirical_mean
    std_ratio = gumbel_std / empirical_std if empirical_std > 0 else float('nan')

    return dict(
        ks_stat=float(ks_stat), ks_p=float(ks_p),
        gumbel_mean=float(gumbel_mean), gumbel_std=float(gumbel_std), gumbel_b=float(b),
        empirical_mean=empirical_mean, empirical_std=empirical_std,
        mean_bias=float(mean_bias), std_ratio=float(std_ratio),
    )


def _make_dro(benchmark_name: str, n_initial: int, seed: int = 42):
    benchmark_spec = get_benchmark(benchmark_name)
    objective = benchmark_spec["make_objective"]()
    cfg = _build_dro_config(
        exp_name="validation_gumbel", benchmark_name=benchmark_name, variant_name="gumbel_audit", seed=seed,
        use_mes_reward=True, rtg_schema="floored", alpha_floor=0.5, alpha_inference=None,
        lambda_rtg=1.0, rtg_warmup=3, benchmark_spec=benchmark_spec,
        gp_num_models=5, rollouts_per_iter=75, rollout_length=4, bo_iterations=50, initial_points=n_initial,
        dt_hidden=128, dt_layers=4, dt_heads=4, dt_lr=1e-4, gp_kernel="rbf", gp_ard=True, verbose=False,
    )
    dro = DirectRegretOptimization(cfg, objective)
    dro.sample_initial_points()
    dro._initialize_models()
    return dro


def _classify(ks_stat: float) -> str:
    if ks_stat < 0.15:
        return "PASS"
    elif ks_stat <= 0.25:
        return "WARN"
    else:
        return "FAIL"


def main():
    from checkpoint import setup_dirs
    setup_dirs("validation_gumbel")

    results = {}
    for benchmark in BENCHMARKS:
        for stage, n_initial in STAGES.items():
            dro = _make_dro(benchmark, n_initial, seed=SEED)
            observed_best = dro.data_y.max()
            best_x, roi_candidates = dro._optimize_acquisition(gp_idx=0, observed_best=observed_best)
            model = dro.gp_ensemble[0]['model']
            quality = gumbel_fit_quality(model, roi_candidates, n_empirical=N_EMPIRICAL, seed=SEED)
            quality["n_roi_candidates"] = int(roi_candidates.shape[0])
            results[(benchmark, stage)] = quality
            print(f"  done: {benchmark:<15} {stage:<7} (n_initial={n_initial}, "
                  f"N_roi={quality['n_roi_candidates']}, KS={quality['ks_stat']:.4f})")

    print("\n=== GUMBEL FIT QUALITY AUDIT ===")
    header = (f"{'Benchmark':<15} {'Stage':<7} {'KS_stat':>8} {'KS_p':>7} {'b':>8} "
              f"{'mean_bias':>10} {'std_ratio':>10}  {'STATUS'}")
    print(header)
    print("-" * len(header))
    for benchmark in BENCHMARKS:
        for stage in STAGES:
            r = results[(benchmark, stage)]
            status = _classify(r['ks_stat'])
            print(f"{benchmark:<15} {stage:<7} {r['ks_stat']:>8.4f} {r['ks_p']:>7.4f} "
                  f"{r['gumbel_b']:>8.4f} {r['mean_bias']:>10.4f} {r['std_ratio']:>10.4f}  {status}")

    warn_list = [(b, s) for b in BENCHMARKS for s in STAGES if _classify(results[(b, s)]['ks_stat']) == "WARN"]
    fail_list = [(b, s) for b in BENCHMARKS for s in STAGES if _classify(results[(b, s)]['ks_stat']) == "FAIL"]

    print()
    if not warn_list and not fail_list:
        print("Phase 1 PASSED. Gumbel approximation is reliable. Proceed to Phase 2.")
    if warn_list:
        warn_benchmarks = sorted(set(b for b, s in warn_list))
        warn_stages = sorted(set(s for b, s in warn_list))
        print(f"Phase 1 WARNING on {warn_benchmarks}. Gumbel approximation is marginal "
              f"on these benchmarks at {warn_stages}. Proceed to Phase 2 with caution for these benchmarks.")
    if fail_list:
        fail_benchmarks = sorted(set(b for b, s in fail_list))
        print(f"Phase 1 FAILED on {fail_benchmarks}. Do NOT switch to joint formulation for these benchmarks.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_results = {
        f"{b}__{s}": {**results[(b, s)], "status": _classify(results[(b, s)]['ks_stat'])}
        for b in BENCHMARKS for s in STAGES
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSaved full results to {RESULTS_PATH}")


if __name__ == '__main__':
    main()
