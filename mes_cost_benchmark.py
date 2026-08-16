"""
Before/after cost + performance comparison for MES reward, on Hartmann_6D:

  BASELINE: compute_mes_reward as it was before this experiment (Gumbel
            y* sampling via gumbel_sample_y_star's product-CDF/brentq
            approach, two separate .posterior() calls).
  FAST:     compute_mes_reward as currently implemented in mes_reward.py
            (solution #1: Thompson-sampled y* instead of the product-CDF
            Gumbel fit; solution #4: roi_candidates + x_tau batched into
            one .posterior() call).

Records, per (variant, seed):
  - wall-clock cost: total run time, plus per-iteration iter_time (from
    the existing checkpoint infrastructure).
  - computational cost: cumulative time spent specifically inside
    compute_mes_reward across the whole run, and call count -- isolates
    the exact code being optimized from GP-fitting/DT-training overhead.
  - final simple regret: regret_curve[-1], the standard metric used
    throughout this investigation.

BASELINE is run via monkey-patching src.policy.dro's compute_mes_reward
name back to the pre-edit implementation for the duration of that run only
(restored immediately after) -- mes_reward.py itself is not modified by
this script, only src.policy.dro's already-imported reference to the
function is temporarily redirected.

Usage:
    python mes_cost_benchmark.py
"""
import functools
import json
import os
import time

import numpy as np
import torch

from checkpoint import setup_dirs
from dro_runner import _build_dro_config
from benchmarks import get_benchmark
import src.policy.dro as dro_module
from src.policy.dro import DirectRegretOptimization
from mes_reward import gumbel_sample_y_star, _EPS_STD, _EPS_PHI, _LOG_SQRT_2PI_E

BENCHMARK = "Hartmann_6D"
SEEDS = [42, 43, 44]
EXP_NAME = "mes_cost_benchmark"

_SHARED = dict(
    use_mes_reward=True, rtg_schema="floored", alpha_floor=0.5,
    gp_num_models=5, rollouts_per_iter=75, rollout_length=4,
    bo_iterations=30, initial_points=5,
    dt_hidden=128, dt_layers=4, dt_heads=4,
    gp_kernel="rbf", gp_ard=True,
)

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "mes_cost_benchmark.json")


def compute_mes_reward_baseline(x_tau: torch.Tensor, gp_model, roi_candidates: torch.Tensor, K: int = 10) -> torch.Tensor:
    """Verbatim reconstruction of compute_mes_reward as it was before this
    experiment (product-CDF Gumbel y* sampling, two separate .posterior()
    calls) -- reuses gumbel_sample_y_star and the private constants, which
    are unchanged in mes_reward.py, so this is faithful to the original."""
    device, dtype = roi_candidates.device, roi_candidates.dtype
    with torch.no_grad():
        roi_posterior = gp_model.posterior(roi_candidates)
        mu_roi = roi_posterior.mean.reshape(-1)
        sigma_roi = roi_posterior.variance.clamp_min(_EPS_STD ** 2).sqrt().reshape(-1)

        y_star_samples = gumbel_sample_y_star(mu_roi, sigma_roi, K=K)

        x_tau_batched = x_tau if x_tau.ndim > 1 else x_tau.unsqueeze(0)
        x_posterior = gp_model.posterior(x_tau_batched)
        mu_x = x_posterior.mean.reshape(-1)[0]
        sigma_x = x_posterior.variance.clamp_min(_EPS_STD ** 2).sqrt().reshape(-1)[0]

    log_sqrt_2pi_e = torch.tensor(_LOG_SQRT_2PI_E, device=device, dtype=dtype)
    term1 = torch.log(sigma_x) + log_sqrt_2pi_e

    gamma = (y_star_samples - mu_x) / sigma_x
    standard_normal = torch.distributions.Normal(
        torch.zeros((), device=device, dtype=dtype),
        torch.ones((), device=device, dtype=dtype),
    )
    Phi = standard_normal.cdf(gamma).clamp_min(_EPS_PHI)
    phi = standard_normal.log_prob(gamma).exp()

    H = torch.log(sigma_x) + log_sqrt_2pi_e + torch.log(Phi) - (gamma * phi) / (2 * Phi)
    term2 = H.mean()

    reward = term1 - term2
    return reward.clamp_min(0.0)


class _MesRewardTimer:
    """Accumulates cumulative time + call count for whichever
    compute_mes_reward implementation is currently active in
    src.policy.dro's namespace."""
    def __init__(self):
        self.total_time = 0.0
        self.call_count = 0

    def reset(self):
        self.total_time = 0.0
        self.call_count = 0

    def wrap(self, fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            self.total_time += time.perf_counter() - t0
            self.call_count += 1
            return result
        return wrapped


def run_one(variant: str, impl_fn, seed: int, timer: _MesRewardTimer):
    timer.reset()
    original = dro_module.compute_mes_reward
    dro_module.compute_mes_reward = timer.wrap(impl_fn)
    try:
        benchmark_spec = get_benchmark(BENCHMARK)
        objective = benchmark_spec["make_objective"]()
        cfg = _build_dro_config(
            exp_name=EXP_NAME, benchmark_name=BENCHMARK, variant_name=variant, seed=seed,
            alpha_inference=None, lambda_rtg=1.0, rtg_warmup=3, benchmark_spec=benchmark_spec,
            dt_lr=1e-4, verbose=False, mes_k=10, **_SHARED,
        )
        dro = DirectRegretOptimization(cfg, objective)

        t_wall_start = time.perf_counter()
        dro.run_optimization()
        wall_clock = time.perf_counter() - t_wall_start

        history = dro.iteration_log_history
        result = dict(
            variant=variant, seed=seed,
            wall_clock_total=wall_clock,
            iter_times=[d["iter_time"] for d in history],
            mes_reward_total_time=timer.total_time,
            mes_reward_call_count=timer.call_count,
            regret_curve=[d["regret"] for d in history],
            final_regret=history[-1]["regret"] if history else None,
        )
        return result
    finally:
        dro_module.compute_mes_reward = original


def main():
    setup_dirs(EXP_NAME)
    timer = _MesRewardTimer()

    variants = {
        "DRO-MES-Baseline": compute_mes_reward_baseline,
        "DRO-MES-Fast": dro_module.compute_mes_reward, # current mes_reward.py implementation (#1 + #4 applied)
    }

    all_results = {}
    for variant, impl_fn in variants.items():
        for seed in SEEDS:
            print(f"Running {variant} seed={seed} ...")
            torch.manual_seed(seed)
            np.random.seed(seed)
            r = run_one(variant, impl_fn, seed, timer)
            all_results[(variant, seed)] = r
            print(f"  wall_clock={r['wall_clock_total']:.1f}s  "
                  f"mes_reward_time={r['mes_reward_total_time']:.2f}s "
                  f"({r['mes_reward_call_count']} calls)  "
                  f"final_regret={r['final_regret']:.4f}")

    print("\n" + "=" * 100)
    print(f"SUMMARY: {BENCHMARK}, {len(SEEDS)} seeds, {_SHARED['bo_iterations']} BO iterations, "
          f"{_SHARED['rollouts_per_iter']} rollouts/iter")
    print("=" * 100)
    header = (f"{'Variant':<20} {'Wall-clock (s)':>16} {'MES-reward time (s)':>20} "
              f"{'MES-reward calls':>18} {'Final regret':>14}")
    print(header)
    print("-" * len(header))
    summary = {}
    for variant in variants:
        wcs = [all_results[(variant, s)]["wall_clock_total"] for s in SEEDS]
        mts = [all_results[(variant, s)]["mes_reward_total_time"] for s in SEEDS]
        ccs = [all_results[(variant, s)]["mes_reward_call_count"] for s in SEEDS]
        frs = [all_results[(variant, s)]["final_regret"] for s in SEEDS]
        summary[variant] = dict(
            wall_clock_mean=float(np.mean(wcs)), wall_clock_se=float(np.std(wcs) / np.sqrt(len(wcs))),
            mes_reward_time_mean=float(np.mean(mts)), mes_reward_time_se=float(np.std(mts) / np.sqrt(len(mts))),
            mes_reward_calls_mean=float(np.mean(ccs)),
            final_regret_mean=float(np.mean(frs)), final_regret_se=float(np.std(frs) / np.sqrt(len(frs))),
        )
        s = summary[variant]
        print(f"{variant:<20} {s['wall_clock_mean']:>9.1f}+/-{s['wall_clock_se']:<5.1f} "
              f"{s['mes_reward_time_mean']:>13.2f}+/-{s['mes_reward_time_se']:<5.2f} "
              f"{s['mes_reward_calls_mean']:>18.0f} "
              f"{s['final_regret_mean']:>7.4f}+/-{s['final_regret_se']:<5.4f}")

    if "DRO-MES-Baseline" in summary and "DRO-MES-Fast" in summary:
        b, f = summary["DRO-MES-Baseline"], summary["DRO-MES-Fast"]
        print(f"\nWall-clock speedup:     {b['wall_clock_mean'] / f['wall_clock_mean']:.2f}x")
        print(f"MES-reward-time speedup: {b['mes_reward_time_mean'] / f['mes_reward_time_mean']:.2f}x")
        print(f"Final regret change:    {b['final_regret_mean']:.4f} -> {f['final_regret_mean']:.4f}")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    json_results = {f"{v}__seed{s}": r for (v, s), r in all_results.items()}
    json_results["_summary"] = summary
    with open(RESULTS_PATH, "w") as fp:
        json.dump(json_results, fp, indent=2)
    print(f"\nSaved full results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
