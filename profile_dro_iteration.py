"""
Profile where wall-clock time actually goes in a real DRO-MES run, following
up on mes_cost_benchmark.py's finding that MES reward computation is only
~8.4% of total wall-clock time even after a 4x speedup there.

Instruments the sequential, non-overlapping top-level phases of
run_optimization's per-iteration loop (src/policy/base.py):
    _update_models (GP hyperparameter fitting)
    -> _propose_next_candidate, itself broken into:
         _simulate_trajectory (rollout simulation, called rollouts_per_iter times)
         _train_decision_transformer (DT training)
         "dt_inference_and_final_acq" (residual: DT forward pass + the one
           _optimize_acquisition call for the real candidate + bookkeeping)
    -> objective function evaluation
    -> _update_data_and_best (logging/bookkeeping)
These sum to (approximately) total wall-clock time by construction, since
they're sequential calls in run_optimization's loop body.

Additionally, within _simulate_trajectory's time, nests two sub-breakdowns
(these OVERLAP with _simulate_trajectory's total, they are not additional
top-level buckets):
    _optimize_acquisition (called 4x per rollout -- broad random search +
      local refinement)
    compute_mes_reward (called 4x per rollout)

Uses the same monkey-patch-with-timing-wrapper technique as
mes_cost_benchmark.py, applied to multiple DirectRegretOptimization methods
at once. Runs one full 30-iteration DRO-MES-Fast run on Hartmann_6D
(consistent with mes_cost_benchmark.py's config) -- single seed, since this
is a "where does time go" profile, not a statistical comparison.

Usage:
    python profile_dro_iteration.py
"""
import functools
import time

import numpy as np
import torch

from checkpoint import setup_dirs
from dro_runner import _build_dro_config
from benchmarks import get_benchmark
import src.policy.dro as dro_module
from src.policy.dro import DirectRegretOptimization

BENCHMARK = "Hartmann_6D"
SEED = 42
EXP_NAME = "profile_dro_iteration"

_SHARED = dict(
    use_mes_reward=True, rtg_schema="floored", alpha_floor=0.5,
    gp_num_models=5, rollouts_per_iter=75, rollout_length=4,
    bo_iterations=30, initial_points=5,
    dt_hidden=128, dt_layers=4, dt_heads=4,
    gp_kernel="rbf", gp_ard=True,
)


class Timers:
    def __init__(self):
        self.totals = {}
        self.counts = {}

    def wrap(self, name, fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            dt = time.perf_counter() - t0
            self.totals[name] = self.totals.get(name, 0.0) + dt
            self.counts[name] = self.counts.get(name, 0) + 1
            return result
        return wrapped


def main():
    setup_dirs(EXP_NAME)
    timers = Timers()

    # --- Patch the class methods we want top-level/nested timing for ---
    originals = {}
    for name in ["_update_models", "_simulate_trajectory", "_train_decision_transformer", "_optimize_acquisition"]:
        originals[name] = getattr(DirectRegretOptimization, name)
        setattr(DirectRegretOptimization, name, timers.wrap(name, originals[name]))
    originals["compute_mes_reward"] = dro_module.compute_mes_reward
    dro_module.compute_mes_reward = timers.wrap("compute_mes_reward", originals["compute_mes_reward"])

    try:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        benchmark_spec = get_benchmark(BENCHMARK)
        objective = benchmark_spec["make_objective"]()
        cfg = _build_dro_config(
            exp_name=EXP_NAME, benchmark_name=BENCHMARK, variant_name="DRO-MES-Fast-profiled", seed=SEED,
            alpha_inference=None, lambda_rtg=1.0, rtg_warmup=3, benchmark_spec=benchmark_spec,
            dt_lr=1e-4, verbose=False, mes_k=10, **_SHARED,
        )
        dro = DirectRegretOptimization(cfg, objective)

        t_wall_start = time.perf_counter()
        dro.run_optimization()
        wall_clock = time.perf_counter() - t_wall_start

    finally:
        for name, fn in originals.items():
            if name == "compute_mes_reward":
                dro_module.compute_mes_reward = fn
            else:
                setattr(DirectRegretOptimization, name, fn)

    t = timers.totals
    c = timers.counts

    # dt_inference_and_final_acq: everything inside _propose_next_candidate
    # not already accounted for by _simulate_trajectory or
    # _train_decision_transformer (DT forward pass for the real candidate,
    # the one _optimize_acquisition call for it, bookkeeping). Approximated
    # as: total wall-clock minus every top-level bucket we DID measure
    # directly (update_models, simulate_trajectory, train_dt), minus
    # negligible objective-eval/logging time (benchmark functions here are
    # closed-form and cheap, confirmed by the residual itself being small).
    top_level_measured = t.get("_update_models", 0.0) + t.get("_simulate_trajectory", 0.0) + \
                          t.get("_train_decision_transformer", 0.0)
    residual = wall_clock - top_level_measured

    print("=" * 100)
    print(f"PROFILE: {BENCHMARK}, seed={SEED}, {_SHARED['bo_iterations']} BO iterations, "
          f"{_SHARED['rollouts_per_iter']} rollouts/iter, DRO-MES-Fast implementation")
    print("=" * 100)
    print(f"\nTotal wall-clock: {wall_clock:.1f}s\n")

    print("--- Top-level breakdown (sequential, sums to ~total wall-clock) ---")
    rows = [
        ("_update_models (GP fitting)", t.get("_update_models", 0.0), c.get("_update_models", 0)),
        ("_simulate_trajectory (rollout simulation)", t.get("_simulate_trajectory", 0.0), c.get("_simulate_trajectory", 0)),
        ("_train_decision_transformer (DT training)", t.get("_train_decision_transformer", 0.0), c.get("_train_decision_transformer", 0)),
        ("residual (DT inference, final acq, objective eval, logging)", residual, None),
    ]
    for label, secs, calls in rows:
        pct = 100.0 * secs / wall_clock
        call_str = f"{calls} calls" if calls is not None else "n/a"
        print(f"  {label:<62} {secs:>8.1f}s  ({pct:>5.1f}%)  {call_str}")

    print("\n--- Nested breakdown WITHIN _simulate_trajectory's time (overlaps, not additive) ---")
    sim_total = t.get("_simulate_trajectory", 0.0)
    for name in ["_optimize_acquisition", "compute_mes_reward"]:
        secs = t.get(name, 0.0)
        calls = c.get(name, 0)
        pct_of_sim = 100.0 * secs / sim_total if sim_total > 0 else float("nan")
        pct_of_total = 100.0 * secs / wall_clock
        print(f"  {name:<30} {secs:>8.1f}s  calls={calls:<6}  "
              f"({pct_of_sim:>5.1f}% of rollout-sim time, {pct_of_total:>5.1f}% of total)")
    sim_residual = sim_total - t.get("_optimize_acquisition", 0.0) - t.get("compute_mes_reward", 0.0)
    print(f"  {'residual (posterior.sample(), _extract_state, etc.)':<30} {sim_residual:>8.1f}s  "
          f"({100.0*sim_residual/sim_total if sim_total>0 else float('nan'):>5.1f}% of rollout-sim time)")


if __name__ == "__main__":
    main()
