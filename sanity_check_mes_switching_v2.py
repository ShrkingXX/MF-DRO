"""
Pre-flight validation for the mes_switching_v2 cluster experiment
(run_experiment.py's EXP9 / "mes_switching_v2_cluster"), before committing
any cluster hours to the real 200-iteration x 3-benchmark x 5-seed matrix.

Three tiers:
  1. Crash smoke test -- every active EXP9 variant (+ the 2 conditional
     jointMES variants), a few iterations each, on Ackley_2D.
  2. Behavioral sanity checks:
     - regret (best_observed) is monotonically non-decreasing per seed --
       a real bug (e.g. a broken argmax) could let it worsen.
     - rollout action diversity: "rotate" should not be LOWER than a fixed
       acquisition function's diversity (its entire purpose is to diversify
       rollout queries) -- a real behavioral check of the mechanism, not
       just "did it crash".
     - MES acquisition (Gumbel-based qMaxValueEntropy, rollout_acq_function=
       "mes") produces non-degenerate (not bit-identical) queries across a
       rollout batch.
     - NaiveBO-EI, run to 50 iterations (matched to Experiment 1's config),
       lands in the same ballpark as results/mes_reward's existing
       Ackley_2D/Ackley_5D NaiveBO checkpoints (same seeds 42-46) -- a
       regression check against already-trusted data.
  3. Timing calibration -- ~20 real iterations of every EXP9 variant on
     Ackley_2D (the cheapest benchmark), reporting per-phase wall-clock
     (gp_refit/rollout_sim/dt_train/real_query) to replace the rough
     extrapolated cost estimates with real numbers, informing the SLURM
     walltime-chain plan and checkpoint interval.

This only ever touches exp_name="sanity_mes_switching_v2" -- never the real
"mes_switching_v2_cluster" checkpoints.

Usage:
    python sanity_check_mes_switching_v2.py
"""
import json
import statistics as st

import torch

from checkpoint import setup_dirs
from dro_runner import run_single_seed
from run_experiment import EXP9_VARIANT_CONFIGS

EXP_NAME = "sanity_mes_switching_v2"
SMOKE_BENCHMARK = "Ackley_2D"
SMOKE_SEED = 42

ALL_VARIANTS = list(EXP9_VARIANT_CONFIGS.keys())  # includes the 2 conditional jointMES ones


def _tiny_override(cfg, bo_iterations, rollouts_per_iter=8, rollout_length=3,
                    gp_num_models=2, dt_hidden=16, dt_layers=1, dt_heads=1):
    """Shrink an EXP9 variant config to a fast, crash-test-scale version --
    keeps every acquisition/reward/rtg_schema/rollout_acq_function choice
    intact (that's the whole point), just cheaper knobs."""
    c = dict(cfg)
    c["bo_iterations"] = bo_iterations
    if not c.get("is_naive_bo", False):
        c["rollouts_per_iter"] = rollouts_per_iter
        c["rollout_length"] = rollout_length
        c["gp_num_models"] = gp_num_models
        c["dt_hidden"] = dt_hidden
        c["dt_layers"] = dt_layers
        c["dt_heads"] = dt_heads
    return c


# Throwaway reference config, NOT one of the real EXP9 variants -- purely to
# give check 2b a genuine single-fixed-acquisition baseline. The real matrix
# deliberately has no plain "DRO-EI-only" variant (every DRO variant is a
# rotate/MES combo), and comparing rotate's diversity against DRO-MES-*
# variants is the wrong test: MES is itself an exploration-seeking
# acquisition function, so "pure MES" trivially shows high diversity
# regardless of whether "rotate" is working -- confirmed empirically (that
# comparison initially looked like a FAIL before switching to this reference).
_FIXED_EI_REFERENCE = dict(
    rollout_acq_function="ei", use_mes_reward=False, rtg_schema="floored", alpha_floor=0.5,
    gp_num_models=5, rollouts_per_iter=75, rollout_length=8, initial_points=5,
    dt_hidden=128, dt_layers=4, dt_heads=4, gp_kernel="rbf", gp_ard=True,
)


def check1_crash_smoke_test():
    print("=" * 70)
    print("CHECK 1: crash smoke test (all variants, tiny config)")
    print("=" * 70)
    results = {}
    for variant in ALL_VARIANTS + ["Fixed-EI-reference"]:
        base_cfg = _FIXED_EI_REFERENCE if variant == "Fixed-EI-reference" else EXP9_VARIANT_CONFIGS[variant]
        cfg = _tiny_override(base_cfg, bo_iterations=3)
        try:
            r = run_single_seed(
                exp_name=EXP_NAME, benchmark_name=SMOKE_BENCHMARK, variant_name=f"smoke-{variant}",
                seed=SMOKE_SEED, verbose=False, **cfg,
            )
            results[variant] = r
            print(f"  {variant:22s} OK   regret_curve={[round(v, 4) for v in r['regret_curve']]}")
        except Exception as e:
            results[variant] = None
            print(f"  {variant:22s} FAIL  {type(e).__name__}: {e}")
    n_ok = sum(1 for v in results.values() if v is not None)
    print(f"\n  {n_ok}/{len(results)} variants ran without crashing")
    return results


def check2_behavioral(smoke_results):
    print("\n" + "=" * 70)
    print("CHECK 2: behavioral sanity")
    print("=" * 70)

    # 2a. Regret monotonically non-increasing (best_observed non-decreasing)
    print("\n-- 2a. regret monotonicity --")
    all_monotone = True
    for variant, r in smoke_results.items():
        if r is None:
            continue
        rc = r["regret_curve"]
        monotone = all(rc[i + 1] <= rc[i] + 1e-9 for i in range(len(rc) - 1))
        if not monotone:
            all_monotone = False
        print(f"  {variant:22s} monotone_non_increasing={monotone}  curve={[round(v,4) for v in rc]}")
    print(f"  ALL MONOTONE: {all_monotone}")

    # 2b. rollout action diversity: rotate should not be lower than a genuine
    # single-fixed-acquisition baseline's (Fixed-EI-reference specifically --
    # NOT the DRO-MES-* variants, which use "mes", itself an
    # exploration-seeking acquisition function that would trivially show high
    # diversity regardless of whether "rotate" is doing anything).
    print("\n-- 2b. rollout action diversity (rotate vs Fixed-EI-reference) --")
    div_rotate = [d for v, r in smoke_results.items() if r and "rotate" in v
                  for d in r.get("rollout_action_diversity", []) if d is not None]
    ref_r = smoke_results.get("Fixed-EI-reference")
    div_fixed = [d for d in ref_r.get("rollout_action_diversity", []) if d is not None] if ref_r else []
    if div_rotate and div_fixed:
        mean_rotate, mean_fixed = st.mean(div_rotate), st.mean(div_fixed)
        print(f"  mean diversity, rotate variants:       {mean_rotate:.4f}  (n={len(div_rotate)})")
        print(f"  mean diversity, Fixed-EI-reference:    {mean_fixed:.4f}  (n={len(div_fixed)})")
        print(f"  rotate >= fixed-EI: {mean_rotate >= mean_fixed}")
    else:
        print("  SKIPPED (insufficient data -- diversity needs rollout_length>1, check rollout_length)")

    # 2c. MES acquisition non-degenerate (rollout actions not bit-identical)
    print("\n-- 2c. MES acquisition non-degeneracy --")
    for variant in ("DRO-MES-Impr", "DRO-MES-MES"):
        r = smoke_results.get(variant)
        if r is None:
            continue
        div = [d for d in r.get("rollout_action_diversity", []) if d is not None]
        degenerate = any(d < 1e-6 for d in div) if div else None
        print(f"  {variant:22s} rollout_action_diversity per iter={[round(d,4) if d else d for d in div]}  degenerate={degenerate}")


def check2d_naivebo_regression():
    print("\n-- 2d. NaiveBO-EI regression check against results/mes_reward (Experiment 1) --")
    print("   Running NaiveBO-EI to 50 iterations (matched), seeds 42-46, Ackley_2D + Ackley_5D.")
    for benchmark in ("Ackley_2D", "Ackley_5D"):
        new_finals = []
        for seed in (42, 43, 44, 45, 46):
            r = run_single_seed(
                exp_name=EXP_NAME, benchmark_name=benchmark, variant_name="NaiveBO-EI-regression",
                seed=seed, is_naive_bo=True, naivebo_acq_function="ei",
                bo_iterations=50, initial_points=5, verbose=False,
            )
            new_finals.append(r["regret_curve"][-1])
        try:
            with open(f"results/mes_reward/checkpoints/{benchmark}__NaiveBO__seed42.json") as f:
                pass  # just confirm the reference exists before loading all 5
            ref_finals = []
            for seed in (42, 43, 44, 45, 46):
                with open(f"results/mes_reward/checkpoints/{benchmark}__NaiveBO__seed{seed}.json") as f:
                    ref_finals.append(json.load(f)["regret_curve"][-1])
            print(f"  {benchmark}: new mean={st.mean(new_finals):.4f}  reference (EXP1) mean={st.mean(ref_finals):.4f}"
                  f"  new_per_seed={[round(v,4) for v in new_finals]}  ref_per_seed={[round(v,4) for v in ref_finals]}")
        except FileNotFoundError:
            print(f"  {benchmark}: reference data not found, skipping comparison -- new_per_seed={[round(v,4) for v in new_finals]}")


def check3_timing_calibration():
    print("\n" + "=" * 70)
    print("CHECK 3: timing calibration (~20 real iterations, all variants, Ackley_2D)")
    print("=" * 70)
    CALIB_ITERS = 20
    rows = []
    for variant in ALL_VARIANTS:
        cfg = dict(EXP9_VARIANT_CONFIGS[variant])
        cfg["bo_iterations"] = CALIB_ITERS
        r = run_single_seed(
            exp_name=EXP_NAME, benchmark_name=SMOKE_BENCHMARK, variant_name=f"calib-{variant}",
            seed=SMOKE_SEED, verbose=False, **cfg,
        )
        total_time = sum(r["iter_times"])
        per_iter = total_time / len(r["iter_times"]) if r["iter_times"] else float("nan")
        row = {"variant": variant, "total_time_s": total_time, "mean_iter_time_s": per_iter}
        if not cfg.get("is_naive_bo", False):
            gp_t = [t for t in r.get("gp_refit_time", []) if t is not None]
            roll_t = [t for t in r.get("rollout_sim_time", []) if t is not None]
            dt_t = [t for t in r.get("dt_train_time", []) if t is not None]
            rq_t = [t for t in r.get("real_query_time", []) if t is not None]
            row.update({
                "mean_gp_refit_s": st.mean(gp_t) if gp_t else None,
                "mean_rollout_sim_s": st.mean(roll_t) if roll_t else None,
                "mean_dt_train_s": st.mean(dt_t) if dt_t else None,
                "mean_real_query_s": st.mean(rq_t) if rq_t else None,
            })
        rows.append(row)
        print(f"  {variant:22s} total={total_time:7.1f}s  mean_iter={per_iter:6.2f}s  "
              f"(gp={row.get('mean_gp_refit_s')}, roll={row.get('mean_rollout_sim_s')}, "
              f"dt={row.get('mean_dt_train_s')}, rq={row.get('mean_real_query_s')})")

    print("\n  Extrapolated to 200 iterations (LINEAR extrapolation from 20 -- almost certainly an")
    print("  UNDERESTIMATE for DRO variants, since GP refit cost grows with dataset size, not")
    print("  captured by a 20-iteration sample; treat as a lower bound):")
    for row in rows:
        est_200 = row["mean_iter_time_s"] * 200
        print(f"  {row['variant']:22s} ~{est_200/60:.1f} min  (~{est_200/3600:.2f} hours)")


def main():
    setup_dirs(EXP_NAME)
    smoke_results = check1_crash_smoke_test()
    check2_behavioral(smoke_results)
    check2d_naivebo_regression()
    check3_timing_calibration()


if __name__ == "__main__":
    main()
