"""
Phase 1 re-run: audit Gumbel fit quality using Thompson sampling (exact,
correlated GP posterior samples) instead of the product-CDF approximation
used in phase1_gumbel_quality.py. The original run found KS stats of
0.19-0.86 across all 15 (benchmark, stage) combinations, traced to the
product-CDF formula's independence assumption being violated under RBF
kernel correlations. This re-run removes that assumption entirely (Thompson
samples are drawn jointly from the true posterior via rsample()) to test
whether the Gumbel *family* itself is a good fit once independence is no
longer the confound.

Read-only diagnostic: does not modify any existing code. Reuses
phase1_gumbel_quality.py's _make_dro helper rather than duplicating it.

Usage:
    python validation/phase1_thompson_gumbel.py
"""
import json
import math
import os
import sys

import numpy as np
import torch
from scipy.stats import ks_2samp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase1_gumbel_quality import _make_dro
from gumbel_thompson import thompson_sample_y_star, fit_gumbel_to_samples

BENCHMARKS = ["Ackley_2D", "Ackley_5D", "Rosenbrock_2D", "Hartmann_6D", "Currin_2D"]
STAGES = [("early", 5), ("mid", 25), ("late", 45)]
SEED = 42
K_FIT = 500
K_TEST = 500

_EULER_GAMMA = 0.5772156649015329

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OLD_RESULTS_PATH = os.path.join(RESULTS_DIR, "phase1_gumbel_quality.json")
NEW_RESULTS_PATH = os.path.join(RESULTS_DIR, "phase1_thompson_gumbel.json")


def run_gumbel_quality_check(model, roi_candidates: torch.Tensor, K_fit: int = 500, K_test: int = 500,
                              benchmark_name: str = "", stage: str = "") -> dict:
    """
    Full Thompson-sampling-based Gumbel quality check for one
    (benchmark, stage) combination. See module docstring for rationale.
    """
    fit_samples = thompson_sample_y_star(model, roi_candidates, K=K_fit)
    a, b = fit_gumbel_to_samples(fit_samples)

    fresh_samples = thompson_sample_y_star(model, roi_candidates, K=K_test)

    u = np.random.uniform(1e-6, 1.0 - 1e-6, size=K_test)
    gumbel_samples = a - b * np.log(-np.log(u))

    ks_stat, ks_p = ks_2samp(fresh_samples, gumbel_samples)

    gumbel_mean = a + b * _EULER_GAMMA
    gumbel_std = b * math.pi / math.sqrt(6)
    gumbel_entropy = math.log(b) + _EULER_GAMMA + 1 # nats

    empirical_mean = float(np.mean(fresh_samples))
    empirical_std = float(np.std(fresh_samples))
    mean_bias = gumbel_mean - empirical_mean
    std_ratio = gumbel_std / empirical_std if empirical_std > 0 else float('nan')

    return dict(
        ks_stat=float(ks_stat), ks_p=float(ks_p), a=a, b=b,
        gumbel_mean=float(gumbel_mean), gumbel_std=float(gumbel_std), gumbel_entropy=float(gumbel_entropy),
        empirical_mean=empirical_mean, empirical_std=empirical_std,
        mean_bias=float(mean_bias), std_ratio=float(std_ratio),
        benchmark_name=benchmark_name, stage=stage,
    )


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

    with open(OLD_RESULTS_PATH) as f:
        old_results = json.load(f)

    results = {}
    for benchmark in BENCHMARKS:
        for stage, n_initial in STAGES:
            dro = _make_dro(benchmark, n_initial, seed=SEED)
            observed_best = dro.data_y.max()
            best_x, roi_candidates = dro._optimize_acquisition(gp_idx=0, observed_best=observed_best)
            model = dro.gp_ensemble[0]['model']
            model.eval()

            torch.manual_seed(SEED)
            np.random.seed(SEED)
            r = run_gumbel_quality_check(model, roi_candidates, K_fit=K_FIT, K_test=K_TEST,
                                          benchmark_name=benchmark, stage=stage)
            r["n_roi_candidates"] = int(roi_candidates.shape[0])
            r["old_ks_stat"] = old_results[f"{benchmark}__{stage}"]["ks_stat"]
            r["status"] = _classify(r["ks_stat"])
            results[(benchmark, stage)] = r
            print(f"  done: {benchmark:<15} {stage:<7} (n_initial={n_initial}, N_roi={r['n_roi_candidates']}, "
                  f"old_KS={r['old_ks_stat']:.4f}, new_KS={r['ks_stat']:.4f})")

    print("\n=== PHASE 1 RE-RUN: THOMPSON GUMBEL FIT ===")
    print("(Comparing product-CDF Gumbel vs Thompson-fitted Gumbel)\n")
    header = f"{'Benchmark':<15} {'Stage':<7} {'Old_KS':>7} {'New_KS':>8} {'b':>8} {'entropy':>9}  STATUS"
    print(header)
    print("-" * len(header))
    for benchmark in BENCHMARKS:
        for stage, _ in STAGES:
            r = results[(benchmark, stage)]
            print(f"{benchmark:<15} {stage:<7} {r['old_ks_stat']:>7.3f} {r['ks_stat']:>8.3f} "
                  f"{r['b']:>8.4f} {r['gumbel_entropy']:>9.4f}  {r['status']}")

    all_stage_names = [s for s, _ in STAGES]
    pass_all = [b for b in BENCHMARKS if all(results[(b, s)]["status"] == "PASS" for s in all_stage_names)]
    any_warn = [b for b in BENCHMARKS if any(results[(b, s)]["status"] == "WARN" for s in all_stage_names)]
    any_fail = [b for b in BENCHMARKS if any(results[(b, s)]["status"] == "FAIL" for s in all_stage_names)]

    old_ks_vals = [results[(b, s)]["old_ks_stat"] for b in BENCHMARKS for s in all_stage_names]
    new_ks_vals = [results[(b, s)]["ks_stat"] for b in BENCHMARKS for s in all_stage_names]

    print("\n=== SUMMARY ===\n")
    print(f"Benchmarks with ALL stages PASS:  {pass_all}")
    print(f"Benchmarks with any WARN:         {any_warn}")
    print(f"Benchmarks with any FAIL:         {any_fail}")
    print(f"\nMean KS stat improvement (old vs new): {np.mean(old_ks_vals):.3f} -> {np.mean(new_ks_vals):.3f}")

    total = len(BENCHMARKS) * len(all_stage_names)
    n_pass = sum(1 for b in BENCHMARKS for s in all_stage_names if results[(b, s)]["status"] == "PASS")
    n_fail = sum(1 for b in BENCHMARKS for s in all_stage_names if results[(b, s)]["status"] == "FAIL")

    print("\nINTERPRETATION:")
    if n_pass == total:
        print("Gumbel family is correct once independence assumption is removed.")
        print("Joint RTG formulation is viable on all benchmarks.")
        print("Use: H(y*) = log(b) + euler_gamma + 1")
        print("where b is fitted via MLE to K=50 Thompson samples per rollout step.")
        print("Proceed to Phase 2 correlation check.")
    elif n_fail == total:
        print("Gumbel family does not fit even with correct samples.")
        print("Maximum distribution is genuinely non-Gumbel.")
        print("Must use non-parametric KL entropy estimator for joint RTG.")
    else:
        passing = [b for b in BENCHMARKS if b not in any_fail]
        failing = any_fail
        print(f"Gumbel family holds on: {passing}")
        print(f"Non-parametric entropy needed on: {failing}")
        print("Recommendation: use Gumbel for passing, KL estimator for failing.")

    print("\nIf joint RTG is used, H(y* | D_tau) at these stages would be:")
    for benchmark in BENCHMARKS:
        for stage, _ in STAGES:
            r = results[(benchmark, stage)]
            print(f"  {benchmark:<15} {stage:<7} -> H = {r['gumbel_entropy']:.4f} nats")
        h_early = results[(benchmark, "early")]["gumbel_entropy"]
        h_late = results[(benchmark, "late")]["gumbel_entropy"]
        rtg_0 = h_early - h_late
        print(f"  RTG_0 = H_early - H_late for {benchmark} would be approximately: {rtg_0:.4f} nats\n")
    print("This gives a preview of joint RTG magnitudes before Phase 2.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_results = {f"{b}__{s}": results[(b, s)] for b in BENCHMARKS for s, _ in STAGES}
    with open(NEW_RESULTS_PATH, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSaved full results to {NEW_RESULTS_PATH}")


if __name__ == '__main__':
    main()
