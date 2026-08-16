"""
Phase 3: check whether the joint RTG formulation produces RTG values at a
scale compatible with the floored-dynamic RTG schema, and whether the joint
RTG signal is large enough (relative to its own noise) to be useful as a DT
training target. Hartmann_6D and Currin_2D only -- Phase 2 confirmed
Ackley_2D/5D and Rosenbrock_2D show 41-57% negative joint RTG[0] (no
measurable entropy decrease over a 4-step rollout), so the joint formulation
is not viable there regardless of scale; they are excluded per that finding.

Phase 2's saved JSON (phase2_rtg_correlation.json) only contains aggregate
statistics, not the raw per-rollout RTG[0] lists, so this script regenerates
rollouts via the same simulate_with_gumbel_b used there.

NOTE on seeding: the task description for this phase states Phase 2 v3 used
`seed = SEED + hash(benchmark) % 1000`. That is not what the actual v3 code
does -- I checked validation/phase2_rtg_correlation.py directly, and it uses
`SEED + BENCHMARKS.index(benchmark)` (deliberately, since Python's string
hash() is randomized per-process by default and would make seeding
irreproducible across runs -- this was flagged and fixed in the last Phase 2
bug-fix round). This script uses that same deterministic, index-based scheme
for consistency with the actual Phase 2 v3 results, not the hash-based one
described in this task.

Read-only diagnostic: does not modify any existing code. Reuses
simulate_with_gumbel_b/compute_both_rtg from phase2_rtg_correlation.py and
_make_dro from phase1_gumbel_quality.py.

Usage:
    python validation/phase3_scale_check.py
"""
import json
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase1_gumbel_quality import _make_dro
from phase2_rtg_correlation import simulate_with_gumbel_b, compute_both_rtg

BENCHMARKS = ["Hartmann_6D", "Currin_2D"] # Ackley_2D/5D, Rosenbrock_2D excluded per Phase 2
ALL_PHASE2_BENCHMARKS = ["Ackley_2D", "Ackley_5D", "Rosenbrock_2D", "Hartmann_6D", "Currin_2D"]
STAGES = [("early", 5), ("mid", 25), ("late", 45)]
SEED = 42
N_ROLLOUTS = 100
MAX_LENGTH = 4
K_THOMPSON = 100
ALPHA_FLOOR = 0.5
N_ITER_SIM = 20
BATCH_SIZE = 5 # 100 rollouts / 20 iterations = 5 rollouts per simulated BO iteration

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots_v4") # v4 (post roi_candidates bugfix) -- separate from original plots/ dir
RESULTS_PATH = os.path.join(RESULTS_DIR, "phase3_scale_check_v4.json") # v3 (pre-bugfix) preserved at phase3_scale_check.json


def _percentiles(values):
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)), "std": float(np.std(arr)),
        "min": float(np.min(arr)), "max": float(np.max(arr)),
        "p10": float(np.percentile(arr, 10)), "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)), "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def simulate_floored_schema(rtg0_list, alpha=ALPHA_FLOOR):
    batches = [rtg0_list[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] for i in range(N_ITER_SIM)]
    running_max = 0.0
    targets = []
    for batch in batches:
        batch_max = max(batch)
        running_max = max(running_max, batch_max)
        target = max(batch_max, alpha * running_max)
        targets.append(target)
    return targets


def _classify(scale_ratio, snr, neg_frac, target_collapse):
    if target_collapse:
        return "FAIL"
    if scale_ratio > 20.0 or scale_ratio < 0.05 or snr < 0.2 or neg_frac >= 0.25:
        return "FAIL"
    if (5.0 <= scale_ratio <= 20.0) or (0.05 <= scale_ratio < 0.2) or (0.2 <= snr < 0.5) or (0.15 <= neg_frac < 0.25):
        return "WARN"
    if 0.2 <= scale_ratio <= 5.0 and snr >= 0.5 and neg_frac < 0.15:
        return "PASS"
    return "WARN" # any other in-between case not explicitly covered


def _collect_rollouts(benchmark, n_initial, seed):
    """Regenerate N_ROLLOUTS rollouts, returning per_step_rtg0, joint_rtg0
    (both length N_ROLLOUTS), and all_step_joint (length N_ROLLOUTS*MAX_LENGTH,
    every tau, unclamped)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dro = _make_dro(benchmark, n_initial, seed=seed)
    dro._update_models()

    per_step_rtg0, joint_rtg0, all_step_joint = [], [], []
    for _ in range(N_ROLLOUTS):
        state = dro._extract_state(dro.data_x, dro.data_y, dro.data_x.shape[0])
        traj = simulate_with_gumbel_b(dro, gp_idx=0, initial_state=state,
                                       max_length=MAX_LENGTH, K_thompson=K_THOMPSON)
        per_step_rtg, joint_rtg = compute_both_rtg(traj)
        per_step_rtg0.append(per_step_rtg[0].item())
        joint_rtg0.append(joint_rtg[0].item())
        all_step_joint.extend(joint_rtg.tolist())

    return per_step_rtg0, joint_rtg0, all_step_joint


def _plot_distribution(benchmark, stage_data):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (stage, d) in zip(axes, stage_data.items()):
        ax.hist(d["per_step_rtg0"], bins=20, alpha=0.5, color="tab:blue", label="per-step RTG[0]")
        ax.hist(d["joint_rtg0"], bins=20, alpha=0.5, color="tab:red", label="joint RTG[0]")
        ax.axvline(d["per_step_stats"]["mean"], color="tab:blue", linestyle="--", linewidth=1.5)
        ax.axvline(d["joint_stats"]["mean"], color="tab:red", linestyle="--", linewidth=1.5)
        # Floored-schema target range for each formulation (final running_max/batch_max)
        ps_targets = simulate_floored_schema(d["per_step_rtg0"])
        j_targets = simulate_floored_schema(d["joint_rtg0"])
        ax.axvspan(ALPHA_FLOOR * max(d["per_step_rtg0"]), max(ps_targets), color="tab:blue", alpha=0.08)
        ax.axvspan(ALPHA_FLOOR * max(d["joint_rtg0"]), max(j_targets), color="tab:red", alpha=0.08)
        ax.set_title(f"{stage}")
        ax.set_xlabel("RTG[0] value")
        ax.legend(fontsize=8)
    fig.suptitle(f"{benchmark}: RTG[0] distribution -- per-step (blue) vs joint (red)")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, f"phase3_distribution_{benchmark}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_floored_target(benchmark, stage_data):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (stage, d) in zip(axes, stage_data.items()):
        iters = np.arange(1, N_ITER_SIM + 1)
        ax.plot(iters, d["per_step_targets"], color="tab:blue", label="per-step target", linewidth=2)
        ax.plot(iters, d["joint_targets"], color="tab:red", label="joint target", linewidth=2)
        ax.set_title(f"{stage}")
        ax.set_xlabel("BO iteration")
        ax.set_ylabel("floored dynamic RTG target")
        ax.legend(fontsize=8)
    fig.suptitle(f"{benchmark}: Floored dynamic RTG target (per-step vs joint)")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, f"phase3_floored_target_{benchmark}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_log_ratio(benchmark, all_step_joint_combined):
    fig, ax = plt.subplots(figsize=(7, 5))
    arr = np.array(all_step_joint_combined)
    bins = np.linspace(arr.min(), arr.max(), 40)
    pos = arr[arr >= 0]
    neg = arr[arr < 0]
    ax.hist(pos, bins=bins, color="tab:green", alpha=0.7, label=f"positive (n={len(pos)})")
    ax.hist(neg, bins=bins, color="tab:red", alpha=0.7, label=f"negative (n={len(neg)})")
    ax.axvline(0.0, color="black", linewidth=1)
    ax.axvline(arr.mean(), color="black", linestyle="--", linewidth=1.5, label=f"mean={arr.mean():.3f}")
    ax.set_title(f"{benchmark}: Step-level entropy decrease log(b_tau/b_T)\n"
                 f"(all 4 steps x 100 rollouts x 3 stages = {len(arr)} values)")
    ax.set_xlabel("log(b_tau / b_T)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, f"phase3_log_ratio_{benchmark}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    from checkpoint import setup_dirs
    setup_dirs("validation_gumbel")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("=" * 100)
    print("PHASE 3: RTG SCALE COMPATIBILITY")
    print("(Hartmann_6D and Currin_2D only -- Ackley/Rosenbrock excluded per Phase 2 neg_frac 41-57%)")
    print("=" * 100)
    print()

    all_results = {}
    for benchmark in BENCHMARKS:
        benchmark_seed = SEED + ALL_PHASE2_BENCHMARKS.index(benchmark) # matches actual Phase 2 v3 scheme
        stage_data = {}
        combined_step_joint = []

        for stage, n_initial in STAGES:
            per_step_rtg0, joint_rtg0, all_step_joint = _collect_rollouts(benchmark, n_initial, benchmark_seed)
            combined_step_joint.extend(all_step_joint)

            per_step_stats = _percentiles(per_step_rtg0)
            joint_stats = _percentiles(joint_rtg0)

            joint_mean = joint_stats["mean"]
            if joint_mean <= 0:
                scale_ratio = float('inf')
            else:
                scale_ratio = per_step_stats["mean"] / max(joint_mean, 1e-9)

            snr = joint_stats["mean"] / max(joint_stats["std"], 1e-9)
            neg_frac = sum(1 for v in joint_rtg0 if v < 0) / len(joint_rtg0)

            per_step_targets = simulate_floored_schema(per_step_rtg0)
            joint_targets = simulate_floored_schema(joint_rtg0)
            j_target_mean = float(np.mean(joint_targets))
            ps_target_mean = float(np.mean(per_step_targets))
            target_mean_ratio = ps_target_mean / max(j_target_mean, 1e-9)
            target_collapse = 1 if j_target_mean < 0.01 else 0

            step_log_ratio_stats = {
                "mean_log_ratio": float(np.mean(all_step_joint)),
                "std_log_ratio": float(np.std(all_step_joint)),
                "frac_positive": float(np.mean([1 if v > 0 else 0 for v in all_step_joint])),
            }

            status = _classify(scale_ratio, snr, neg_frac, target_collapse)

            stage_data[stage] = dict(
                per_step_rtg0=per_step_rtg0, joint_rtg0=joint_rtg0,
                per_step_stats=per_step_stats, joint_stats=joint_stats,
                scale_ratio=scale_ratio, snr=snr, neg_frac=neg_frac,
                per_step_targets=per_step_targets, joint_targets=joint_targets,
                target_mean_ratio=target_mean_ratio, target_collapse=target_collapse,
                step_log_ratio_stats=step_log_ratio_stats,
                status=status,
            )
            print(f"  done: {benchmark:<12} {stage:<7} PS_mean={per_step_stats['mean']:.4f} "
                  f"J_mean={joint_stats['mean']:.4f} scale_ratio={scale_ratio:.3f} snr={snr:.3f} "
                  f"neg_frac={neg_frac:.3f} status={status}")

        all_results[benchmark] = dict(
            stages=stage_data,
            combined_step_joint_stats={
                "mean_log_ratio": float(np.mean(combined_step_joint)),
                "std_log_ratio": float(np.std(combined_step_joint)),
                "frac_positive": float(np.mean([1 if v > 0 else 0 for v in combined_step_joint])),
                "n": len(combined_step_joint),
            },
        )
        all_results[benchmark]["_combined_step_joint_raw"] = combined_step_joint

    # --- Plots ---
    print("\nGenerating plots...")
    for benchmark in BENCHMARKS:
        stage_data = all_results[benchmark]["stages"]
        p1 = _plot_distribution(benchmark, stage_data)
        p2 = _plot_floored_target(benchmark, stage_data)
        p3 = _plot_log_ratio(benchmark, all_results[benchmark]["_combined_step_joint_raw"])
        print(f"  {benchmark}: {p1}\n  {benchmark}: {p2}\n  {benchmark}: {p3}")

    # --- Table ---
    print("\n=== PHASE 3: RTG SCALE COMPATIBILITY ===")
    print("(Hartmann_6D and Currin_2D only -- Ackley/Rosenbrock excluded per Phase 2)\n")
    header = (f"{'Benchmark':<12} {'Stage':<7} {'PS_mean':>8} {'J_mean':>8} {'Scale_ratio':>11} "
              f"{'SNR':>6} {'Neg%':>6} {'Target_ratio':>12} {'Collapse':>8}  STATUS")
    print(header)
    print("-" * len(header))
    for benchmark in BENCHMARKS:
        for stage, _ in STAGES:
            d = all_results[benchmark]["stages"][stage]
            sr_str = "inf" if math.isinf(d["scale_ratio"]) else f"{d['scale_ratio']:.3f}x"
            print(f"{benchmark:<12} {stage:<7} {d['per_step_stats']['mean']:>8.4f} "
                  f"{d['joint_stats']['mean']:>8.4f} {sr_str:>11} {d['snr']:>6.2f} "
                  f"{d['neg_frac']*100:>5.1f}% {d['target_mean_ratio']:>11.3f}x "
                  f"{'Yes' if d['target_collapse'] else 'No':>8}  {d['status']}")

    # --- Summary ---
    print("\n=== SUMMARY ===\n")
    for benchmark in BENCHMARKS:
        statuses = [all_results[benchmark]["stages"][s]["status"] for s, _ in STAGES]
        target_ratios = [all_results[benchmark]["stages"][s]["target_mean_ratio"] for s, _ in STAGES]
        print(f"{benchmark}: {statuses}, target_mean_ratio={[round(t,3) for t in target_ratios]}")

    print("\nLog ratio distribution (step-level, all stages combined):")
    for benchmark in BENCHMARKS:
        c = all_results[benchmark]["combined_step_joint_stats"]
        print(f"{benchmark:<12} mean={c['mean_log_ratio']:.3f} std={c['std_log_ratio']:.3f} "
              f"frac_positive={c['frac_positive']*100:.1f}%  (n={c['n']})")

    print("\nRECOMMENDATION:")
    for benchmark in BENCHMARKS:
        statuses = [all_results[benchmark]["stages"][s]["status"] for s, _ in STAGES]
        all_targets = [t for s, _ in STAGES for t in all_results[benchmark]["stages"][s]["joint_targets"]]
        min_t, max_t = min(all_targets), max(all_targets)
        print(f"\n{benchmark}:")
        if all(s == "PASS" for s in statuses):
            print(f"  Joint RTG is scale-compatible with floored schema.")
            print(f"  If switching, reset running_max_rtg=0.0 at start of first iteration using")
            print(f"  joint RTG to avoid stale scale from per-step RTG.")
            print(f"  Expected joint RTG target range: [{min_t:.3f}, {max_t:.3f}]")
            print(f"  Proceed to Phase 4 empirical comparison.")
        elif any(s == "FAIL" for s in statuses):
            print(f"  Joint RTG is NOT scale-compatible or signal is too weak to be a useful")
            print(f"  training target. Keep per-step formulation for this benchmark.")
        else:
            print(f"  Joint RTG is marginally compatible. If switching:")
            print(f"  - If WARN due to scale: reset running_max_rtg=0.0")
            print(f"  - If WARN due to SNR: consider increasing K from 100 to 200 for better")
            print(f"    Gumbel MLE reliability.")
            print(f"  Proceed to Phase 4 with caution.")

    if any(all_results[b]["stages"][s]["status"] in ("PASS", "WARN") for b in BENCHMARKS for s, _ in STAGES):
        print("\nNote: alpha_floor=0.5 requires no change (dimensionless ratio). The only")
        print("mechanical change when switching is resetting running_max_rtg=0.0 at the")
        print("start of the first joint-RTG iteration.")

    # --- Save JSON (drop raw combined arrays to keep file size reasonable, keep per-stage raw lists) ---
    json_results = {}
    for benchmark in BENCHMARKS:
        json_results[benchmark] = {
            "stages": all_results[benchmark]["stages"],
            "combined_step_joint_stats": all_results[benchmark]["combined_step_joint_stats"],
        }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSaved full results to {RESULTS_PATH}")


if __name__ == '__main__':
    main()
