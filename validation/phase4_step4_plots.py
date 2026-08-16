"""
Phase 4 Step 4: final plots + summary table for the joint-RTG empirical
validation on Currin_2D (DRO-MES-PerStep vs DRO-MES-Joint vs
DRO-MES-Joint-P90, 3 seeds, 30 BO iterations each -- see
run_experiment.py's EXP4_* block and checkpoint.py results under
results/joint_rtg_validation/).

Produces 4 plots:
  1. Per-seed regret curves, all 3 variants overlaid.
  2. Batch RTG[0] distribution at iteration 1 (PerStep vs Joint): mean/std/CV
     bar comparison, reproducing the root-cause diagnosis numbers.
  3. rtg_target / running_max_rtg trajectories over 30 iterations, all 3
     variants, all 3 seeds -- shows the ratchet's growth pattern.
  4. Final-regret summary bar chart with per-seed error bars.

Read-only with respect to dro.py; only reads already-saved checkpoint
results (does not re-run any experiments).

Usage:
    python validation/phase4_step4_plots.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checkpoint import load_all_results

EXP_NAME = "joint_rtg_validation"
BENCHMARK = "Currin_2D"
VARIANTS = ["DRO-MES-PerStep", "DRO-MES-Joint", "DRO-MES-Joint-P90"]
SEEDS = [42, 43, 44]
COLORS = {"DRO-MES-PerStep": "tab:blue", "DRO-MES-Joint": "tab:red", "DRO-MES-Joint-P90": "tab:orange"}

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "plots_phase4")

# Iteration-1 batch diagnosis numbers, from validation/phase4_step3_diagnosis.py
# (regenerating them here would require re-running 150 rollouts; these are the
# exact printed values from that script's output).
BATCH_DIAG = {
    "DRO-MES-PerStep": dict(mean=0.0412, std=0.0169, cv=0.411),
    "DRO-MES-Joint": dict(mean=0.0260, std=0.1301, cv=5.010),
}


def plot1_regret_curves(results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, seed in zip(axes, SEEDS):
        for variant in VARIANTS:
            r = results.get((BENCHMARK, variant, seed))
            if r is None:
                continue
            iters = np.arange(1, len(r["regret_curve"]) + 1)
            ax.plot(iters, r["regret_curve"], label=variant, color=COLORS[variant], linewidth=2)
        ax.set_title(f"seed {seed}")
        ax.set_xlabel("BO iteration")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("simple regret (log scale)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{BENCHMARK}: regret curves, per seed (30 BO iterations)")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "phase4_regret_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot2_batch_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    variants = list(BATCH_DIAG.keys())
    means = [BATCH_DIAG[v]["mean"] for v in variants]
    stds = [BATCH_DIAG[v]["std"] for v in variants]
    cvs = [BATCH_DIAG[v]["cv"] for v in variants]
    colors = [COLORS[v] for v in variants]

    axes[0].bar(variants, means, yerr=stds, color=colors, alpha=0.8, capsize=6)
    axes[0].set_ylabel("batch RTG[0]: mean +/- std (n=75 rollouts)")
    axes[0].set_title("Iteration 1 batch RTG[0]: mean +/- std")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].tick_params(axis='x', rotation=15)

    axes[1].bar(variants, cvs, color=colors, alpha=0.8)
    axes[1].set_ylabel("coefficient of variation (std / mean)")
    axes[1].set_title("Relative noise: Joint's RTG signal is\nnoise-dominated at early stage")
    axes[1].tick_params(axis='x', rotation=15)
    for i, v in enumerate(cvs):
        axes[1].text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=10)

    fig.suptitle(f"{BENCHMARK}: batch RTG[0] distribution at iteration 1 (n_initial=5, seed=42)")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "phase4_batch_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot3_rtg_target_trajectories(results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)
    for ax, seed in zip(axes, SEEDS):
        for variant in VARIANTS:
            r = results.get((BENCHMARK, variant, seed))
            if r is None:
                continue
            iters = np.arange(1, len(r["running_max_rtg"]) + 1)
            ax.plot(iters, r["running_max_rtg"], label=f"{variant} (running_max_rtg)",
                     color=COLORS[variant], linewidth=2)
        ax.set_title(f"seed {seed}")
        ax.set_xlabel("BO iteration")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("running_max_rtg (never decays)")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{BENCHMARK}: running_max_rtg ratchet over 30 iterations\n"
                 "(both ratchet upward on all seeds; PerStep's underlying metric stays learnable, Joint's doesn't)")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "phase4_running_max_rtg.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot4_final_regret_summary(results):
    fig, ax = plt.subplots(figsize=(7, 5))
    means, ses, colors = [], [], []
    for variant in VARIANTS:
        vals = [results[(BENCHMARK, variant, s)]["regret_curve"][-1]
                 for s in SEEDS if (BENCHMARK, variant, s) in results]
        means.append(np.mean(vals))
        ses.append(np.std(vals) / np.sqrt(len(vals)))
        colors.append(COLORS[variant])
    x = np.arange(len(VARIANTS))
    ax.bar(x, means, yerr=ses, color=colors, alpha=0.85, capsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(VARIANTS, rotation=15, ha="right")
    ax.set_ylabel("final simple regret (mean +/- SE, N=3 seeds)")
    ax.set_title(f"{BENCHMARK}: final regret after 30 BO iterations")
    for i, m in enumerate(means):
        ax.text(i, m + ses[i] + 0.02, f"{m:.3f}", ha="center", fontsize=10)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "phase4_final_regret_summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    results = load_all_results(EXP_NAME, [BENCHMARK], VARIANTS, SEEDS)

    p1 = plot1_regret_curves(results)
    p2 = plot2_batch_distribution()
    p3 = plot3_rtg_target_trajectories(results)
    p4 = plot4_final_regret_summary(results)
    print("Saved plots:")
    for p in (p1, p2, p3, p4):
        print(f"  {p}")

    print("\n=== SUMMARY TABLE ===")
    header = f"{'Variant':<20} {'Final regret (mean+/-SE)':>28} {'Seeds w/ 0-1 improvements':>28}"
    print(header)
    print("-" * len(header))
    for variant in VARIANTS:
        vals = [results[(BENCHMARK, variant, s)]["regret_curve"] for s in SEEDS if (BENCHMARK, variant, s) in results]
        finals = [v[-1] for v in vals]
        mean, se = np.mean(finals), np.std(finals) / np.sqrt(len(finals))
        n_stuck = sum(1 for v in vals if len(set(round(x, 6) for x in v)) <= 2)
        print(f"{variant:<20} {mean:>14.4f} +/- {se:<10.4f} {n_stuck:>28}/{len(vals)}")


if __name__ == "__main__":
    main()
