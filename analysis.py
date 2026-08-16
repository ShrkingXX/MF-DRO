"""
Reads whatever checkpoint results currently exist (partial results are fine)
and produces plots + summary tables for Experiment 1 (MES reward ablation)
and Experiment 2 (RTG schema comparison).

Usage:
    python analysis.py --experiment mes_reward
    python analysis.py --experiment rtg_schema
    python analysis.py --experiment mes_reward --benchmarks Ackley_2D  # subset
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

from checkpoint import RESULTS_ROOT, load_all_results
from run_experiment import (EXP1_NAME, EXP1_BENCHMARKS, EXP1_VARIANTS,
                             EXP2_NAME, EXP2_BENCHMARKS, EXP2_VARIANTS, SEEDS)

EXP1_COLORS = {"DRO_Improvement": "tab:blue", "DRO_MES": "tab:red", "NaiveBO": "tab:green"}
EXP1_ROLLOUT_VARIANTS = ["DRO_Improvement", "DRO_MES"] # NaiveBO has no rollouts

EXP2_STYLE = {
    "DRO_Fixed": ("gray", "solid"),
    "DRO_Dynamic": ("black", "dashed"),
    "DRO_Floored": ("tab:blue", "solid"),
    "DRO_Quantile_0.25": ("purple", "dashed"),
    "DRO_Quantile_0.5": ("green", "dashed"),
    "DRO_Quantile_0.75": ("orange", "solid"),
    "DRO_Quantile_0.9": ("red", "solid"),
}
EXP2_QUANTILE_VARIANTS = ["DRO_Quantile_0.25", "DRO_Quantile_0.5", "DRO_Quantile_0.75", "DRO_Quantile_0.9"]
ALPHA_LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]

# Comparisons reported (Mann-Whitney U on final-iteration regret) in the summary tables.
EXP1_KEY_PAIRS = [("DRO_Improvement", "DRO_MES"), ("DRO_MES", "NaiveBO")]
EXP2_KEY_PAIRS = [
    ("DRO_Fixed", "DRO_Floored"),
    ("DRO_Dynamic", "DRO_Floored"),
    ("DRO_Floored", "DRO_Quantile_0.75"),
]


def _plots_dir(exp_name):
    d = os.path.join(RESULTS_ROOT, exp_name, "plots")
    os.makedirs(d, exist_ok=True)
    return d


def _mean_se_curve(all_res, benchmark, variant, seeds, key):
    """Stack the `key` series across all completed seeds for (benchmark, variant),
    truncating to the shortest available length. Returns (mean, se, n) or
    (None, None, 0) if no seeds are available."""
    series = [all_res[(benchmark, variant, s)][key] for s in seeds
              if (benchmark, variant, s) in all_res and all_res[(benchmark, variant, s)].get(key)]
    if not series:
        return None, None, 0
    min_len = min(len(s) for s in series)
    if min_len == 0:
        return None, None, 0
    arr = np.array([s[:min_len] for s in series], dtype=float)
    mean = np.nanmean(arr, axis=0)
    se = np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0])
    return mean, se, arr.shape[0]


def _plot_mean_se(ax, mean, se, iterations, label, color, linestyle="solid"):
    ax.plot(iterations, mean, label=label, color=color, linestyle=linestyle, linewidth=2)
    ax.fill_between(iterations, mean - se, mean + se, color=color, alpha=0.15)


# --------------------------------------------------------------------------
# Experiment 1 plots
# --------------------------------------------------------------------------
def plot_exp1(benchmarks, variants, seeds):
    all_res = load_all_results(EXP1_NAME, benchmarks, variants, seeds)
    out_dir = _plots_dir(EXP1_NAME)

    for benchmark in benchmarks:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for variant in variants:
            mean, se, n = _mean_se_curve(all_res, benchmark, variant, seeds, "regret_curve")
            if mean is None:
                continue
            iterations = np.arange(1, len(mean) + 1)
            log_mean = np.log10(np.clip(mean, 1e-12, None))
            # Propagate SE to log space via d(log10 x)/dx = 1/(x ln10); clip to
            # avoid blow-up when mean is near zero.
            log_se = se / (np.clip(mean, 1e-12, None) * np.log(10))
            _plot_mean_se(ax1, log_mean, log_se, iterations, f"{variant} (N={n})", EXP1_COLORS.get(variant))

        ax1.set_xlabel("BO Iteration")
        ax1.set_ylabel("log10(simple regret)")
        ax1.set_title(f"{benchmark}: DRO-Original vs DRO-MES vs NaiveBO")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        for variant in EXP1_ROLLOUT_VARIANTS:
            mean, se, n = _mean_se_curve(all_res, benchmark, variant, seeds, "zero_frac")
            if mean is None:
                continue
            iterations = np.arange(1, len(mean) + 1)
            _plot_mean_se(ax2, mean, se, iterations, f"{variant} (N={n})", EXP1_COLORS.get(variant))

        ax2.set_xlabel("BO Iteration")
        ax2.set_ylabel("zero_frac (fraction of zero-reward rollout steps)")
        ax2.set_title(f"{benchmark}: Reward Sparsity (NaiveBO has no rollouts)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        path = os.path.join(out_dir, f"regret_{benchmark}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")


# --------------------------------------------------------------------------
# Experiment 2 plots
# --------------------------------------------------------------------------
def plot_exp2(benchmarks, variants, seeds):
    all_res = load_all_results(EXP2_NAME, benchmarks, variants, seeds)
    out_dir = _plots_dir(EXP2_NAME)

    for benchmark in benchmarks:
        # --- Figure 1: log10(regret) ---
        fig1, ax1 = plt.subplots(figsize=(9, 6))
        for variant in variants:
            mean, se, n = _mean_se_curve(all_res, benchmark, variant, seeds, "regret_curve")
            if mean is None:
                continue
            iterations = np.arange(1, len(mean) + 1)
            log_mean = np.log10(np.clip(mean, 1e-12, None))
            log_se = se / (np.clip(mean, 1e-12, None) * np.log(10))
            color, style = EXP2_STYLE.get(variant, ("tab:gray", "solid"))
            _plot_mean_se(ax1, log_mean, log_se, iterations, f"{variant} (N={n})", color, style)
        ax1.set_xlabel("BO Iteration")
        ax1.set_ylabel("log10(simple regret)")
        ax1.set_title(f"{benchmark}: RTG Schema Comparison")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        path1 = os.path.join(out_dir, f"regret_{benchmark}.png")
        fig1.savefig(path1, dpi=150)
        plt.close(fig1)
        print(f"Saved {path1}")

        # --- Figure 2: raw rtg_target ---
        fig2, ax2 = plt.subplots(figsize=(9, 6))
        for variant in variants:
            mean, se, n = _mean_se_curve(all_res, benchmark, variant, seeds, "rtg_target")
            if mean is None:
                continue
            iterations = np.arange(1, len(mean) + 1)
            color, style = EXP2_STYLE.get(variant, ("tab:gray", "solid"))
            _plot_mean_se(ax2, mean, se, iterations, f"{variant} (N={n})", color, style)
        ax2.set_xlabel("BO Iteration")
        ax2.set_ylabel("RTG target at inference (raw)")
        ax2.set_title(f"{benchmark}: RTG Target at Inference")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        path2 = os.path.join(out_dir, f"rtg_target_{benchmark}.png")
        fig2.savefig(path2, dpi=150)
        plt.close(fig2)
        print(f"Saved {path2}")

        # --- Figure 3: calibration for DRO_Quantile_0.75 ---
        if "DRO_Quantile_0.75" in variants:
            fig3, ax3 = plt.subplots(figsize=(9, 6))
            calib_series = [all_res[(benchmark, "DRO_Quantile_0.75", s)]["calibration"]
                             for s in seeds
                             if (benchmark, "DRO_Quantile_0.75", s) in all_res
                             and all_res[(benchmark, "DRO_Quantile_0.75", s)].get("calibration")]
            if calib_series:
                min_len = min(len(s) for s in calib_series)
                # calibration[t] is a length-M list, except during the first
                # rtg_warmup iterations (schema still falling back to floored,
                # quantile head not yet used), where it's None -- replace with
                # NaN-filled placeholders of the same width so the array below
                # has a uniform shape (NaN then propagates as a natural gap).
                M = len(ALPHA_LEVELS)
                nan_row = [float("nan")] * M
                arr = np.array(
                    [[(row if row is not None else nan_row) for row in s[:min_len]] for s in calib_series],
                    dtype=float,
                )
                mean_calib = np.nanmean(arr, axis=0) # [T, M]
                iterations = np.arange(1, min_len + 1)
                cmap = plt.get_cmap("viridis")
                for j, alpha_j in enumerate(ALPHA_LEVELS):
                    ax3.plot(iterations, mean_calib[:, j], label=f"alpha={alpha_j}",
                              color=cmap(j / (len(ALPHA_LEVELS) - 1)))
                    ax3.axhline(y=alpha_j, color=cmap(j / (len(ALPHA_LEVELS) - 1)), linestyle=":", alpha=0.5)
                ax3.set_xlabel("BO Iteration")
                ax3.set_ylabel("Empirical coverage")
                ax3.set_title(f"{benchmark}: Quantile Calibration (DRO_Quantile_0.75)")
                ax3.legend(fontsize=8)
                ax3.grid(True, alpha=0.3)
                fig3.tight_layout()
                path3 = os.path.join(out_dir, f"calibration_{benchmark}.png")
                fig3.savefig(path3, dpi=150)
                print(f"Saved {path3}")
            plt.close(fig3)

        # --- Figure 4: quantile spread ---
        fig4, ax4 = plt.subplots(figsize=(9, 6))
        any_data = False
        for variant in EXP2_QUANTILE_VARIANTS:
            if variant not in variants:
                continue
            mean, se, n = _mean_se_curve(all_res, benchmark, variant, seeds, "quantile_spread")
            if mean is None:
                continue
            any_data = True
            iterations = np.arange(1, len(mean) + 1)
            color, style = EXP2_STYLE.get(variant, ("tab:gray", "solid"))
            _plot_mean_se(ax4, mean, se, iterations, f"{variant} (N={n})", color, style)
        ax4.set_xlabel("BO Iteration")
        ax4.set_ylabel("Quantile spread (Q[0.9] - Q[0.1])")
        ax4.set_title(f"{benchmark}: Quantile Spread Over Iterations")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        fig4.tight_layout()
        if any_data:
            path4 = os.path.join(out_dir, f"spread_{benchmark}.png")
            fig4.savefig(path4, dpi=150)
            print(f"Saved {path4}")
        plt.close(fig4)


# --------------------------------------------------------------------------
# Summary tables
# --------------------------------------------------------------------------
def summary_table(exp_name, benchmarks, variants, seeds, key_pairs):
    all_res = load_all_results(exp_name, benchmarks, variants, seeds)
    lines = [f"=== {exp_name}: Final Regret Summary (mean +/- SE, iteration {'50'}) ==="]

    for benchmark in benchmarks:
        lines.append(f"\n{benchmark}:")
        stats = {} # variant -> (mean, se, n, values)
        for variant in variants:
            values = [all_res[(benchmark, variant, s)]["regret_curve"][-1]
                      for s in seeds
                      if (benchmark, variant, s) in all_res and all_res[(benchmark, variant, s)].get("regret_curve")]
            if values:
                stats[variant] = (np.mean(values), np.std(values) / np.sqrt(len(values)), len(values), values)

        if not stats:
            lines.append("  (no completed runs yet)")
            continue

        best_variant = min(stats, key=lambda v: stats[v][0])
        best_mean, best_se = stats[best_variant][0], stats[best_variant][1]

        for variant in variants:
            if variant not in stats:
                lines.append(f"  {variant:20s}: (no completed runs)")
                continue
            mean, se, n, _ = stats[variant]
            is_best = (variant == best_variant)
            within_1se = abs(mean - best_mean) <= (se + best_se)
            marker = "*" if (within_1se and not is_best) else ""
            label = f"**{variant}**" if is_best else f"{variant}{marker}"
            lines.append(f"  {label:24s}: {mean:.4f} +/- {se:.4f}  (N={n})")

        lines.append("  Mann-Whitney U tests:")
        for a, b in key_pairs:
            if a in stats and b in stats:
                _, p = mannwhitneyu(stats[a][3], stats[b][3], alternative="two-sided")
                lines.append(f"    {a} vs {b}: p={p:.4f}")
            else:
                lines.append(f"    {a} vs {b}: (insufficient data)")

    text = "\n".join(lines)
    print(text)
    summary_path = os.path.join(RESULTS_ROOT, exp_name, "summary.txt")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        f.write(text + "\n")
    print(f"\nSaved {summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate plots and summary tables from DRO experiment checkpoints.")
    parser.add_argument("--experiment", required=True, choices=[EXP1_NAME, EXP2_NAME])
    parser.add_argument("--benchmarks", type=str, nargs="+", default=None)
    parser.add_argument("--variants", type=str, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    seeds = args.seeds or SEEDS
    if args.experiment == EXP1_NAME:
        benchmarks = args.benchmarks or EXP1_BENCHMARKS
        variants = args.variants or EXP1_VARIANTS
        plot_exp1(benchmarks, variants, seeds)
        summary_table(EXP1_NAME, benchmarks, variants, seeds, EXP1_KEY_PAIRS)
    else:
        benchmarks = args.benchmarks or EXP2_BENCHMARKS
        variants = args.variants or EXP2_VARIANTS
        plot_exp2(benchmarks, variants, seeds)
        summary_table(EXP2_NAME, benchmarks, variants, seeds, EXP2_KEY_PAIRS)
