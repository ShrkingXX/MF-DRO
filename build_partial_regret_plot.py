"""
Build a regret-vs-iteration plot directly from the LIVE per-iteration log
files (results/{exp_name}/logs/{benchmark}__{variant}__seed{seed}.log,
written by checkpoint.py's log_iter -- one line appended after every real BO
iteration, flushed to disk immediately). Works on an experiment that's still
running or was killed mid-run: whatever iterations have completed so far are
plotted, with no requirement that any (benchmark, variant, seed) run has
actually finished (.done flag). This is the tool for "show partial progress
before the run is complete" (e.g. 100/200 iterations in before a morning
meeting) -- NOT a replacement for the final analysis, which should use the
completed .json checkpoints once every run is done.

Regex-parses just the `iter=` and `regret=` fields out of each line (the
other fields -- mean_reward, rtg_target, gp_refit_time, etc. -- aren't needed
for this plot and are ignored, so this doesn't need to keep in sync with
every field checkpoint.py's log_iter happens to write).

Usage:
    python build_partial_regret_plot.py --exp-name mes_switching_v2_cluster \\
        --benchmarks Ackley_2D Ackley_5D Ackley_10D \\
        --variants DRO-rotate-Impr DRO-rotate-MES DRO-MES-Impr DRO-MES-MES NaiveBO-EI NaiveBO-MES \\
        --seeds 42 43 44 45 46 \\
        --out-dir results/mes_switching_v2_cluster/plots

    # Or just point it at an exp_name and it'll infer benchmarks/variants/seeds
    # from whatever log files already exist:
    python build_partial_regret_plot.py --exp-name mes_switching_v2_cluster
"""
import argparse
import glob
import os
import re
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ITER_RE = re.compile(r"\biter=(-?\d+)")
_REGRET_RE = re.compile(r"\bregret=(-?[\d.eE+-]+)")


def parse_log_file(path):
    """Returns dict {iter_index: regret} for one (benchmark, variant, seed)
    log file, deduplicated by iter (last line wins, in case of a rare
    duplicate write from a killed-and-immediately-resumed process writing
    the same iteration twice)."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            m_iter = _ITER_RE.search(line)
            m_regret = _REGRET_RE.search(line)
            if m_iter and m_regret:
                try:
                    out[int(m_iter.group(1))] = float(m_regret.group(1))
                except ValueError:
                    continue
    return out


def discover_grid(exp_name):
    """Infer (benchmarks, variants, seeds) from whatever log files exist,
    when the caller doesn't specify them explicitly."""
    log_dir = os.path.join("results", exp_name, "logs")
    pattern = os.path.join(log_dir, "*.log")
    benchmarks, variants, seeds = set(), set(), set()
    for path in glob.glob(pattern):
        base = os.path.basename(path)[:-4]  # strip ".log"
        parts = base.split("__")
        if len(parts) != 3 or not parts[2].startswith("seed"):
            continue
        benchmark, variant, seed_str = parts
        try:
            seed = int(seed_str[len("seed"):])
        except ValueError:
            continue
        benchmarks.add(benchmark)
        variants.add(variant)
        seeds.add(seed)
    return sorted(benchmarks), sorted(variants), sorted(seeds)


def build_plot(exp_name, benchmarks, variants, seeds, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    log_dir = os.path.join("results", exp_name, "logs")

    for benchmark in benchmarks:
        fig, ax = plt.subplots(figsize=(9, 6))
        any_data = False
        max_iter_seen = 0

        for variant in variants:
            per_seed_curves = []
            for seed in seeds:
                path = os.path.join(log_dir, f"{benchmark}__{variant}__seed{seed}.log")
                iter_to_regret = parse_log_file(path)
                if iter_to_regret:
                    per_seed_curves.append(iter_to_regret)

            if not per_seed_curves:
                continue
            any_data = True

            # Only plot iterations where AT LEAST ONE seed has data --
            # mean/SE computed from however many seeds have reached that
            # iteration so far (a partial run may have seeds at different
            # points, e.g. one resumed further than another).
            all_iters = sorted(set().union(*(c.keys() for c in per_seed_curves)))
            max_iter_seen = max(max_iter_seen, max(all_iters, default=0))
            means, ses, n_seeds_per_iter = [], [], []
            for it in all_iters:
                vals = [c[it] for c in per_seed_curves if it in c]
                means.append(st.mean(vals))
                ses.append((st.stdev(vals) / len(vals) ** 0.5) if len(vals) > 1 else 0.0)
                n_seeds_per_iter.append(len(vals))

            n_complete_seeds = sum(1 for c in per_seed_curves if c)
            label = f"{variant} (n={n_complete_seeds} seed{'s' if n_complete_seeds != 1 else ''}, " \
                    f"up to iter {max(all_iters)})"
            means_arr = means
            los = [m - s for m, s in zip(means, ses)]
            his = [m + s for m, s in zip(means, ses)]
            ax.plot(all_iters, means_arr, label=label, linewidth=1.8)
            ax.fill_between(all_iters, los, his, alpha=0.15)

        if not any_data:
            plt.close(fig)
            print(f"  {benchmark}: no log data found yet, skipping")
            continue

        ax.set_xlabel("Real BO iteration")
        ax.set_ylabel("Simple regret")
        ax.set_title(f"{benchmark} -- simple regret vs iteration (PARTIAL, up to iter {max_iter_seen})")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{benchmark}_regret_partial.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  {benchmark}: saved {out_path} (up to iteration {max_iter_seen})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--benchmarks", nargs="+", default=None)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    benchmarks, variants, seeds = args.benchmarks, args.variants, args.seeds
    if benchmarks is None or variants is None or seeds is None:
        d_benchmarks, d_variants, d_seeds = discover_grid(args.exp_name)
        benchmarks = benchmarks or d_benchmarks
        variants = variants or d_variants
        seeds = seeds or d_seeds
        print(f"Inferred from existing logs: benchmarks={benchmarks}  variants={variants}  seeds={seeds}")

    out_dir = args.out_dir or os.path.join("results", args.exp_name, "plots")
    build_plot(args.exp_name, benchmarks, variants, seeds, out_dir)


if __name__ == "__main__":
    main()
