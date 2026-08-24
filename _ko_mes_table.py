"""
Comparison table and regret plots for the KO-MES paper experiment.

    PYTHONPATH=. .venv/bin/python3 _ko_mes_table.py

Primary metric: log10(simple regret) against POST-INIT cost, reported as
mean +/- 1 SE over the 5 seeds, read at the 10x / 30x / 100x c_H checkpoints.

CHECKPOINT READING: a run's regret curve is best-so-far, hence monotone
non-increasing, so the regret "achieved by budget C" is the curve value at
the LAST iteration whose cumulative post-init cost is <= C. Methods spend
their budget at different granularities (MF-MI-Greedy logs once per round of
several queries; SF-MES spends c_H per iteration), so reading by cost rather
than by iteration index is what makes the columns comparable. A run that has
not yet made any query within budget C falls back to its post-initialization
best, reconstructed from initial_hf_values.
"""
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

from benchmarks import get_benchmark

EXP_DIR = os.path.join("results", "ko_mes_paper")
CKPT_DIR = os.path.join(EXP_DIR, "checkpoints")
SEEDS = (42, 43, 44, 45, 46)
BENCHMARKS = ("Currin_2D", "Hartmann_6D", "Borehole_8D")
# Display order; the two rows the KO-vs-additive contrast lives on are last.
METHOD_ORDER = ("SF-MES", "MF-GP-UCB", "MF-MI-Greedy",
                "Additive-MES-Song", "Additive-MES", "KO-MES")
METHOD_NOTE = {
    "Additive-MES": "<- additive model (rho pinned to 1)",
    "Additive-MES-Song": "<- additive model (Song two-stage)",
    "KO-MES": "<- KO model (rho fitted)",
}
CHECKPOINT_MULTIPLIERS = (10, 30, 100)
# Simple regret can legitimately reach 0 (Currin_2D routinely does), which has
# no logarithm; floor it so the entry stays readable and flag it in the legend
# rather than dropping the seed from the mean.
REGRET_FLOOR = 1e-8


def regret_at_cost(result, budget, fallback_regret):
    costs = result["cost_curve"]
    regrets = result["regret_curve"]
    best = fallback_regret
    for c, r in zip(costs, regrets):
        if c <= budget:
            best = r
        else:
            break
    return best


def post_init_regret(result, known_opt):
    """Best HF value from the initial design alone, as a regret."""
    init_vals = result.get("initial_hf_values")
    if not init_vals:
        return float("nan")
    return -max(init_vals) - known_opt


def load_all():
    runs = defaultdict(dict)  # (benchmark, method) -> {seed: result}
    for path in sorted(glob.glob(os.path.join(CKPT_DIR, "*.json"))):
        name = os.path.basename(path)[: -len(".json")]
        method, benchmark, seed_s = name.split("__")
        with open(path) as f:
            runs[(benchmark, method)][int(seed_s[len("seed"):])] = json.load(f)
    return runs


def mean_se(values):
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    se = arr.std(ddof=1) / math.sqrt(arr.size) if arr.size > 1 else 0.0
    return float(arr.mean()), float(se), int(arr.size)


def main():
    runs = load_all()
    if not runs:
        print(f"No checkpoints found in {CKPT_DIR}")
        return

    summary = {}

    for benchmark in BENCHMARKS:
        hf = get_benchmark(f"{benchmark}_HF")
        c_H = hf["cost"]
        known_opt = hf["known_optimal_value"]
        checkpoints = [m * c_H for m in CHECKPOINT_MULTIPLIERS]

        rho_vals = [r.get("final_rho") for r in runs.get((benchmark, "KO-MES"), {}).values()
                    if r.get("final_rho") is not None]
        rho_str = f"rho fitted ~{np.mean(rho_vals):.3f}" if rho_vals else "rho n/a"

        print()
        print(f"Benchmark: {benchmark}  (d={hf['dim']}, c_H={c_H:g}, c_L="
              f"{get_benchmark(f'{benchmark}_LF')['cost']:g}, KO model: {rho_str})")
        header = (f"{'':<19}" + "".join(f"{f'@{m}xc_H':>16}" for m in CHECKPOINT_MULTIPLIERS)
                  + f"{'lf_frac':>9}  {'n':>2}")
        print(header)
        print("-" * len(header))

        for method in METHOD_ORDER:
            per_seed = runs.get((benchmark, method), {})
            if not per_seed:
                print(f"{method:<19}" + "".join(f"{'--':>16}" for _ in checkpoints)
                      + f"{'--':>9}  {0:>2}")
                continue

            cells, n_seen = [], 0
            for budget in checkpoints:
                logs = []
                for seed in SEEDS:
                    res = per_seed.get(seed)
                    if res is None:
                        continue
                    r = regret_at_cost(res, budget, post_init_regret(res, known_opt))
                    logs.append(math.log10(max(r, REGRET_FLOOR)))
                m, se, n = mean_se(logs)
                n_seen = max(n_seen, n)
                cells.append(f"{m:.2f}+-{se:.2f}" if n else "--")
                summary[f"{benchmark}|{method}|{budget:g}"] = dict(mean=m, se=se, n=n)

            lf = [r.get("lf_fraction") for r in per_seed.values()
                  if r.get("lf_fraction") is not None]
            lf_str = f"{np.mean(lf):.2f}" if lf else "--"
            note = METHOD_NOTE.get(method, "")
            print(f"{method:<19}" + "".join(f"{c:>16}" for c in cells)
                  + f"{lf_str:>9}  {n_seen:>2}  {note}")

        print(f"{'KO-MES+DKL':<19}" + "".join(f"{'--':>16}" for _ in checkpoints)
              + f"{'--':>9}  {0:>2}  <- pending Exp B (DKL diagnostic)")

    print()
    print("Cells are log10(simple regret), mean +- 1 SE over seeds; lower is better.")
    print(f"Regret values <= {REGRET_FLOOR:g} are floored to {REGRET_FLOOR:g} "
          f"(log10 = {math.log10(REGRET_FLOOR):.0f}) -- exact zeros occur on Currin_2D.")

    out = os.path.join(EXP_DIR, "summary_table.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out}")


def plot():
    """log10(simple regret) vs post-init cost, mean +- 1 SE band per method."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load_all()
    plot_dir = os.path.join(EXP_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for benchmark in BENCHMARKS:
        hf = get_benchmark(f"{benchmark}_HF")
        c_H, known_opt = hf["cost"], hf["known_optimal_value"]
        grid = np.linspace(0, 100 * c_H, 200)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        plotted = False
        for method in METHOD_ORDER:
            per_seed = runs.get((benchmark, method), {})
            if not per_seed:
                continue
            curves = []
            for res in per_seed.values():
                fallback = post_init_regret(res, known_opt)
                vals = [max(regret_at_cost(res, g, fallback), REGRET_FLOOR) for g in grid]
                curves.append(np.log10(np.asarray(vals)))
            arr = np.vstack(curves)
            mean = arr.mean(axis=0)
            se = arr.std(axis=0, ddof=1) / math.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros_like(mean)
            ax.plot(grid, mean, label=f"{method} (n={arr.shape[0]})")
            ax.fill_between(grid, mean - se, mean + se, alpha=0.15)
            plotted = True

        if not plotted:
            plt.close(fig)
            continue
        for m in CHECKPOINT_MULTIPLIERS:
            ax.axvline(m * c_H, color="grey", ls=":", lw=0.8)
        ax.set_xlabel("post-init cost")
        ax.set_ylabel("log10(simple regret)")
        ax.set_title(f"{benchmark}: mean +- 1 SE")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(plot_dir, f"{benchmark}_regret.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
    plot()
