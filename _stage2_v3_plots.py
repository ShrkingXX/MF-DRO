"""
Log-regret vs cost plot for Stage 2 v3 (MF-DRO vs SF-DRO, fixed-100-iteration
protocol -- see _stage2_v3_worker.py's docstring for why cost is tracked
per-iteration but NOT the stopping condition here, unlike the older
_stage2_plots.py's cost_budget-based checkpoints).

Same step-interpolation-onto-a-shared-cost-grid / log10-floored-at-1e-3 /
mean+-SE-across-seeds convention as _stage2_plots.py, adapted because final
cost now varies per run (fixed iterations, not fixed budget) instead of
being a shared checkpoint grid -- the grid here spans 0 to the max final
cost observed across BOTH methods for that benchmark, and each curve holds
its last value flat past its own final cost (standard step-interpolation
behavior for a run that finished before the grid's right edge).
"""
import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CKPT_DIR = "results/mfdro_stage2_v3/checkpoints"
# Old baselines (MF-GP-UCB/MF-MI-Greedy/Greedy-MES): kept as-is from the
# pre-v3 Stage 2 run rather than rerun (their GP-fitting code is untouched
# by this session's fixes -- see _stage2_v3_orchestrate.sh's own docstring
# for the full reasoning). Two real caveats this introduces, both accepted
# as OK for a trend-level read: (1) old 1x init sizing / cost_budget-
# terminated protocol, not the new 2x sizing / fixed-100-iteration one
# MF-DRO and SF-DRO now use; (2) Ackley_10D didn't exist yet when these
# were run, so those 3 methods are simply absent from that subplot.
OLD_CKPT_DIR = "results/mfdro_stage2/checkpoints"
OUT_DIR = "results/mfdro_stage2_v3/plots"
os.makedirs(OUT_DIR, exist_ok=True)

BENCHMARKS = ["Currin_2D", "Hartmann_6D", "Borehole_8D", "Ackley_10D"]
METHODS = ["MF-DRO", "SF-DRO", "MF-GP-UCB", "MF-MI-Greedy", "Greedy-MES"]
# Greedy-MES reran at the current protocol (2x init, fixed 100 iterations) --
# reads from CKPT_DIR now, not OLD_CKPT_DIR. MF-GP-UCB/MF-MI-Greedy still at
# their old numbers (not rerun this pass).
OLD_METHODS = {"MF-GP-UCB", "MF-MI-Greedy"}
COLORS = {
    "MF-DRO": "tab:red", "SF-DRO": "tab:purple",
    "MF-GP-UCB": "tab:orange", "MF-MI-Greedy": "tab:green", "Greedy-MES": "tab:blue",
}
SEEDS = [42, 43, 44, 45, 46]


def load_curves(method, benchmark):
    ckpt_dir = OLD_CKPT_DIR if method in OLD_METHODS else CKPT_DIR
    curves = []
    for seed in SEEDS:
        path = os.path.join(ckpt_dir, f"{method}__{benchmark}__seed{seed}.json")
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        rc = d.get("hf_regret_curve") or d.get("regret_curve")
        cc = d.get("cost_curve")
        curves.append((np.array(cc), np.array(rc)))
    return curves


def step_curve_on_grid(cc, rc, grid):
    """Step-interpolate (cc, rc) onto a shared cost grid, holding the last
    value flat past the run's own final cost."""
    out = np.empty(len(grid))
    j = 0
    cur = rc[0]
    for i, g in enumerate(grid):
        while j < len(cc) and cc[j] <= g:
            cur = rc[j]
            j += 1
        out[i] = cur
    return out


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, bm in zip(axes, BENCHMARKS):
    all_max_cost = 0.0
    method_curves = {}
    for method in METHODS:
        curves = load_curves(method, bm)
        if not curves:
            continue
        method_curves[method] = curves
        all_max_cost = max(all_max_cost, max(cc[-1] for cc, rc in curves))

    grid = np.linspace(0.01, all_max_cost, 300)

    for method in METHODS:
        curves = method_curves.get(method)
        if not curves:
            continue
        curves_on_grid = np.array([step_curve_on_grid(cc, rc, grid) for cc, rc in curves])
        floor = 1e-3
        clipped = np.clip(curves_on_grid, floor, None)
        log_vals = np.log10(clipped)
        mean = log_vals.mean(axis=0)
        se = (log_vals.std(axis=0, ddof=1) / np.sqrt(len(curves))
              if len(curves) > 1 else np.zeros_like(mean))
        n_tag = f" (n={len(curves)})" if len(curves) < 5 else ""
        old_tag = "*" if method in OLD_METHODS else ""
        ax.plot(grid, mean, label=f"{method}{old_tag}{n_tag}", color=COLORS[method])
        ax.fill_between(grid, mean - se, mean + se, color=COLORS[method], alpha=0.12)

    ax.set_xlabel("Post-init cost")
    ax.set_ylabel("log10(simple regret) [floored at 1e-3]")
    ax.set_title(bm)
    ax.legend(fontsize=8)

fig.suptitle(
    "Stage 2 v3: log-regret vs cost\n"
    "MF-DRO/SF-DRO/Greedy-MES: 100 fixed iterations, 2.0x init sizing, cost tracked not budget-gated  |  "
    "*MF-GP-UCB/MF-MI-Greedy: OLD 1x init sizing, cost_budget-terminated (kept as-is, not rerun; absent on Ackley_10D)",
    fontsize=9,
)
fig.tight_layout()
out_path = os.path.join(OUT_DIR, "logregret_vs_cost.png")
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Saved {out_path}")
