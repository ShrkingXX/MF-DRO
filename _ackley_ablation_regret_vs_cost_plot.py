"""
Regret-vs-cost plot for the Ackley_10D MF-DRO ablation (results/mfdro_ackley_test/):
MF-DRO vs Greedy-MES vs MF-MI-Greedy vs MF-GP-UCB, 3 seeds each (42,43,44), at the
registered 5:1 HF:LF cost ratio, cost_budget=500 (post-init).

Reads the final .json checkpoints (not the .mf.json intermediate/duplicate files
MF-DRO also happens to write alongside them). Field names differ by method: MF-DRO's
result dict uses hf_regret_curve, baselines use regret_curve -- both are "best HF
regret so far" step functions over post-init cost (cost_curve, common to all methods).

Since each seed's cost_curve lands on different actual cost values (methods mix
HF/LF queries at different rates -- e.g. Greedy-MES's per-query cost isn't a clean
multiple of 5 like MF-DRO's near-all-HF trace), curves are step-interpolated
(previous-value-hold, matching "best regret so far is right-continuous in cost")
onto a shared cost grid before averaging across seeds.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_DIR = "results/mfdro_ackley_test/checkpoints"
BENCHMARK = "Ackley_10D"
SEEDS = [42, 43, 44]
METHODS = ["MF-DRO", "Greedy-MES", "MF-MI-Greedy", "MF-GP-UCB"]
COST_BUDGET = 500
N_GRID = 200

COLORS = {
    "MF-DRO": "#d62728",
    "Greedy-MES": "#1f77b4",
    "MF-MI-Greedy": "#2ca02c",
    "MF-GP-UCB": "#9467bd",
}


def load_curve(method, seed):
    path = os.path.join(EXP_DIR, f"{method}__{BENCHMARK}__seed{seed}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    regret_key = "hf_regret_curve" if "hf_regret_curve" in d else "regret_curve"
    cost = d["cost_curve"]
    regret = d[regret_key]
    n = min(len(cost), len(regret))
    return np.array(cost[:n], dtype=float), np.array(regret[:n], dtype=float)


def step_interp(grid, cost, regret):
    """Previous-value-hold onto `grid`. Grid points before the first real
    post-init query (cost < cost[0]) hold regret[0] as a stand-in for the
    initial-design regret, which isn't separately recorded."""
    raw_idx = np.searchsorted(cost, grid, side="right") - 1
    idx = np.clip(raw_idx, 0, len(regret) - 1)
    out = regret[idx]
    out[raw_idx < 0] = regret[0]
    return out


grid = np.linspace(0, COST_BUDGET, N_GRID)

fig, ax = plt.subplots(figsize=(9, 6))
any_data = False

for method in METHODS:
    curves = []
    for seed in SEEDS:
        loaded = load_curve(method, seed)
        if loaded is not None:
            curves.append(loaded)
        else:
            print(f"  {method} seed{seed}: missing, skipping")
    if not curves:
        print(f"  {method}: no data found, skipping entirely")
        continue
    any_data = True

    interp_curves = np.stack([step_interp(grid, c, r) for c, r in curves])
    means = interp_curves.mean(axis=0)
    if len(curves) > 1:
        ses = interp_curves.std(axis=0, ddof=1) / np.sqrt(len(curves))
    else:
        ses = np.zeros_like(means)

    label = f"{method} (n={len(curves)} seed{'s' if len(curves) != 1 else ''}, final={means[-1]:.3f})"
    ax.plot(grid, means, label=label, linewidth=1.8, color=COLORS.get(method))
    ax.fill_between(grid, means - ses, means + ses, alpha=0.15, color=COLORS.get(method))

if not any_data:
    raise SystemExit("No data found for any method -- check results/mfdro_ackley_test/checkpoints/")

ax.set_xlabel("Cost (post-init; c_H=5, c_L=1)")
ax.set_ylabel("Simple regret (best HF value found)")
ax.set_title(f"{BENCHMARK} -- MF-DRO ablation, regret vs. cost (5:1 HF:LF ratio, 3 seeds)")
ax.legend(fontsize=9, loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()

out_dir = "results/mfdro_ackley_test/plots"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"{BENCHMARK}_regret_vs_cost.png")
fig.savefig(out_path, dpi=150)
print(f"Saved {out_path}")

# --- log-scale y-axis version (same data, same grid/interpolation above) ---
ax.set_yscale("log")
ax.set_title(f"{BENCHMARK} -- MF-DRO ablation, log regret vs. cost (5:1 HF:LF ratio, 3 seeds)")
ax.grid(which="both", alpha=0.3)
fig.tight_layout()
out_path_log = os.path.join(out_dir, f"{BENCHMARK}_log_regret_vs_cost.png")
fig.savefig(out_path_log, dpi=150)
plt.close(fig)
print(f"Saved {out_path_log}")
