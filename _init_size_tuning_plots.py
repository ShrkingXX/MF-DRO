"""
Log-regret vs cost for the initial_hf/initial_lf tuning sweep
(_init_size_tuning_worker.py: 0.5x/1x/2x the Song-2019 3d/5d sizing,
3 benchmarks x 3 seeds, fixed 30-iteration horizon, cost_budget=9999 i.e.
not budget-terminated -- cost varies by run since fidelity mix varies).
Same step-interpolation / log10-floored-at-1e-3 / mean+-SE-across-seeds
convention as the Stage 2 v3 plots.
"""
import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CKPT_DIR = "results/mfdro_init_size_tuning/checkpoints"
OUT_DIR = "results/mfdro_init_size_tuning/plots"
os.makedirs(OUT_DIR, exist_ok=True)

BENCHMARKS = ["Currin_2D", "Hartmann_6D", "Borehole_8D"]
MULTS = ["0.5", "1.0", "2.0"]
COLORS = {"0.5": "tab:blue", "1.0": "tab:gray", "2.0": "tab:red"}
SEEDS = [42, 43, 44]


def load_curves(bm, mult):
    curves = []
    for seed in SEEDS:
        path = os.path.join(CKPT_DIR, f"{bm}__mult{mult}__seed{seed}.json")
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        curves.append((np.array(d["cost_curve"]), np.array(d["hf_regret_curve"])))
    return curves


def step_curve_on_grid(cc, rc, grid):
    out = np.empty(len(grid))
    j = 0
    cur = rc[0]
    for i, g in enumerate(grid):
        while j < len(cc) and cc[j] <= g:
            cur = rc[j]
            j += 1
        out[i] = cur
    return out


fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

for ax, bm in zip(axes, BENCHMARKS):
    mult_curves = {mult: load_curves(bm, mult) for mult in MULTS}
    all_max_cost = max(cc[-1] for curves in mult_curves.values() for cc, rc in curves)
    grid = np.linspace(0.01, all_max_cost, 200)

    for mult in MULTS:
        curves = mult_curves[mult]
        if not curves:
            continue
        curves_on_grid = np.array([step_curve_on_grid(cc, rc, grid) for cc, rc in curves])
        clipped = np.clip(curves_on_grid, 1e-3, None)
        log_vals = np.log10(clipped)
        mean = log_vals.mean(axis=0)
        se = (log_vals.std(axis=0, ddof=1) / np.sqrt(len(curves))
              if len(curves) > 1 else np.zeros_like(mean))
        label = f"{mult}x" + (" (current)" if mult == "1.0" else "")
        ax.plot(grid, mean, label=label, color=COLORS[mult])
        ax.fill_between(grid, mean - se, mean + se, color=COLORS[mult], alpha=0.15)

    ax.set_xlabel("Cost (fixed 30-iter horizon, not budget-gated)")
    ax.set_ylabel("log10(simple regret) [floored at 1e-3]")
    ax.set_title(bm)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

fig.suptitle("initial_hf/initial_lf sizing ablation: log-regret vs cost "
             "(0.5x/1x/2x Song-2019 3d/5d convention, 3 seeds, MF-DRO)")
fig.tight_layout()
out_path = os.path.join(OUT_DIR, "logregret_vs_cost.png")
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Saved {out_path}")
