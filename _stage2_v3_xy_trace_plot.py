"""
Plot x_t_trace/y_t_trace (raw per-query observed values, not the running-max
regret curve) for every method that actually saves them. Checked against
the real saved keys: only MF-DRO and Greedy-MES include x_t_trace/y_t_trace
in this codebase -- SF-DRO, MF-GP-UCB, and MF-MI-Greedy's result dicts don't
save per-query trajectories at all (confirmed by direct inspection), so
they're excluded here rather than silently faked.

y_t_trace vs iteration index, colored by fidelity (H=solid/dark, L=lighter/
open), one column per benchmark, one row per method, all 5 seeds overlaid.
Horizontal dashed line = the benchmark's true optimal value (negated to
match y_t's maximization-ready scale, since f_hf/f_lf in this codebase
return negated raw objective values).
"""
import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGE2_V3_DIR = "results/mfdro_stage2_v3/checkpoints"
OUT_DIR = "results/mfdro_stage2_v3/plots"
os.makedirs(OUT_DIR, exist_ok=True)

BENCHMARKS = ["Currin_2D", "Hartmann_6D", "Borehole_8D", "Ackley_10D"]
METHODS = ["MF-DRO", "Greedy-MES"]
SEEDS = [42, 43, 44, 45, 46]
SEED_COLORS = plt.cm.tab10.colors

# Negated known_optimal_value (raw registry value is pre-negation/minimize-
# convention; f_hf/y_t are already negated to maximization scale).
OPTIMAL_Y = {
    "Currin_2D": 13.798722044512703,
    "Hartmann_6D": 3.32237,
    "Borehole_8D": 309.5755876604079,
    "Ackley_10D": 0.0,
}


def load_run(method, bm, seed):
    path = os.path.join(STAGE2_V3_DIR, f"{method}__{bm}__seed{seed}.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path))


fig, axes = plt.subplots(len(METHODS), len(BENCHMARKS), figsize=(20, 8), sharex="col")

for row, method in enumerate(METHODS):
    for col, bm in enumerate(BENCHMARKS):
        ax = axes[row, col]
        for i, seed in enumerate(SEEDS):
            d = load_run(method, bm, seed)
            if d is None or "y_t_trace" not in d:
                continue
            y_t = np.array(d["y_t_trace"])
            fid = np.array(d["fidelity_trace"])
            iters = np.arange(len(y_t))
            color = SEED_COLORS[i % len(SEED_COLORS)]
            hf_mask = fid == 1
            lf_mask = fid == 0
            ax.scatter(iters[hf_mask], y_t[hf_mask], color=color, s=14, marker="o",
                       label=f"seed{seed}" if col == 0 else None, alpha=0.85)
            ax.scatter(iters[lf_mask], y_t[lf_mask], facecolors="none", edgecolors=color,
                       s=10, marker="o", alpha=0.5)

        ax.axhline(OPTIMAL_Y[bm], color="black", linestyle="--", linewidth=1, alpha=0.6)
        if row == 0:
            ax.set_title(bm)
        if col == 0:
            ax.set_ylabel(f"{method}\ny_t (observed value)")
        if row == len(METHODS) - 1:
            ax.set_xlabel("iteration")
        ax.grid(alpha=0.2)

axes[0, 0].legend(fontsize=7, loc="lower right", ncol=1)
fig.suptitle("x_t/y_t trace: raw observed values per query, filled=HF / open=LF, "
             "dashed=true optimum (negated to maximize scale)\n"
             "(SF-DRO, MF-GP-UCB, MF-MI-Greedy don't save per-query traces in this codebase -- not plotted)")
fig.tight_layout()
out_path = os.path.join(OUT_DIR, "xy_trace.png")
fig.savefig(out_path, dpi=140)
plt.close(fig)
print(f"Saved {out_path}")
