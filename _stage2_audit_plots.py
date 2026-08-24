"""
Stage 2 intermediate-metrics audit (5 methods x 3 benchmarks x 5 seeds,
results/mfdro_stage2/checkpoints/). Reads the JSON checkpoints directly
(not the old scratchpad pickle _stage2_plots.py depended on). Reuses the
established per-benchmark checkpoint convention (10x/30x/100x c_H cost
units) already used for this comparison earlier this session.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CKPT = "results/mfdro_stage2/checkpoints"
OUT = "results/mfdro_stage2/plots"
os.makedirs(OUT, exist_ok=True)

BENCHMARKS = {
    "Currin_2D":   dict(cH=3.0, checkpoints=[30, 90, 300]),
    "Hartmann_6D": dict(cH=8.0, checkpoints=[80, 240, 800]),
    "Borehole_8D": dict(cH=2.0, checkpoints=[20, 60, 200]),
}
METHODS = ["MF-DRO", "Greedy-MES", "MF-MI-Greedy", "MF-GP-UCB", "SF-DRO"]
SEEDS = [42, 43, 44, 45, 46]
COLOR = {  # dataviz skill default categorical slots 1-5, light mode
    "MF-DRO": "#2a78d6",        # blue -- the paper's proposed method
    "Greedy-MES": "#eb6834",    # orange
    "MF-MI-Greedy": "#1baf7a",  # aqua
    "MF-GP-UCB": "#eda100",     # yellow
    "SF-DRO": "#e87ba4",        # magenta
}
LINEWIDTH = {"MF-DRO": 2.6}  # emphasize the method under audit; others default below
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dc"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b", "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def regret_of(d):
    return d.get("hf_regret_curve", d.get("regret_curve"))


def step_eval(cost_curve, regret_curve, grid):
    cost_curve = np.asarray(cost_curve)
    regret_curve = np.asarray(regret_curve)
    idx = np.searchsorted(cost_curve, grid, side="right") - 1
    idx = np.clip(idx, 0, len(regret_curve) - 1)
    out = regret_curve[idx].astype(float)
    out[grid < cost_curve[0]] = regret_curve[0]
    return out


def regret_at_cost(cost_curve, regret_curve, c_ref):
    idx = None
    for i, c in enumerate(cost_curve):
        if c <= c_ref:
            idx = i
        else:
            break
    return regret_curve[idx] if idx is not None else regret_curve[0]


data = {}
for bm in BENCHMARKS:
    data[bm] = {}
    for m in METHODS:
        runs = []
        for s in SEEDS:
            path = f"{CKPT}/{m}__{bm}__seed{s}.json"
            if not os.path.exists(path):
                continue
            with open(path) as f:
                d = json.load(f)
            runs.append((d["cost_curve"], regret_of(d)))
        data[bm][m] = runs

# ════════════════════════════════════
# FIG S1: regret-vs-cost, log10 scale (floored), 5 methods overlaid, per benchmark
# ════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
FLOOR = 1e-2
for ax, (bm, spec) in zip(axes, BENCHMARKS.items()):
    checkpoints = spec["checkpoints"]
    grid = np.linspace(0.5, checkpoints[-1], 300)
    for m in METHODS:
        runs = data[bm][m]
        if not runs:
            continue
        curves = np.stack([step_eval(cc, rc, grid) for cc, rc in runs])
        clipped = np.clip(curves, FLOOR, None)
        log_curves = np.log10(clipped)
        mean, se = log_curves.mean(axis=0), log_curves.std(axis=0, ddof=1) / np.sqrt(len(runs))
        ax.plot(grid, mean, color=COLOR[m], linewidth=LINEWIDTH.get(m, 1.8), label=m)
        ax.fill_between(grid, mean - se, mean + se, color=COLOR[m], alpha=0.12, linewidth=0)
    for c in checkpoints:
        ax.axvline(c, color=TEXT_SECONDARY, linestyle=":", linewidth=0.9, alpha=0.6)
    ax.set_title(bm)
    ax.set_xlabel("Post-init cumulative cost")
    ax.grid(True, linewidth=0.8)
axes[0].set_ylabel(f"log10(HF regret)  [floored at {FLOOR}]")
axes[0].legend(loc="upper right", frameon=False, fontsize=8)
fig.suptitle("MF-DRO (bold blue) flatlines from the first checkpoint onward on every benchmark;\n"
             "Greedy-MES and MF-MI-Greedy keep improving through all three checkpoints",
             fontsize=12.5, fontweight="bold", y=1.08)
fig.tight_layout()
fig.savefig(f"{OUT}/audit_fig1_regret_vs_cost.png", bbox_inches="tight")
plt.close(fig)

# ════════════════════════════════════
# FIG S2: checkpoint-value grouped bars, one panel per benchmark (own y-scale
# each -- regret magnitudes are not comparable across benchmarks)
# ════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
n_methods = len(METHODS)
width = 0.15
for ax, (bm, spec) in zip(axes, BENCHMARKS.items()):
    checkpoints = spec["checkpoints"]
    x = np.arange(len(checkpoints))
    for i, m in enumerate(METHODS):
        runs = data[bm][m]
        vals = []
        for c in checkpoints:
            per_seed = [regret_at_cost(cc, rc, c) for cc, rc in runs]
            vals.append(np.mean(per_seed) if per_seed else np.nan)
        offs = (i - (n_methods - 1) / 2) * width
        ax.bar(x + offs, vals, width=width, color=COLOR[m], label=m,
               edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"cost={c}\n({int(c/spec['cH'])}x c_H)" for c in checkpoints], fontsize=8.5)
    ax.axhline(0, color=GRID, linewidth=1.0)
    ax.set_title(bm)
    ax.set_ylabel("mean HF regret across 5 seeds")
    ax.grid(True, axis="y", linewidth=0.8)
    ax.grid(False, axis="x")
axes[0].legend(loc="upper right", frameon=False, fontsize=8)
fig.suptitle("Same-height bars across checkpoints = no progress after the first one.\n"
             "MF-DRO and MF-GP-UCB are flat on every benchmark; SF-DRO is flat only on Hartmann_6D.",
             fontsize=12.5, fontweight="bold", y=1.08)
fig.tight_layout()
fig.savefig(f"{OUT}/audit_fig2_checkpoint_bars.png", bbox_inches="tight")
plt.close(fig)

# Print the exact audited numbers for the written report
print("Checkpoint regret table (mean across available seeds):")
for bm, spec in BENCHMARKS.items():
    print(f"\n{bm}:")
    for m in METHODS:
        runs = data[bm][m]
        n = len(runs)
        vals = [np.mean([regret_at_cost(cc, rc, c) for cc, rc in runs]) for c in spec["checkpoints"]]
        frozen = np.allclose(vals, vals[0], atol=1e-6)
        print(f"  {m:14s} n={n}  vals={[round(v,3) for v in vals]}  FROZEN={frozen}")

print("\nSaved:")
for f in sorted(os.listdir(OUT)):
    print(f"  {OUT}/{f}")
