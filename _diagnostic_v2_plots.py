"""
Plots for the diagnostic_v2 4-variant x 3-seed study (results/mfdro_diagnostic_v2/).
Palette: dataviz skill's validated default categorical slots 1-4 (blue, orange,
aqua, yellow) -- one fixed hue per variant, used consistently across every figure.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

CKPT = "results/mfdro_diagnostic_v2/checkpoints"
OUT = "results/mfdro_diagnostic_v2/plots"
os.makedirs(OUT, exist_ok=True)

VARIANTS = ["BASELINE", "INIT_ONLY", "INIT_RTG", "FULL"]
LABELS = {"BASELINE": "BASELINE", "INIT_ONLY": "INIT", "INIT_RTG": "INIT+RTG", "FULL": "FULL"}
SEEDS = [42, 43, 44]
COLOR = {  # dataviz skill default categorical slots 1-4, light mode
    "BASELINE": "#2a78d6",   # blue
    "INIT_ONLY": "#eb6834",  # orange
    "INIT_RTG": "#1baf7a",   # aqua
    "FULL": "#eda100",       # yellow
}
SEED_STYLE = {42: "-", 43: "--", 44: ":"}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dc"
TRUE_OPT_LINE = "#9a9890"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_PRIMARY,
    "text.color": TEXT_PRIMARY, "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
})

data = {}
for v in VARIANTS:
    data[v] = {}
    for s in SEEDS:
        with open(f"{CKPT}/{v}__Hartmann_6D__seed{s}.json") as f:
            r = json.load(f)
        with open(f"{CKPT}/{v}__Hartmann_6D__seed{s}.diag.json") as f:
            d = json.load(f)
        data[v][s] = dict(result=r, diag=d)

# ════════════════════════════════════
# FIG 1: Regret-vs-cost, small multiples (one panel per variant, 3 seed lines)
# ════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
for ax, v in zip(axes.flat, VARIANTS):
    for s in SEEDS:
        rec = data[v][s]["result"]
        ax.step(rec["cost_curve"], rec["hf_regret_curve"], where="post",
                color=COLOR[v], linestyle=SEED_STYLE[s], linewidth=1.6,
                alpha=0.9, label=f"seed {s}")
    ax.set_title(LABELS[v], color=COLOR[v])
    ax.set_xlim(0, 400)
    ax.grid(True, linewidth=0.8)
for ax in axes[-1, :]:
    ax.set_xlabel("Cumulative HF+LF cost")
for ax in axes[:, 0]:
    ax.set_ylabel("HF regret")
axes[0, 0].legend(loc="upper right", frameon=False, fontsize=8, title="seed", title_fontsize=8)
fig.suptitle("Incumbent freeze: regret flatlines almost immediately in every run",
             fontsize=13, fontweight="bold", y=1.02)
fig.text(0.5, -0.02,
         "Line style = seed (solid 42 / dashed 43 / dotted 44). Regret is non-increasing by construction\n"
         "(best_hf can only grow); a flat line after the first few cost units means no HF query has improved\n"
         "the incumbent since. All 12 runs show this pattern, most within the first ~20-30 cost units of 400.",
         ha="center", fontsize=8.5, color=TEXT_SECONDARY)
fig.savefig(f"{OUT}/fig1_regret_small_multiples.png")
plt.close(fig)

# ════════════════════════════════════
# FIG 2: Variant comparison -- mean regret-vs-cost with seed min/max band, one panel
# ════════════════════════════════════
cost_grid = np.arange(0, 401, 2.0)


def step_eval(cost_curve, regret_curve, grid):
    cost_curve = np.asarray(cost_curve)
    regret_curve = np.asarray(regret_curve)
    idx = np.searchsorted(cost_curve, grid, side="right") - 1
    idx = np.clip(idx, 0, len(regret_curve) - 1)
    out = regret_curve[idx]
    out[grid < cost_curve[0]] = regret_curve[0]
    return out


fig, ax = plt.subplots(figsize=(7.5, 5.5))
for v in VARIANTS:
    curves = np.stack([
        step_eval(data[v][s]["result"]["cost_curve"], data[v][s]["result"]["hf_regret_curve"], cost_grid)
        for s in SEEDS
    ])
    mean_c, min_c, max_c = curves.mean(axis=0), curves.min(axis=0), curves.max(axis=0)
    ax.plot(cost_grid, mean_c, color=COLOR[v], linewidth=2.2, label=LABELS[v])
    ax.fill_between(cost_grid, min_c, max_c, color=COLOR[v], alpha=0.12, linewidth=0)
ax.set_xlabel("Cumulative HF+LF cost")
ax.set_ylabel("HF regret (mean across 3 seeds, band = seed min-max)")
ax.set_xlim(0, 400)
ax.set_title("No variant meaningfully outperforms BASELINE by cost=400")
ax.legend(loc="upper right", frameon=False, fontsize=9)
fig.savefig(f"{OUT}/fig2_variant_comparison.png")
plt.close(fig)

# ════════════════════════════════════
# FIG 3: two-panel -- (a) REAL P[choose HF] at inference and (b) neg_rtg_frac, vs iteration
# fid_mean_per_iter (rollout-batch mean during _train_dt) is a DIFFERENT
# quantity from fidelity_trace (the actual real-query decision) -- the two
# diverge sharply (see fig5), so panel (a) uses fidelity_trace, the ground
# truth of what fidelity was actually queried.
# ════════════════════════════════════


def smooth(y, k=15):
    y = np.asarray(y, dtype=float)
    if len(y) < k:
        return y
    kernel = np.ones(k) / k
    pad = k // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(yp, kernel, mode="valid")[: len(y)]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
for v in VARIANTS:
    max_len = max(len(data[v][s]["result"]["fidelity_trace"]) for s in SEEDS)
    p_stack, n_stack = [], []
    for s in SEEDS:
        rec = data[v][s]["result"]
        p = smooth(rec["fidelity_trace"])
        n = smooth(rec["neg_rtg_frac_per_iter"])
        p_stack.append(np.pad(p, (0, max_len - len(p)), constant_values=np.nan))
        n_stack.append(np.pad(n, (0, max_len - len(n)), constant_values=np.nan))
    p_mean = np.nanmean(np.stack(p_stack), axis=0)
    n_mean = np.nanmean(np.stack(n_stack), axis=0)
    it = np.arange(max_len)
    ax1.plot(it, p_mean, color=COLOR[v], linewidth=2.0, label=LABELS[v])
    ax2.plot(it, n_mean, color=COLOR[v], linewidth=2.0, label=LABELS[v])

ax1.set_xlabel("Iteration")
ax1.set_ylabel("Actual P[query HF]  (15-iter rolling mean of real choices)")
ax1.set_ylim(-0.02, 1.02)
ax1.set_title("Real queries collapse to LF-only within ~20 iterations")
ax1.legend(loc="upper right", frameon=False, fontsize=8)

ax2.set_xlabel("Iteration")
ax2.set_ylabel("Fraction of batch with RTG < 0  (15-iter rolling mean)")
ax2.set_ylim(-0.02, 1.02)
ax2.set_title("RTG stays miscalibrated regardless of grounding")
ax2.legend(loc="lower right", frameon=False, fontsize=8)

fig.suptitle("Two co-occurring failure signals (mean across 3 seeds per variant)",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{OUT}/fig3_fidelity_and_rtg.png", bbox_inches="tight")
plt.close(fig)

# ════════════════════════════════════
# FIG 5: train/inference mismatch on the fidelity head, one panel per variant
# fid_mean_per_iter = mean P[HF] the fidelity head assigns on SIMULATED
# rollout-batch states during _train_dt (rises over training); fidelity_trace
# = what was actually queried in reality (collapses to ~0). Same [0,1]
# probability scale, so both belong on one axis -- this is not the dual-axis
# antipattern, it's two measurements of the same quantity on two data sources.
# ════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
for ax, v in zip(axes, VARIANTS):
    max_len = max(len(data[v][s]["result"]["fidelity_trace"]) for s in SEEDS)
    train_stack, real_stack = [], []
    for s in SEEDS:
        rec = data[v][s]["result"]
        tr = smooth(rec["fid_mean_per_iter"])
        re = smooth(rec["fidelity_trace"])
        train_stack.append(np.pad(tr, (0, max_len - len(tr)), constant_values=np.nan))
        real_stack.append(np.pad(re, (0, max_len - len(re)), constant_values=np.nan))
    train_mean = np.nanmean(np.stack(train_stack), axis=0)
    real_mean = np.nanmean(np.stack(real_stack), axis=0)
    it = np.arange(max_len)
    ax.plot(it, train_mean, color=COLOR[v], linewidth=2.0, linestyle="--",
            label="training batch (simulated)")
    ax.plot(it, real_mean, color=COLOR[v], linewidth=2.2, linestyle="-",
            label="real queries (actual)")
    ax.set_title(LABELS[v], color=COLOR[v])
    ax.set_xlabel("Iteration")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linewidth=0.8)
axes[0].set_ylabel("P[HF]")
axes[0].legend(loc="upper left", frameon=False, fontsize=7.5)
fig.suptitle("Train/inference mismatch: the fidelity head learns HF is worth trying "
             "on simulated rollouts, but the deployed policy essentially never queries HF for real",
             fontsize=11.5, fontweight="bold", y=1.06)
fig.tight_layout()
fig.savefig(f"{OUT}/fig5_train_inference_gap.png", bbox_inches="tight")
plt.close(fig)

# ════════════════════════════════════
# FIG 4: two-panel grouped bars -- init_max_hf and final regret, by seed x variant
# ════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
x = np.arange(len(SEEDS))
width = 0.19
for i, v in enumerate(VARIANTS):
    offs = (i - 1.5) * width
    init_vals = [data[v][s]["diag"]["init_max_hf"] for s in SEEDS]
    final_vals = [data[v][s]["result"]["hf_regret_curve"][-1] for s in SEEDS]
    ax1.bar(x + offs, init_vals, width=width, color=COLOR[v], label=LABELS[v])
    ax2.bar(x + offs, final_vals, width=width, color=COLOR[v], label=LABELS[v])

for ax, ylabel, title in [
    (ax1, "max HF value seen at init", "Init quality by seed"),
    (ax2, "final HF regret (lower is better)", "Final outcome by seed"),
]:
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linewidth=0.8)
    ax.grid(False, axis="x")
ax1.set_ylim(0, 3.6)
ax1.axhline(3.322, color=TRUE_OPT_LINE, linestyle="--", linewidth=1.2)
ax1.text(2.55, 3.322, " true optimum", fontsize=7.5, color=TRUE_OPT_LINE, va="bottom", ha="right")

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.90),
           ncol=4, frameon=False, fontsize=9)

fig.suptitle("Seed 44 is an outlier across every variant -- an init-luck effect, not a fix effect",
             fontsize=12, fontweight="bold", y=1.02)
fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig(f"{OUT}/fig4_seed_breakdown.png", bbox_inches="tight")
plt.close(fig)

print("Saved:")
for f in sorted(os.listdir(OUT)):
    print(f"  {OUT}/{f}")
