"""
Stage 2 train/inference gap for MF-DRO (the only Stage 2 method with a
fid_mean_per_iter field, since it's the only one with a Decision
Transformer): fid_mean_per_iter (mean P[HF] the fidelity head assigns on
SIMULATED rollout-batch states during _train_dt) vs fidelity_trace (what
was actually queried in reality) -- same construction as diagnostic_v2's
fig5, run across all 3 Stage 2 benchmarks x 5 seeds.
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

BENCHMARKS = ["Currin_2D", "Hartmann_6D", "Borehole_8D"]
SEEDS = [42, 43, 44, 45, 46]
COLOR = "#2a78d6"  # MF-DRO's slot from the Stage 2 audit palette
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


def smooth(y, k=15):
    y = np.asarray(y, dtype=float)
    if len(y) < k:
        return y
    kernel = np.ones(k) / k
    pad = k // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(yp, kernel, mode="valid")[: len(y)]


data = {}
for bm in BENCHMARKS:
    runs = []
    for s in SEEDS:
        path = f"{CKPT}/MF-DRO__{bm}__seed{s}.json"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        runs.append(d)
    data[bm] = runs

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
for ax, bm in zip(axes, BENCHMARKS):
    runs = data[bm]
    max_len = max(len(r["fidelity_trace"]) for r in runs)
    train_stack, real_stack = [], []
    for r in runs:
        tr = smooth(r["fid_mean_per_iter"])
        re = smooth(r["fidelity_trace"])
        train_stack.append(np.pad(tr, (0, max_len - len(tr)), constant_values=np.nan))
        real_stack.append(np.pad(re, (0, max_len - len(re)), constant_values=np.nan))
    train_mean = np.nanmean(np.stack(train_stack), axis=0)
    real_mean = np.nanmean(np.stack(real_stack), axis=0)
    it = np.arange(max_len)
    ax.plot(it, train_mean, color=COLOR, linewidth=2.0, linestyle="--",
            label="training batch (simulated)")
    ax.plot(it, real_mean, color=COLOR, linewidth=2.4, linestyle="-",
            label="real queries (actual)")
    ax.set_title(f"{bm}  (n={len(runs)} seeds)")
    ax.set_xlabel("Iteration")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linewidth=0.8)
axes[0].set_ylabel("P[HF]")
axes[0].legend(loc="upper right", frameon=False, fontsize=8.5)
fig.suptitle("MF-DRO's train/inference gap holds across all 3 Stage 2 benchmarks:\n"
             "the fidelity head learns HF looks worthwhile on simulated rollouts, "
             "but real queries abandon HF almost immediately",
             fontsize=12, fontweight="bold", y=1.12)
fig.tight_layout()
fig.savefig(f"{OUT}/audit_fig3_train_inference_gap.png", bbox_inches="tight")
plt.close(fig)

print(f"Saved {OUT}/audit_fig3_train_inference_gap.png")
for bm in BENCHMARKS:
    runs = data[bm]
    train_final = np.mean([np.mean(r["fid_mean_per_iter"][-20:]) for r in runs])
    real_final = np.mean([np.mean(r["fidelity_trace"][-20:]) for r in runs])
    real_overall = np.mean([np.mean(r["fidelity_trace"]) for r in runs])
    print(f"{bm}: last-20-iter mean P[HF] -- training={train_final:.3f}  real={real_final:.3f}  "
          f"(real P[HF] over whole run={real_overall:.3f})")
