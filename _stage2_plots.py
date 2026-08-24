import json
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRATCH = "/private/tmp/claude-501/-Users-yurucui-Desktop-DRO-Code-DRO-aistats-submission/98fb03cd-2518-4785-a528-d0101b191c5a/scratchpad"
with open(os.path.join(SCRATCH, "stage2_results.pkl"), "rb") as f:
    all_results = pickle.load(f)

OUT_DIR = "results/mfdro_stage2/plots"
os.makedirs(OUT_DIR, exist_ok=True)

BENCHMARKS = {
    "Currin_2D":   dict(cH=3.0, checkpoints=[30, 90, 300]),
    "Hartmann_6D": dict(cH=8.0, checkpoints=[80, 240, 800]),
    "Borehole_8D": dict(cH=2.0, checkpoints=[20, 60, 200]),
}
METHODS = ["MF-DRO", "Greedy-MES", "MF-MI-Greedy", "MF-GP-UCB", "SF-DRO"]
COLORS = {
    "MF-DRO": "tab:red", "Greedy-MES": "tab:blue", "MF-MI-Greedy": "tab:green",
    "MF-GP-UCB": "tab:orange", "SF-DRO": "tab:purple",
}


def step_curve_on_grid(cc, rc, grid):
    """Step-interpolate (cc, rc) onto a shared cost grid."""
    out = np.empty(len(grid))
    j = 0
    cur = rc[0]
    for i, g in enumerate(grid):
        while j < len(cc) and cc[j] <= g:
            cur = rc[j]
            j += 1
        out[i] = cur
    return out


# ==== PLOT 1: Regret vs post-init cost, per benchmark ====
for bm, spec in BENCHMARKS.items():
    cH = spec["cH"]
    checkpoints = spec["checkpoints"]
    max_cost = checkpoints[-1]
    grid = np.linspace(0.01, max_cost, 200)

    fig, ax = plt.subplots(figsize=(7, 5))
    for method in METHODS:
        curves = all_results[bm][method]
        if not curves:
            continue
        curves_on_grid = np.array([step_curve_on_grid(cc, rc, grid) for cc, rc in curves])
        # log10 requires positive regret; clip tiny/negative values to a
        # small positive floor purely for the log-scale plot (annotated).
        floor = 1e-3
        clipped = np.clip(curves_on_grid, floor, None)
        log_vals = np.log10(clipped)
        mean = log_vals.mean(axis=0)
        se = log_vals.std(axis=0, ddof=1) / np.sqrt(len(curves)) if len(curves) > 1 else np.zeros_like(mean)
        n_tag = f" (n={len(curves)})" if len(curves) < 5 else ""
        ax.plot(grid, mean, label=f"{method}{n_tag}", color=COLORS[method])
        ax.fill_between(grid, mean - se, mean + se, color=COLORS[method], alpha=0.15)

    for c in checkpoints:
        ax.axvline(c, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Post-init cumulative cost")
    ax.set_ylabel("log10(simple regret)  [floored at 1e-3]")
    ax.set_title(f"{bm}: regret vs. post-init cost")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f"{bm}_regret.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

# ==== PLOT 2: MF-DRO fidelity trace, per benchmark, one subplot per seed ====
for bm in BENCHMARKS:
    dro_dir = "results/mfdro_stage2/checkpoints"
    seeds = [42, 43, 44, 45, 46]
    fig, axes = plt.subplots(1, 5, figsize=(18, 3), sharey=True)
    for ax, seed in zip(axes, seeds):
        path = os.path.join(dro_dir, f"MF-DRO__{bm}__seed{seed}.json")
        if not os.path.exists(path):
            ax.set_title(f"seed{seed} (missing)")
            ax.axis("off")
            continue
        with open(path) as f:
            d = json.load(f)
        cc = d["cost_curve"]
        fid = d["fidelity_trace"]
        ax.scatter(cc, fid, s=4, alpha=0.5)
        ax.set_title(f"seed{seed}")
        ax.set_ylim(-0.3, 1.3)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["L", "H"])
        ax.set_xlabel("post-init cost")
    fig.suptitle(f"{bm}: MF-DRO fidelity trace by seed")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f"{bm}_fidelity.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

print("\nAll plots generated.")
