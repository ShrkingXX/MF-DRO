"""
Real-query distance to the true optimum x*, vs. cost, per seed -- the two
benchmarks with a registered known_optimal_x (Hartmann_6D; Ackley_10D via
_stage2_v3_worker.py's explicit KNOWN_OPTIMAL_X_OVERRIDE). Currin_2D/
Borehole_8D have no known optimum location, so query_dist_to_xstar_per_iter
is empty for them (not plotted).
"""
import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CKPT_DIR = "results/mfdro_stage2_v3/checkpoints"
OUT_DIR = "results/mfdro_stage2_v3/plots"
os.makedirs(OUT_DIR, exist_ok=True)

BENCHMARKS = ["Hartmann_6D", "Ackley_10D"]
SEEDS = [42, 43, 44, 45, 46]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, bm in zip(axes, BENCHMARKS):
    for seed in SEEDS:
        path = os.path.join(CKPT_DIR, f"MF-DRO__{bm}__seed{seed}.json")
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        cc = d["cost_curve"]
        dist = d.get("query_dist_to_xstar_per_iter")
        if not dist:
            continue
        rc = d["hf_regret_curve"]
        n_improved = sum(1 for i in range(1, len(rc)) if rc[i] < rc[i - 1] - 1e-12)
        frozen = n_improved == 0
        style = "--" if frozen else "-"
        label = f"seed{seed}" + (" (frozen)" if frozen else " (improved)")
        ax.plot(cc, dist, style, label=label, alpha=0.85, linewidth=1.5)

    ax.set_xlabel("Post-init cost")
    ax.set_ylabel("L2 distance from real query to x* (normalized [0,1]^d)")
    ax.set_title(bm)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

fig.suptitle("MF-DRO Stage 2 v3: real-query distance to true optimum vs. cost, per seed\n"
             "(dashed = frozen run, solid = incumbent improved at least once)")
fig.tight_layout()
out_path = os.path.join(OUT_DIR, "xstar_dist_vs_cost.png")
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Saved {out_path}")
