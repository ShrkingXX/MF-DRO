"""h201 regret-vs-cost plot. Reuses h83's frozen sr_curve/grid -- the same
trace-recomputed quantity the readout tables report, not a re-derivation.

Quality comparison is FINAL simple regret only (standing instruction); this plot
is a diagnostic trajectory view, not a substitute for that comparison.
"""
import os, sys, glob, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments/h83-main-comparison/code"))
from analyse import sr_curve, grid
from benchmarks import get_benchmark

OUT = os.path.join(H, "..", "results")
OPT = float(get_benchmark("Borehole_8D_HF")["known_optimal_value"])
G = np.linspace(0, 200, 201)
SEEDS = (42, 43, 44, 45, 46)

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

RUNS = {
    "oracle teacher + K=8 window (h201A)":
        f"{REPO}/experiments/h201-oracle-teacher-window/results/Borehole_8D__H201A-ORACLE-K8__seed{{s}}.json",
    "MES teacher, no window (CTRL-K1)":
        f"{REPO}/experiments/h194-expert-plan-window/results/Borehole_8D__CTRL-K1__seed{{s}}.json",
    "default MF-DRO (paper baseline)":
        f"{REPO}/experiments/h83-main-comparison/results/Borehole_8D__MF-DRO__seed{{s}}.json",
}
# Arm B (K=1, matched oracle-teacher control) is still running as of this plot --
# included only if/when its 5 finals exist, so the figure never silently omits it
# once it lands without a code change.
_armB_pat = f"{REPO}/experiments/h201-oracle-teacher-window/results/Borehole_8D__H201B-ORACLE-K1__seed{{s}}.json"
if all(os.path.exists(_armB_pat.format(s=s)) for s in SEEDS):
    RUNS["oracle teacher + K=1 (h201B, matched ctrl)"] = _armB_pat

C = {"oracle teacher + K=8 window (h201A)": "#C0392B",
    "MES teacher, no window (CTRL-K1)": "#2471A3",
    "default MF-DRO (paper baseline)": "#7F8C8D",
    "oracle teacher + K=1 (h201B, matched ctrl)": "#27AE60"}

curves = {}
for label, pat in RUNS.items():
    rows = []
    for s in SEEDS:
        p = pat.format(s=s)
        if not os.path.exists(p): continue
        run = json.load(open(p))
        c, sr = sr_curve(run, OPT)
        rows.append(grid(c, sr, G))
    if rows:
        curves[label] = np.vstack(rows)
        print(f"  {label}: {len(rows)}/5 seeds")
    else:
        print(f"  {label}: NO DATA")

fig, ax = plt.subplots(figsize=(7.5, 5.2))
for label, A in curves.items():
    mu = np.nanmean(A, axis=0)
    se = np.nanstd(A, axis=0, ddof=1) / np.sqrt(A.shape[0])
    rel = 100.0 * mu / abs(OPT)
    rel_se = 100.0 * se / abs(OPT)
    ax.plot(G, rel, color=C[label], lw=2.0, label=label)
    ax.fill_between(G, rel - rel_se, rel + rel_se, color=C[label], alpha=0.15, lw=0)
ax.set_yscale("log")
ax.set_xlabel("cost (post-init)")
ax.set_ylabel("simple regret, % of |optimum|")
ax.set_title("Borehole_8D: simple regret vs cost (5 seeds, shaded = ±1 s.e.)")
ax.legend(loc="upper right", frameon=False, fontsize=8)
fig.savefig(os.path.join(OUT, "regret_vs_cost.png"))
plt.close(fig)
print(f"\n  wrote {os.path.join(OUT, 'regret_vs_cost.png')}")

# ---------- seed-by-seed panel ----------
fig, axes = plt.subplots(1, 5, figsize=(17, 3.4), sharey=True)
for j, s in enumerate(SEEDS):
    ax = axes[j]
    for label, pat in RUNS.items():
        p = pat.format(s=s)
        if not os.path.exists(p): continue
        run = json.load(open(p))
        c, sr = sr_curve(run, OPT)
        gcurve = grid(c, sr, G)
        ax.plot(G, 100.0 * gcurve / abs(OPT), color=C[label], lw=1.4)
    ax.set_yscale("log"); ax.set_title(f"seed {s}")
    ax.set_xlabel("cost (post-init)")
    if j == 0: ax.set_ylabel("simple regret, % of |optimum|")
fig.legend(handles=[Line2D([], [], color=C[l], lw=2, label=l) for l in curves],
           loc="lower center", ncol=len(curves), frameon=False, bbox_to_anchor=(0.5, -0.12))
fig.suptitle("Borehole_8D: simple regret vs cost, per seed", y=1.05, fontsize=11)
fig.savefig(os.path.join(OUT, "regret_vs_cost_per_seed.png"))
plt.close(fig)
print(f"  wrote {os.path.join(OUT, 'regret_vs_cost_per_seed.png')}")
