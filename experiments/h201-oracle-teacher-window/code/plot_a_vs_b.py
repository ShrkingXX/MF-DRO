"""h201: focused A (K=8 window) vs B (K=1, matched control) comparison.

Same oracle teacher in both arms -- the ONLY difference is inference_context_k.
Reuses h83's frozen sr_curve/grid (imported, never re-derived).
"""
import os, sys, json, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

PATTERNS = {
    "h201A: oracle teacher + K=8 window": f"{REPO}/experiments/h201-oracle-teacher-window/results/Borehole_8D__H201A-ORACLE-K8__seed{{s}}.json",
    "h201B: SAME oracle teacher, K=1 (matched control)": f"{REPO}/experiments/h201-oracle-teacher-window/results/Borehole_8D__H201B-ORACLE-K1__seed{{s}}.json",
}
COLORS = {"h201A: oracle teacher + K=8 window": "#C0392B",
         "h201B: SAME oracle teacher, K=1 (matched control)": "#229954"}

curves, finals = {}, {}
for label, pat in PATTERNS.items():
    rows, fin = [], []
    for s in SEEDS:
        p = pat.format(s=s)
        run = json.load(open(p))
        c, sr = sr_curve(run, OPT)
        g = grid(c, sr, G)
        rows.append(g)
        fin.append(100.0 * g[-1] / abs(OPT))
    curves[label] = np.vstack(rows)
    finals[label] = np.array(fin)
    print(f"  {label}: {len(rows)}/5 seeds, final mean {np.mean(fin):.2f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})

for label, A in curves.items():
    mu = np.nanmean(A, axis=0); se = np.nanstd(A, axis=0, ddof=1) / np.sqrt(A.shape[0])
    rel, rel_se = 100.0 * mu / abs(OPT), 100.0 * se / abs(OPT)
    ax1.plot(G, rel, color=COLORS[label], lw=2.2, label=label)
    ax1.fill_between(G, np.clip(rel - rel_se, 1e-6, None), rel + rel_se, color=COLORS[label], alpha=0.15, lw=0)
ax1.set_yscale("log")
ax1.set_xlabel("cost (post-init)")
ax1.set_ylabel("simple regret, % of |optimum|")
ax1.set_title("Same oracle teacher -- only the window differs")
ax1.legend(loc="upper right", frameon=False, fontsize=8.5)

labels_short = ["h201A\n(K=8 window)", "h201B\n(K=1, matched ctrl)"]
means = [np.mean(finals[l]) for l in PATTERNS]
bars = ax2.bar(labels_short, means, color=[COLORS[l] for l in PATTERNS], width=0.55, zorder=2)
for i, l in enumerate(PATTERNS):
    xs = np.full(5, i) + np.linspace(-0.12, 0.12, 5)
    ax2.scatter(xs, finals[l], color="black", s=18, zorder=3, alpha=0.7)
for bar, m in zip(bars, means):
    ax2.text(bar.get_x() + bar.get_width()/2, m + 1.0, f"{m:.2f}", ha="center", fontsize=10, weight="bold")
ax2.set_ylabel("final simple regret, % of |optimum|")
ax2.set_title("Final regret @ cost 200\n(the only quality comparison)")
d = finals[list(PATTERNS)[0]] - finals[list(PATTERNS)[1]]
ax2.text(0.5, -0.22, f"paired A-B: {np.mean(d):+.2f}  (se {np.std(d, ddof=1)/np.sqrt(5):.2f}, 5/5 seeds)",
         transform=ax2.transAxes, ha="center", fontsize=8.5, color="#555")

fig.suptitle("h201 ablation: the WINDOW alone does nothing without a teacher worth reading at the readout position",
            y=1.03, fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "h201_a_vs_b.png"))
plt.close(fig)
print(f"\n  wrote {os.path.join(OUT, 'h201_a_vs_b.png')}")
