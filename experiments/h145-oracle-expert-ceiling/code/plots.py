"""h145 regret-vs-cost plot. Reuses h83's frozen sr_curve/grid.

h145 (ORACLE-EXPERT) never sets use_roi or inference_context_k, so it runs at
use_roi=False, K=1 -- its correct control is DEFAULT MF-DRO (h83), not CTRL-K1
(which is a ROI-Q10 arm). h201A (oracle teacher + K=8 window, current code, ROI
on) is included for context only -- it differs from h145 in BOTH use_roi and K,
so it is not a clean comparison, just a marker of where the K=8 result landed.

Quality comparison is FINAL simple regret only (standing instruction); this plot
is a diagnostic trajectory view, not a substitute for that comparison.
"""
import os, sys, json
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
BENCH_LIST = ("Borehole_8D", "Hartmann_6D")
G = np.linspace(0, 200, 201)
SEEDS = (42, 43, 44, 45, 46)

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

def runs_for(bench):
    d = {
        "h145 oracle teacher, K=1, no ROI (its own setting)":
            f"{REPO}/experiments/h145-oracle-expert-ceiling/results/{bench}__ORACLE-EXPERT__seed{{s}}.json",
        "default MF-DRO (h145's correct control, no ROI)":
            f"{REPO}/experiments/h83-main-comparison/results/{bench}__MF-DRO__seed{{s}}.json",
    }
    if bench == "Borehole_8D":
        # Context only: DIFFERENT setting (ROI on, K=8) -- not a clean pairing
        # with h145, included as a marker of where the K=8 result landed.
        d["[context only] oracle teacher, K=8, ROI-on (h201A)"] = \
            f"{REPO}/experiments/h201-oracle-teacher-window/results/{bench}__H201A-ORACLE-K8__seed{{s}}.json"
    return d

C = {"h145 oracle teacher, K=1, no ROI (its own setting)": "#7D3C98",
    "default MF-DRO (h145's correct control, no ROI)": "#7F8C8D",
    "[context only] oracle teacher, K=8, ROI-on (h201A)": "#C0392B"}

fig, axes = plt.subplots(1, len(BENCH_LIST), figsize=(11.5, 4.6))
for ax, bench in zip(axes, BENCH_LIST):
    OPT = float(get_benchmark(f"{bench}_HF")["known_optimal_value"])
    RUNS = runs_for(bench)
    for label, pat in RUNS.items():
        rows = []
        for s in SEEDS:
            p = pat.format(s=s)
            if not os.path.exists(p): continue
            run = json.load(open(p))
            c, sr = sr_curve(run, OPT)
            rows.append(grid(c, sr, G))
        if not rows:
            print(f"  {bench} / {label}: NO DATA"); continue
        A = np.vstack(rows)
        print(f"  {bench} / {label}: {len(rows)}/5 seeds")
        mu = np.nanmean(A, axis=0); se = np.nanstd(A, axis=0, ddof=1) / np.sqrt(A.shape[0])
        rel, rel_se = 100.0 * mu / abs(OPT), 100.0 * se / abs(OPT)
        ax.plot(G, rel, color=C[label], lw=2.0, label=label)
        ax.fill_between(G, rel - rel_se, rel + rel_se, color=C[label], alpha=0.15, lw=0)
    ax.set_yscale("log"); ax.set_xlabel("cost (post-init)")
    ax.set_ylabel("simple regret, % of |optimum|")
    ax.set_title(bench)
handles = [Line2D([], [], color=C[l], lw=2, label=l) for l in
          ["h145 oracle teacher, K=1, no ROI (its own setting)",
           "default MF-DRO (h145's correct control, no ROI)",
           "[context only] oracle teacher, K=8, ROI-on (h201A)"]]
fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False, bbox_to_anchor=(0.5, -0.22))
fig.suptitle("h145: oracle (interpolating) teacher, K=1 -- vs default MF-DRO", y=1.03, fontsize=11)
fig.subplots_adjust(bottom=0.32)
fig.savefig(os.path.join(OUT, "regret_vs_cost.png"))
plt.close(fig)
print(f"\n  wrote {os.path.join(OUT, 'regret_vs_cost.png')}")
