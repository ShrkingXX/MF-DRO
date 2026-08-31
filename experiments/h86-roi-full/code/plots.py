"""H86 figures, matching H83's fig2/fig3 conventions exactly.

Reuses h83's frozen `sr_curve`/`grid` rather than reimplementing simple regret --
a second implementation of the metric is how a read-point mismatch gets in.

H86 is the ROI arm on the two benchmarks the ROI had never seen (Currin, Ackley).
The seed-by-seed panel therefore compares ROI-Q10 against its no-ROI control
(h83 MF-DRO, same seeds), which is the contrast h86 exists to make.
"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import importlib.util

H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
_s = importlib.util.spec_from_file_location(
    "a83", os.path.join(REPO, "experiments/h83-main-comparison/code/analyse.py"))
a83 = importlib.util.module_from_spec(_s); sys.modules["a83"] = a83; _s.loader.exec_module(a83)
from benchmarks import get_benchmark

OUT   = os.path.join(H, "..", "results")
H83   = os.path.join(REPO, "experiments/h83-main-comparison/results")
BENCH = ("Currin_2D", "Ackley_10D")
SEEDS = (42, 43, 44, 45, 46)
G     = np.linspace(0, 200, 201)
C     = {"ROI-Q10": "#C0392B", "MF-DRO (no ROI)": "#2471A3"}

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

def units(b):
    o = float(get_benchmark(f"{b}_HF")["known_optimal_value"])
    return (1.0, "absolute simple regret") if abs(o) < 1e-9 else (100.0/abs(o), "relative regret (%)")

def curve(path, b):
    opt = float(get_benchmark(f"{b}_HF")["known_optimal_value"])
    r = json.load(open(path))
    c, s = a83.sr_curve(r, opt)
    return c, s, r

PATHS = {}
for b in BENCH:
    for s in SEEDS:
        p_roi = os.path.join(OUT,  f"{b}__ROI-Q10__seed{s}.json")
        p_ctl = os.path.join(H83,  f"{b}__MF-DRO__seed{s}.json")
        if os.path.exists(p_roi): PATHS[(b, "ROI-Q10", s)] = p_roi
        if os.path.exists(p_ctl): PATHS[(b, "MF-DRO (no ROI)", s)] = p_ctl
print(f"  located {len(PATHS)} runs "
      f"(ROI {sum(1 for k in PATHS if k[1]=='ROI-Q10')}, control {sum(1 for k in PATHS if k[1]!='ROI-Q10')})")

# ---------- Figure 2 analogue: seed-by-seed ----------
fig, axes = plt.subplots(len(BENCH), len(SEEDS), figsize=(16, 5.6), sharex=True)
for i, b in enumerate(BENCH):
    k, lab = units(b)
    for j, s in enumerate(SEEDS):
        ax = axes[i, j]
        for m in ("MF-DRO (no ROI)", "ROI-Q10"):
            if (b, m, s) not in PATHS: continue
            c, sr, _ = curve(PATHS[(b, m, s)], b)
            ax.plot(c, sr*k, color=C[m], lw=1.4)
        ax.set_yscale("log")
        if i == 0: ax.set_title(f"seed {s}")
        if j == 0: ax.set_ylabel(f"{b}\n{lab}", fontsize=8)
        if i == len(BENCH)-1: ax.set_xlabel("cost (post-init)")
fig.legend(handles=[Line2D([], [], color=C[m], lw=2, label=m) for m in C],
           loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.suptitle("H86 simple regret vs cost, per seed — ROI-Q10 vs no-ROI control", y=1.0, fontsize=12)
fig.savefig(os.path.join(OUT, "fig2_seed_by_seed_sr_vs_cost.png")); plt.close(fig)

# ---------- Figure 3 analogue: queried value + fidelity per iteration ----------
fig, axes = plt.subplots(len(BENCH), len(SEEDS), figsize=(16, 5.4))
for i, b in enumerate(BENCH):
    opt = float(get_benchmark(f"{b}_HF")["known_optimal_value"])
    for j, s in enumerate(SEEDS):
        ax = axes[i, j]
        key = (b, "ROI-Q10", s)
        if key not in PATHS: ax.axis("off"); continue
        q = [e for e in json.load(open(PATHS[key]))["queries"] if not e.get("is_init")]
        it = np.arange(len(q)); y = np.array([e["y"] for e in q]); fid = np.array([e["fid"] for e in q])
        # y is stored LARGER-IS-BETTER (regret = -max(y_HF) - known_optimal_value),
        # so the best attainable y is -opt. Plot y as-is; negating it would make
        # maximum.accumulate track the WORST query. Same note as H83's fig3.
        ax.scatter(it[fid == 0], y[fid == 0], s=9, c="#5DADE2", marker="x", lw=0.8)
        ax.scatter(it[fid == 1], y[fid == 1], s=11, c="#C0392B", marker="o", edgecolors="none")
        run = np.maximum.accumulate(np.where(fid == 1, y, -np.inf))
        ax.plot(it, np.where(np.isfinite(run), run, np.nan), color="k", lw=1.0, alpha=0.7)
        ax.axhline(-opt, color="k", ls="--", lw=0.8, alpha=0.6)
        ax.text(0.97, 0.05, f"LF {100.0*np.mean(fid==0):.0f}%", transform=ax.transAxes,
                ha="right", fontsize=7.5, bbox=dict(fc="w", ec="none", alpha=0.7, pad=1.5))
        if i == 0: ax.set_title(f"seed {s}")
        if j == 0: ax.set_ylabel(f"{b}\nqueried value", fontsize=8)
        if i == len(BENCH)-1: ax.set_xlabel("optimization iteration")
fig.legend(handles=[Line2D([], [], color="#C0392B", marker="o", ls="", label="HF query"),
                    Line2D([], [], color="#5DADE2", marker="x", ls="", label="LF query"),
                    Line2D([], [], color="k", lw=1, label="running best HF"),
                    Line2D([], [], color="k", ls="--", lw=1, label="true optimum")],
           loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.suptitle("H86 ROI-Q10: queried value and fidelity per iteration", y=1.0, fontsize=12)
fig.savefig(os.path.join(OUT, "fig3_roi_query_fidelity.png")); plt.close(fig)
print("  wrote fig2_seed_by_seed_sr_vs_cost.png, fig3_roi_query_fidelity.png")
