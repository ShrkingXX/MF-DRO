"""Regret-vs-cost curves, current arms vs baselines. Metric IMPORTED from h83's
sr_curve/grid (never re-derived). Curves are DIAGNOSTIC; quality is compared only
at the endpoint (cost 200), which is direct-labelled. h210's registered leans are
drawn as hollow markers at cost 200 so expectation and measurement are visibly
different objects; measured h210 arms replace them as they land."""
import os, sys, json, glob, importlib.util
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
_s = importlib.util.spec_from_file_location("a83", os.path.join(REPO, "experiments/h83-main-comparison/code/analyse.py"))
a83 = importlib.util.module_from_spec(_s); sys.modules["a83"] = a83; _s.loader.exec_module(a83)
from benchmarks import get_benchmark
E = os.path.join(REPO, "experiments"); G = np.linspace(0, 200, 201)
C = {"MF-MES": "#3a3a38", "MF-GP-UCB": "#8a8a86", "SF-DRO": "#b0b0ab", "MF-DRO no-ROI": "#c9c9c4",
     "ROI-Q10 K=1": "#2a78d6", "NIR": "#eb6834", "MIXR": "#1baf7a", "MIXO": "#4a3aa7", "CTRL-K1-H": "#2a78d6"}
def mean_curve(pat, b):
    opt = float(get_benchmark(f"{b}_HF")["known_optimal_value"]); k = 100.0 / abs(opt)
    rows = []
    for f in sorted(glob.glob(pat)):
        c, s = a83.sr_curve(json.load(open(f)), opt); rows.append(a83.grid(np.asarray(c), np.asarray(s), G) * k)
    if not rows: return None, 0
    A = np.vstack(rows); return np.nanmean(A, axis=0), len(rows)
SERIES = {
 "Borehole_8D": [
  ("MF-DRO no-ROI", f"{E}/h83-main-comparison/results/Borehole_8D__MF-DRO__seed4[2-6].json", 1.0, "-"),
  ("SF-DRO",        f"{E}/h83-main-comparison/results/Borehole_8D__SF-DRO__seed4[2-6].json", 1.0, "-"),
  ("MF-GP-UCB",     f"{E}/h83-main-comparison/results/Borehole_8D__MF-GP-UCB__seed4[2-6].json", 1.0, "-"),
  ("MF-MES",        f"{E}/h83-main-comparison/results/Borehole_8D__MF-MES__seed4[2-6].json", 1.8, "-"),
  ("ROI-Q10 K=1",   f"{E}/h84-roi-strategy/results/Borehole_8D__ROI-Q10__seed4[2-6].json", 1.6, "-"),
  ("NIR",           f"{E}/h207-random-rollout-dataset/results/Borehole_8D__H207NIR-MES-IR__seed4[2-6].json", 1.6, "-"),
  ("MIXR",          f"{E}/h207-random-rollout-dataset/results/Borehole_8D__H207MIXR__seed4[2-6].json", 2.2, "-"),
  ],
 "Hartmann_6D": [
  ("MF-DRO no-ROI", f"{E}/h83-main-comparison/results/Hartmann_6D__MF-DRO__seed4[2-6].json", 1.0, "-"),
  ("SF-DRO",        f"{E}/h83-main-comparison/results/Hartmann_6D__SF-DRO__seed4[2-6].json", 1.0, "-"),
  ("MF-GP-UCB",     f"{E}/h83-main-comparison/results/Hartmann_6D__MF-GP-UCB__seed4[2-6].json", 1.0, "-"),
  ("MF-MES",        f"{E}/h83-main-comparison/results/Hartmann_6D__MF-MES__seed4[2-6].json", 1.8, "-"),
  ("ROI-Q10 K=1",   f"{E}/h84-roi-strategy/results/Hartmann_6D__ROI-Q10__seed4[2-6].json", 1.6, "-"),
  ("NIR",           f"{E}/h209-mixr-replication/results/Hartmann_6D__H209NIR__seed4[2-6].json", 1.6, "-"),
  ("MIXR",          f"{E}/h209-mixr-replication/results/Hartmann_6D__H209MIXR__seed4[2-6].json", 2.2, "-"),
  ("CTRL-K1-H",     f"{E}/h210-mean-vs-median/results/Hartmann_6D__H210CTRLK1__seed4[2-6].json", 1.6, ":")]}
# h210 registered leans (protocol.md), drawn HOLLOW at cost 200; replaced by measured curves as they land
EXPECT = {"Borehole_8D": [("L1-NIR-B", 8.0, "#eb6834"), ("MIXR-P72-B", 9.0, "#1baf7a"), ("L1-MIXR-B", 7.0, "#1baf7a")],
          "Hartmann_6D": [("CTRL-K1-H", 5.93, "#2a78d6"), ("L1-NIR-H", 9.75, "#eb6834")]}
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.22, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 160, "savefig.bbox": "tight", "font.family": "sans-serif"})
fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2)); fig.subplots_adjust(wspace=0.62)
table = {}
for ax, (b, series) in zip(axes, SERIES.items()):
    ends = []
    for nm, pat, lw, ls in series:
        m, n = mean_curve(pat, b)
        if m is None: continue
        ax.plot(G, m, color=C[nm], lw=lw, ls=ls, zorder=3 if lw > 1.2 else 2)
        ends.append((float(m[-1]), nm, C[nm], n)); table[(b, nm)] = (float(m[-1]), n)
    # one right-hand label stack: measured endpoints (solid) + h210 leans (hollow), de-collided together
    items = [(v, f"{nm}  {v:.2f}", col, "measured") for v, nm, col, n in ends]
    items += [(v, f"expected {nm}  ~{v:.1f}", col, "expected") for nm, v, col in EXPECT[b] if (b, nm) not in table]
    items.sort(key=lambda t: t[0]); ys = []
    for v, txt, col, kind in items:
        y = v
        for prev in ys:
            if abs(np.log10(y) - np.log10(prev)) < 0.055: y = prev * 10 ** 0.055
        ys.append(y)
        if kind == "expected":
            ax.plot([200], [v], marker="o", ms=6.5, mfc="white", mec=col, mew=1.3, ls="", zorder=4, clip_on=False)
        ax.annotate(txt, (200, v), xytext=(207, y), fontsize=7.6, va="center",
                    color=("#52514e" if kind == "expected" else "#0b0b0b"),
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.6, ls=(":" if kind == "expected" else "-"), shrinkA=0, shrinkB=2))
    if b == "Borehole_8D":
        ax.text(0.02, 0.96, "MIXO (MES + oracle half; ceiling, not deployable) reaches 0.00 by cost ~45 on 5/5 -- not drawn",
                transform=ax.transAxes, fontsize=7.2, color="#52514e", va="top")
    ax.set_yscale("log"); ax.set_xlim(0, 200); ax.set_xlabel("cost after initial design")
    ax.set_title(f"{b.replace('_', ' ')} — mean over seeds 42–46")
    if b == "Borehole_8D": ax.set_ylabel("simple regret, % of |optimum|  (log)")
    ax.axvline(200, color="#c3c2b7", lw=0.8, ls=":")
leg = [Line2D([], [], color="#3a3a38", lw=1.8, label="MF-MES (strongest baseline)"),
       Line2D([], [], color="#8a8a86", lw=1.0, label="MF-GP-UCB / SF-DRO / MF-DRO no-ROI (baselines, greys)"),
       Line2D([], [], color="#2a78d6", lw=1.6, label="ROI-Q10 K=1 — MF-DRO in the last full run"),
       Line2D([], [], color="#eb6834", lw=1.6, label="NIR — K=8, no positional emb., terminal_improvement (control)"),
       Line2D([], [], color="#1baf7a", lw=2.2, label="MIXR — NIR + 20 unselected random rollouts/member"),
       Line2D([], [], marker="o", mfc="white", mec="#52514e", ls="", label="h210 registered lean (not yet measured)")]
fig.legend(handles=leg, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.06))
fig.suptitle("Regret vs cost — current arms against the baselines (endpoint is the only verdict; curves are diagnostic)", y=1.01, fontsize=11)
out = os.path.join(REPO, "to_human", "regret_cost_current_vs_baselines.png"); fig.savefig(out); print("wrote", out)
print("\nendpoint (cost 200), mean over seeds:")
for (b, nm), (v, n) in sorted(table.items()): print(f"  {b:12s} {nm:14s} {v:7.2f}  n={n}")
