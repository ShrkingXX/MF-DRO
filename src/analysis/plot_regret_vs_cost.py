"""Relative regret vs cost spent, all methods, all three benchmarks.

Multi-fidelity arms (h57) carry an explicit cost_curve. Single-fidelity arms
(h59) do not: every query costs c_H, so cost at iteration i is (i+1)*c_H.
Verified against the caps -- Borehole 100x2, Hartmann 25x8, Currin 66x3, all
landing on the 200 budget.
"""
import os, sys, json, glob
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benchmarks import get_benchmark

H57 = os.path.join(REPO, "experiments", "h57-baseline-comparison", "results")
H59 = os.path.join(REPO, "experiments", "h59-sfdro-baseline", "results")
BENCH = ["Currin_2D", "Hartmann_6D", "Borehole_8D"]
SEEDS = [44, 46, 48]
STYLE = [("SF-DRO",       "#d62728", "-",  2.4),
         ("MF-DRO",       "#d62728", "--", 1.8),
         ("SF-MES",       "#1f77b4", "-",  1.8),
         ("MF-MES",       "#1f77b4", "--", 1.8),
         ("MF-MI-Greedy", "#2ca02c", "--", 1.8),
         ("MF-GP-UCB",    "#999999", ":",  1.5)]

def _curves(bench, method):
    out = []
    for s in SEEDS:
        if method.startswith("SF-"):
            f = os.path.join(H59, f"{bench}__{method}__seed{s}.json")
            if not os.path.exists(f): continue
            d = json.load(open(f)); r = np.array(d["regret_curve"], float)
            c = (np.arange(len(r)) + 1) * float(d["c_H"])      # SF: every query costs c_H
        else:
            f = os.path.join(H57, f"{bench}__{method}__seed{s}.json")
            if not os.path.exists(f): continue
            d = json.load(open(f))
            r = np.array(d["hf_regret_curve"], float)
            c = np.array(d.get("cost_curve") or [], float)
            if c.size != r.size: continue
        if r.size >= 2: out.append((c, r))
    return out

def plot(out=os.path.join(REPO, "to_human", "regret_vs_cost_all.png")):
    grid = np.linspace(0, 200, 160)
    fig, ax = plt.subplots(1, 3, figsize=(15.4, 4.6))
    for j, b in enumerate(BENCH):
        a = ax[j]; fstar = -float(get_benchmark(f"{b}_HF")["known_optimal_value"])
        for me, col, ls, lw in STYLE:
            cs = _curves(b, me)
            if not cs: continue
            M = np.array([np.interp(grid, c, r, left=r[0], right=r[-1]) for c, r in cs]) / fstar
            a.plot(grid, M.mean(0), color=col, ls=ls, lw=lw, label=me,
                   zorder=4 if me.endswith("DRO") else 3)
        a.set_yscale("log"); a.set_xlim(0, 200); a.grid(alpha=.25, which="both")
        a.set_xlabel("cost spent (post-init)")
        a.set_title(f"{b}   f(x*)={fstar:.4g}", fontsize=10)
        if j == 0: a.set_ylabel("relative simple regret   regret / f(x*)")
    ax[0].legend(fontsize=8, loc="lower left", ncol=2)
    fig.suptitle("Regret vs cost — SF-DRO (solid red) vs MF-DRO (dashed red) vs baselines.  "
                 "Mean over seeds 44/46/48, budget 200 post-init.", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=145, bbox_inches="tight")
    print(f"  wrote {out}")

if __name__ == "__main__":
    plot()
