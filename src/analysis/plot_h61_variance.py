"""h61: the teacher's local refinement collapses MF-DRO's seed-to-seed spread.

Per-seed values, not just means -- the means of POOL600 and REFINE are nearly
identical (60.23 vs 59.75) and the whole finding is in the spread.
"""
import os, sys, json
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benchmarks import get_benchmark

H57 = os.path.join(REPO, "experiments", "h57-baseline-comparison", "results")
H61 = os.path.join(REPO, "experiments", "h61-teacher-optimizer", "results")
S = (44, 46, 48)

def main(out=os.path.join(REPO, "to_human", "h61_variance.png")):
    fs = -float(get_benchmark("Borehole_8D_HF")["known_optimal_value"])
    arms = [("BASE\n200 pts, argmax",
             [json.load(open(os.path.join(H57, f"Borehole_8D__MF-DRO__seed{s}.json")))["final_regret"] for s in S],
             "#777777"),
            ("POOL600\n600 pts, argmax",
             [json.load(open(os.path.join(H61, f"Borehole_8D__POOL600__seed{s}.json")))["final_regret"] for s in S],
             "#1f77b4"),
            ("REFINE\n200 + 100 local",
             [json.load(open(os.path.join(H61, f"Borehole_8D__REFINE__seed{s}.json")))["final_regret"] for s in S],
             "#d62728")]
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for i, (name, v, c) in enumerate(arms):
        v = np.array(v) / fs * 100
        ax.plot([i] * 3, v, "o", ms=11, mfc="none", mec=c, mew=2.2, zorder=3)
        ax.plot([i - .17, i + .17], [v.mean()] * 2, "-", color=c, lw=3, zorder=4)
        ax.vlines(i, v.min(), v.max(), color=c, lw=1.2, alpha=.5, zorder=2)
        for y, s in zip(v, S):
            ax.annotate(f"s{s}", (i + .055, y), fontsize=7.5, color=c, va="center")
        ax.annotate(f"mean {v.mean():.1f}%\nspread {v.max()-v.min():.1f} pp",
                    (i, v.max() + 0.9), ha="center", fontsize=8.5, color=c)
    for y, lbl, c in ((8.3, "MI-Greedy 8.3%", "#2ca02c"), (11.3, "MF-MES 11.3%", "#1f77b4")):
        ax.axhline(y, color=c, ls=":", lw=1.3, alpha=.75)
        ax.annotate(lbl, (2.42, y), fontsize=8, color=c, va="center")
    ax.set_xticks(range(3)); ax.set_xticklabels([a[0] for a in arms], fontsize=9)
    ax.set_xlim(-.45, 2.75); ax.set_ylabel("relative simple regret  (regret / f(x*))  %")
    ax.set_title("H61 — teacher refinement collapses the seed spread (Borehole 8D, budget 200)",
                 fontsize=10.5)
    ax.yaxis.set_major_formatter(lambda x, p: f"{x:.0f}%")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  wrote {out}")

if __name__ == "__main__":
    main()
