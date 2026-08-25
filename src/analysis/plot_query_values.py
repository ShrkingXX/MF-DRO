"""Scatter of the REAL queried value against iteration number, per seed.

x = iteration index (post-initial-design), y = the observed y at that query.
HF and LF are drawn separately because they are DIFFERENT FUNCTIONS -- plotting
them on one axis without that distinction invites reading a high LF value as
progress, when only HF observations can move the incumbent.
"""
import os, sys, json, glob
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benchmarks import get_benchmark
RES = os.path.join(REPO, "experiments", "h57-baseline-comparison", "results")

def plot(benches=("Hartmann_6D", "Borehole_8D"), method="MF-DRO", seeds=(44, 46, 48),
         out=os.path.join(REPO, "to_human", "mfdro_query_values.png")):
    fig, axes = plt.subplots(len(benches), len(seeds),
                             figsize=(5.0 * len(seeds), 3.7 * len(benches)), squeeze=False)
    for r, b in enumerate(benches):
        hf = get_benchmark(f"{b}_HF"); lf = get_benchmark(f"{b}_LF")
        fstar = -float(hf["known_optimal_value"]); lstar = float(lf["known_optimal_value"])
        for c, s in enumerate(seeds):
            ax = axes[r][c]
            f = os.path.join(RES, f"{b}__{method}__seed{s}.json")
            if not os.path.exists(f):
                ax.set_title(f"{b} s{s} — missing"); continue
            d = json.load(open(f))
            q = [e for e in d.get("queries", []) if not e.get("is_init")]
            if not q:
                ax.set_title(f"{b} s{s} — no trace"); continue
            Y = np.array([e["y"] for e in q]); F = np.array([e["fid"] for e in q])
            it = np.arange(len(q))
            # incumbent = running max over HF observations only
            inc = np.full(len(q), np.nan); best = -np.inf
            for i in range(len(q)):
                if F[i] == 1 and Y[i] > best: best = Y[i]
                inc[i] = best if np.isfinite(best) else np.nan
            ax.axhline(fstar, color="k", ls="--", lw=1.1, zorder=1)
            ax.scatter(it[F == 0], Y[F == 0], s=13, c="#9ecae1", marker="v",
                       linewidths=0, label="LF query", zorder=2)
            ax.scatter(it[F == 1], Y[F == 1], s=26, c="#d62728", marker="o",
                       edgecolors="k", linewidths=.35, label="HF query", zorder=4)
            ax.step(it, inc, where="post", color="#111", lw=1.6,
                    label="incumbent (best HF)", zorder=5)
            nh = int((F == 1).sum())
            ax.set_title(f"{b}  seed {s}   {nh}HF/{len(q)-nh}LF   "
                         f"regret {d['final_regret']:.3g}", fontsize=9)
            ax.grid(alpha=.22)
            if c == 0: ax.set_ylabel("observed value  y")
            if r == len(benches) - 1: ax.set_xlabel("iteration (post-init)")
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="lower right", framealpha=.9)
                ax.text(.02, .97, f"dashed = f(x*)={fstar:.3g}", transform=ax.transAxes,
                        va="top", fontsize=7, color="k")
            elif c == 0:
                ax.text(.02, .97, f"dashed = f(x*)={fstar:.3g}", transform=ax.transAxes,
                        va="top", fontsize=7, color="k")
    fig.suptitle(f"{method} — real queried value per iteration (h57, budget 200)", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=145, bbox_inches="tight")
    print(f"  wrote {out}")

if __name__ == "__main__":
    plot()
