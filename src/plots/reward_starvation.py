"""Figures for the reward-starvation finding. Reusable: reads gate13.json."""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "to_human", "figs")
os.makedirs(OUT, exist_ok=True)


def fig_starvation(gate_path):
    g = json.load(open(gate_path))
    order = [k for k in g if k != "gate"]
    labels = [k.replace("baseline ", "").replace("  ", "\n") for k in order]
    dead = [g[k]["dead_frac"] * 100 for k in order]
    lfnz = [g[k]["lf_nonzero_frac"] * 100 for k in order]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(order))

    ax[0].bar(x, dead, color=["#b23", "#e8a33d", "#4a90d9", "#3a9"][:len(x)])
    ax[0].axhline(20, ls="--", c="k", lw=1)
    ax[0].text(len(x) - .45, 21.5, "gate G1 (<20%)", ha="right", fontsize=8)
    ax[0].set_ylabel("trajectories with rtg[0] == 0  (%)")
    ax[0].set_title("Dead conditioning signal", fontsize=11)

    ax[1].bar(x, lfnz, color=["#b23", "#e8a33d", "#4a90d9", "#3a9"][:len(x)])
    ax[1].axhline(50, ls="--", c="k", lw=1)
    ax[1].text(len(x) - .45, 52, "gate G2 (>50%)", ha="right", fontsize=8)
    ax[1].set_ylabel("LF steps earning nonzero reward  (%)")
    ax[1].set_title("Does an LF query earn any credit?", fontsize=11)

    for a in ax:
        a.set_xticks(x); a.set_xticklabels(labels, fontsize=8)
        a.spines[["top", "right"]].set_visible(False)
        a.set_ylim(0, 105)
    fig.suptitle("Why three RTG manipulations voided: the reward is zero almost everywhere",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    p = os.path.join(OUT, "reward_starvation.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p


def fig_band_cap():
    """The provable cap that killed H9 and H10, with the measured points."""
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    a = np.linspace(0.05, 1.0, 200)
    ax.plot(a, 1 / a, "k-", lw=1.8, label=r"provable ceiling $1/\alpha_{rtg}$")
    ax.fill_between(a, 1, 1 / a, color="#4a90d9", alpha=.12,
                    label="reachable band")
    ax.scatter([0.5, 0.5], [1.76, 2.59], zorder=5, s=55,
               c=["#b23", "#e8a33d"], edgecolor="k", lw=.6)
    ax.annotate("H10 control (normalised) 1.76x", (0.5, 1.76),
                textcoords="offset points", xytext=(12, -14), fontsize=8)
    ax.annotate("H10 raw 2.59x", (0.5, 2.59),
                textcoords="offset points", xytext=(12, 4), fontsize=8)
    ax.axhline(5, ls="--", c="r", lw=1)
    ax.text(0.97, 5.3, "H10 required 5x", ha="right", fontsize=8, color="r")
    ax.set_xlabel(r"$\alpha_{rtg}$"); ax.set_ylabel("RTG band width (max/min)")
    ax.set_ylim(0, 12); ax.set_xlim(0.05, 1.0)
    ax.set_title("target = max(batch_max, " r"$\alpha$" "·running_max)\n"
                 "caps the band at 1/" r"$\alpha$" " — two lines of algebra",
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = os.path.join(OUT, "rtg_band_cap.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p


if __name__ == "__main__":
    gp = os.path.join(REPO, "experiments", "h13-kg-dense-reward",
                      "results", "gate13.json")
    print(fig_starvation(gp))
    print(fig_band_cap())
