"""The core figure: every conditioning intervention, and the one measurement
that explains all of them."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "to_human", "figs"); os.makedirs(OUT, exist_ok=True)

# (label, what was changed, % of probes where the argmax MOVED, status)
ROWS = [
    ("H4  AdaLN-Zero conditioning",      0.0,  "refuted"),
    ("H5  deny score head GP features",  0.0,  "refuted"),
    ("H8  RTG swept in realised band",   0.0,  "null"),
    ("H9  alpha_rtg floor",              None, "void"),
    ("H10 un-normalised RTG",            None, "void"),
    ("H11 real history, T=1 -> T=8",     0.0,  "null"),
    ("H11 + DT-style RTG decrement",     None, "void"),
    ("H16 joint-MES reward (alive sig.)",0.0,  "null"),
    ("H19 deterministic dynamics",       0.0,  "null"),
    ("H19 + stochastic behaviour policy",0.0,  "null"),
]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6),
                              gridspec_kw={"width_ratios": [2.05, 1]})
y = np.arange(len(ROWS))[::-1]
for yi, (lab, val, status) in zip(y, ROWS):
    if val is None:
        ax.text(0.5, yi, "VOID — manipulation check failed", va="center",
                fontsize=8.5, color="#9b9791", style="italic")
    else:
        ax.barh(yi, max(val, 0.6), color="#b23223", height=.55)
        ax.text(1.2, yi, f"{val:.0f}%", va="center", fontsize=8.5, color="#b23223")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in ROWS], fontsize=8.5)
ax.set_xlim(0, 35); ax.set_xlabel("% of probes where the proposed argmax MOVED")
ax.axvline(30, ls="--", c="k", lw=1)
ax.text(30.4, len(ROWS) - 1.2, "pre-registered\nsuccess bar (30%)", fontsize=7.5)
ax.set_title("Ten pre-registered interventions on the conditioning pathway",
             fontsize=11, loc="left")
ax.spines[["top", "right"]].set_visible(False)

# The explanation
labels = ["swap h for another\nstate's hidden vector",
          "coords-only features\n(ranking MUST use h)",
          "perturb state at\n1x batch sd"]
ax2.barh([2, 1, 0], [0.6, 0.6, 0.6], color="#b23223", height=.5)
for i, l in enumerate(labels):
    ax2.text(1.0, 2 - i, "argmax unchanged 12/12", va="center", fontsize=8.5,
             color="#b23223")
ax2.set_yticks([2, 1, 0]); ax2.set_yticklabels(labels, fontsize=8.5)
ax2.set_xlim(0, 14); ax2.set_xticks([])
ax2.set_title("H5: why — the score head barely reads h", fontsize=11, loc="left")
ax2.spines[["top", "right", "bottom"]].set_visible(False)

fig.suptitle("Nothing that reaches $h$ can move a decision that is not a function of $h$",
             fontsize=12.5, y=1.02)
fig.tight_layout()
p = os.path.join(OUT, "intervention_ladder.png")
fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
print(p)
