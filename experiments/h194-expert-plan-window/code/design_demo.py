"""Demonstration of the sliding-window inference design vs the original DT setting.

Side by side:
  LEFT  -- K=1, the original single-token readout (Old_dro.py's
           _propose_next_candidate, and MF-DRO's default: hist=None branch,
           decisionTransformer.py ~line 720-722).
  RIGHT -- K=8, the sliding window (decisionTransformer.py propose_mf's
           `if hist:` branch, ~line 660-680): up to 7 past real queries +
           the current step, real actions fed into history slots (h196 fix),
           causal transformer, readout = LAST state token.

Boxes/labels are drawn from the actual code structure, not a generic
schematic -- token types, the causal mask, and the readout slice
(`h_full[0, ...][-1]`) all match what propose_mf actually does.
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

H = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(H, "..", "results")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "axes.spines.bottom": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

TOK_C = {"rtg": "#27AE60", "btg": "#E67E22", "state": "#2471A3", "action": "#7D3C98"}
TOK_LABEL = {"rtg": "RTG", "btg": "BTG", "state": "state", "action": "action"}

def token_column(ax, x, y0, h, w, kinds, faded=False, action_real=True):
    """Draw one timestep's stacked token boxes, bottom to top."""
    for i, k in enumerate(kinds):
        y = y0 + i * h
        fc = TOK_C[k]
        alpha = 0.35 if faded else (0.5 if (k == "action" and not action_real) else 0.9)
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h * 0.86, boxstyle="round,pad=0.01",
                     fc=fc, ec="none", alpha=alpha))
        if not faded:
            ax.text(x + w/2, y + h*0.43, TOK_LABEL[k], ha="center", va="center",
                    fontsize=6.3, color="white", weight="bold")

def draw_panel(ax, K, title, subtitle):
    ax.set_xlim(-0.6, 9.2); ax.set_ylim(-1.7, 8.2); ax.axis("off")
    w, h, gap = 0.78, 0.62, 0.18
    kinds = ["action", "state", "btg", "rtg"]  # bottom to top (matches seq stack order)
    n = K
    xs = np.arange(n) * (w + gap)
    y0 = 0.0
    for i, x in enumerate(xs):
        is_last = (i == n - 1)
        faded = False
        action_real = True if (i < n - 1) else False  # current step's action slot stays zero (h196 note)
        token_column(ax, x, y0, h, w, kinds, faded=faded, action_real=action_real)
        ax.text(x + w/2, y0 - 0.35, f"$\\tau$={i}" if n > 1 else "(only step)",
                fontsize=7, ha="center", color="#555")
        if is_last:
            # kinds = ["action","state","btg","rtg"] bottom-to-top -> "state" is index 1.
            # The readout is the STATE token (decisionTransformer.py's own comment:
            # "STATE token, matching forward_mf's h_act"), NOT the RTG token at the top --
            # caught by inspecting the first render, where the highlight sat on RTG.
            _state_i = kinds.index("state")
            ax.add_patch(mpatches.FancyBboxPatch((x - 0.06, y0 + _state_i*h - 0.06), w + 0.12, h*0.86 + 0.12,
                         boxstyle="round,pad=0.01", fc="none", ec="#C0392B", lw=2.2))

    # causal mask indicator (only meaningful for K>1)
    if n > 1:
        ax.text(xs[-1] + w + 0.5, 2.3, "causal\nmask", fontsize=7, color="#7F8C8D",
                ha="center", style="italic")
        ax.annotate("", xy=(xs[0] + w/2, -0.85), xytext=(xs[-1] + w/2, -0.85),
                    arrowprops=dict(arrowstyle="->", color="#7F8C8D", lw=1.2))
        ax.text((xs[0] + xs[-1])/2 + w/2, -1.15, "oldest → most recent (real queries)",
                fontsize=7, ha="center", color="#7F8C8D")

    # transformer block
    tx0, tx1 = -0.3, xs[-1] + w + 0.3
    ax.add_patch(mpatches.FancyBboxPatch((tx0, 4.55), tx1 - tx0, 0.9, boxstyle="round,pad=0.02",
                 fc="#EAECEE", ec="#566573", lw=1.3))
    ax.text((tx0+tx1)/2, 5.0, f"causal transformer ({4*n} tokens)" if n > 1 else "causal transformer (4 tokens)",
            ha="center", va="center", fontsize=8)
    for x in xs:
        ax.annotate("", xy=(x + w/2, 4.55), xytext=(x + w/2, 3*h + h*0.86 + 0.15),
                    arrowprops=dict(arrowstyle="->", color="#566573", lw=1.0))

    # readout arrow -- from the LAST column's STATE token specifically (index 1 of
    # ["action","state","btg","rtg"]), through the transformer block, to the readout label.
    rx = xs[-1] + w/2
    ax.annotate("", xy=(rx, 6.15), xytext=(rx, 5.45),
                arrowprops=dict(arrowstyle="->", color="#C0392B", lw=2.0,
                                connectionstyle="arc3,rad=0"))
    ax.annotate("", xy=(rx, 4.55), xytext=(rx, 1*h + h*0.86*0.5),
                arrowprops=dict(arrowstyle="-", color="#C0392B", lw=1.6, ls=(0,(3,2)), alpha=0.55))
    ax.add_patch(mpatches.FancyBboxPatch((rx - 1.05, 6.15), 2.1, 0.55, boxstyle="round,pad=0.02",
                 fc="#FADBD8", ec="#C0392B", lw=1.3))
    pos_txt = "position 0" if n == 1 else f"position {n-1}"
    ax.text(rx, 6.42, f"readout: LAST state token\n({pos_txt})", ha="center", va="center",
            fontsize=7.3, color="#7B241C")

    # heads
    for dx, lbl in ((-1.3, "location\nhead"), (1.3, "fidelity\nhead")):
        hx = rx + dx
        ax.annotate("", xy=(hx, 7.1), xytext=(rx, 6.72),
                    arrowprops=dict(arrowstyle="->", color="#566573", lw=1.1))
        ax.add_patch(mpatches.FancyBboxPatch((hx - 0.55, 7.1), 1.1, 0.5, boxstyle="round,pad=0.02",
                     fc="#D6EAF8", ec="#2471A3", lw=1.1))
        ax.text(hx, 7.35, lbl, ha="center", va="center", fontsize=7)

    ax.text((tx0+tx1)/2, 7.95, title, ha="center", fontsize=11.5, weight="bold")
    ax.text((tx0+tx1)/2, 7.62, subtitle, ha="center", fontsize=8, color="#555")

fig, axes = plt.subplots(1, 2, figsize=(13, 6.6))
draw_panel(axes[0], 1,
          "K=1 -- original DT setting",
          "Old_dro.py's single-token call; MF-DRO's default (hist=None)")
draw_panel(axes[1], 8,
          "K=8 -- sliding window",
          "propose_mf's `if hist:` branch -- 7 past real queries + current")

legend_handles = [mpatches.Patch(color=c, label=TOK_LABEL[k]) for k, c in TOK_C.items()]
fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False,
          bbox_to_anchor=(0.5, -0.02), fontsize=9)

fig.text(0.5, 1.0,
         "How the sliding window changes DT inference: SAME transformer, SAME two heads --\n"
         "only the SEQUENCE LENGTH and the READ POSITION change",
         ha="center", fontsize=12.5, va="top")
fig.text(0.5, -0.10,
         "position embeddings: K=1 always uses timestep=0 (hardcoded, matching Old_dro.py's dummy_timestep).\n"
         "K=8 uses ts=arange(T): positions 0..7, so the read position moves from 0 to 7 once the window fills.\n"
         "the CURRENT step's own action slot is always zero (its action is what's being predicted); h196 fixed\n"
         "the HISTORY slots (t<7 here), which previously also carried zeros -- a real train/inference defect.",
         ha="center", fontsize=8, color="#555")
fig.subplots_adjust(wspace=0.15, bottom=0.24, top=0.82)
fig.savefig(os.path.join(OUT, "window_design_demo.png"))
plt.close(fig)
print(f"wrote {os.path.join(OUT, 'window_design_demo.png')}")
