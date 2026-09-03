"""Demonstration of h145's oracle (interpolating) teacher design.

Uses the EXACT construction from h145's worker.py:
    x_start ~ Uniform(domain)
    x_tau = x_start + (x* - x_start) * tau/(T-1),  tau = 0..T-1

Not a re-implementation -- the formula is copied verbatim so the demo cannot
silently drift from what was actually run.
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
from benchmarks import get_benchmark

OUT = os.path.join(H, "..", "results")
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

BENCH = "Borehole_8D"
DIMS = ["rw", "r", "Tu", "Hu", "Tl", "Hl", "L", "Kw"]  # Borehole's own parameter names
XSTAR = np.array([0.15, 100.0, 95090.9777, 1110.0, 116.0, 700.0, 1120.0, 12045.0])
T = 8

hf = get_benchmark(f"{BENCH}_HF")
lo = np.array(hf["domain_min"]); hi = np.array(hf["domain_max"])
f_h = hf["make_objective"]()
opt = float(hf["known_optimal_value"])

def expert_path(rng, x_start):
    """h145's _expert_path, verbatim (worker.py:41-46)."""
    return np.stack([x_start + (XSTAR - x_start) * (t / max(T - 1, 1)) for t in range(T)])

rng = np.random.default_rng(0)
N_DEMO, N_STAT = 12, 400
demo_paths = []
for _ in range(N_DEMO):
    x0 = lo + (hi - lo) * rng.random(8)
    demo_paths.append(expert_path(rng, x0))
demo_paths = np.array(demo_paths)                                    # [N_DEMO, T, 8]
demo_norm = (demo_paths - lo) / (hi - lo)                             # normalised [0,1]

stat_paths = []
for _ in range(N_STAT):
    x0 = lo + (hi - lo) * rng.random(8)
    stat_paths.append(expert_path(rng, x0))
stat_paths = np.array(stat_paths)
import torch
y_stat = f_h(torch.tensor(stat_paths.reshape(-1, 8))).numpy().reshape(N_STAT, T)

fig = plt.figure(figsize=(13, 7.2))
gs = fig.add_gridspec(3, 8, height_ratios=[1, 1, 1.3], hspace=0.9, wspace=0.6)

# ---- Row 1: per-dimension normalised trajectories, small multiples ----
for d in range(8):
    ax = fig.add_subplot(gs[0, d])
    for p in range(N_DEMO):
        ax.plot(range(T), demo_norm[p, :, d], color="#7D3C98", alpha=0.35, lw=1.1)
    ax.axhline((XSTAR[d] - lo[d]) / (hi[d] - lo[d]), color="#C0392B", lw=1.4, ls="--")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(DIMS[d], fontsize=8)
    ax.set_xticks([0, 7]); ax.tick_params(labelsize=6)
    if d == 0: ax.set_ylabel("normalised\nvalue", fontsize=7)
fig.text(0.5, 0.945, r"$x_\tau = x_{start} + (x^* - x_{start})\cdot\tau/(T-1)$,  "
         r"$x_{start}\sim\mathrm{Uniform(domain)}$  —  12 example trajectories, all 8 Borehole dims",
         ha="center", fontsize=10)
fig.text(0.5, 0.925, r"dashed red = $x^*$ (the fixed target every path reaches at $\tau=7$)",
         ha="center", fontsize=8, color="#7F8C8D")

# ---- Row 2: f(x_tau) by tau -- monotone improvement, 400 random starts ----
ax2 = fig.add_subplot(gs[1, :4])
mu, sd = y_stat.mean(0), y_stat.std(0)
ax2.plot(range(T), mu, color="#7D3C98", lw=2.2, marker="o", ms=4)
ax2.fill_between(range(T), mu - sd, mu + sd, color="#7D3C98", alpha=0.18, lw=0)
ax2.axhline(opt, color="#C0392B", lw=1.2, ls="--", label=f"true optimum ({opt:.1f})")
n_mono = int(np.all(np.diff(y_stat, axis=1) > 0, axis=1).sum())
ax2.set_title(f"f(x_tau) by step  —  every step improves on {n_mono}/{N_STAT} random starts",
              fontsize=9)
ax2.set_xlabel(r"$\tau$"); ax2.set_ylabel("objective value (higher = better)")
ax2.legend(fontsize=7, frameon=False, loc="lower right")

# ---- tau=0 vs tau=7: what the DT actually sees at each readout position ----
ax3 = fig.add_subplot(gs[1, 4:])
ax3.hist(y_stat[:, 0], bins=30, color="#2471A3", alpha=0.65, label=r"$\tau=0$ (K=1 readout)")
ax3.hist(y_stat[:, 7], bins=30, color="#C0392B", alpha=0.85, label=r"$\tau=7$ (K=8 readout)")
ax3.axvline(opt, color="black", lw=1.0, ls=":")
ax3.set_title(r"distribution of $f(x_\tau)$ across 400 rollouts", fontsize=9)
ax3.set_xlabel("objective value"); ax3.set_ylabel("count")
ax3.legend(fontsize=7, frameon=False)
ax3.text(0.02, 0.95, f"$\\tau$=0: mean {y_stat[:,0].mean():.0f}, sd {y_stat[:,0].std():.0f}  (noise)",
         transform=ax3.transAxes, fontsize=7, va="top", color="#2471A3")
ax3.text(0.02, 0.86, f"$\\tau$=7: mean {y_stat[:,7].mean():.4f}, sd {y_stat[:,7].std():.2e}  (= x*, exactly)",
         transform=ax3.transAxes, fontsize=7, va="top", color="#C0392B")

# ---- bottom row: the mechanism, stated plainly ----
ax4 = fig.add_subplot(gs[2, :])
ax4.axis("off")
ax4.text(0.5, 0.75,
         "The DT is a per-timestep constant predictor: it emits roughly its teacher's MEAN ACTION at the position it is read.",
         ha="center", fontsize=10.5, transform=ax4.transAxes)
ax4.text(0.5, 0.42,
         r"At $K{=}1$ (default) the DT reads $\tau{=}0$: the mean of 400 UNIFORM RANDOM points $\to$ the box centre.  "
         r"h145 measured $+28.13$ rel%, 0/5 seeds.",
         ha="center", fontsize=9.5, color="#2471A3", transform=ax4.transAxes)
ax4.text(0.5, 0.14,
         r"At $K{=}8$ (window) the DT reads $\tau{=}7$: EVERY path's exact value, zero variance $\to x^*$.  "
         r"h201 measured $0.00$ rel%, 5/5 seeds.",
         ha="center", fontsize=9.5, color="#C0392B", transform=ax4.transAxes)

fig.suptitle(f"How the '{'perfect'}' teacher was designed, and why only its READ POSITION matters ({BENCH})",
             y=1.01, fontsize=12.5)
fig.savefig(os.path.join(OUT, "teacher_design_demo.png"))
plt.close(fig)
print(f"wrote {os.path.join(OUT, 'teacher_design_demo.png')}")
